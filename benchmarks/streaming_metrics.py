"""Client-observed token timing. No CUDA synchronization or hot-path I/O."""

from dataclasses import dataclass, field
import hashlib
import json
import math
import statistics


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None}

    def percentile(q):
        # Nearest-rank percentile, shared by every reported metric.
        return values[max(0, math.ceil(q * len(values)) - 1)]

    return {"count": len(values), "mean": statistics.mean(values),
            "p50": percentile(.50), "p95": percentile(.95), "p99": percentile(.99)}


@dataclass
class RequestTrace:
    request_id: str
    input_tokens: int
    submitted_at: float
    cohort: str = "default"
    events: list = field(default_factory=list)
    output_tokens: int = 0
    finished_at: float | None = None
    handoff_s: float | None = None
    token_ids: list = field(default_factory=list, repr=False)

    def observe(self, output, now, *, cumulative):
        if self.finished_at is not None:
            raise ValueError(f"Output after completion: {self.request_id}")
        if output.request_id != self.request_id or len(output.outputs) != 1:
            raise ValueError("Benchmark requires one completion per known request")
        count = len(output.outputs[0].token_ids)
        added = count - self.output_tokens if cumulative else count
        if added < 0:
            raise ValueError("Cumulative token count went backwards")
        previous = self.events[-1][0] if self.events else self.submitted_at
        if now < previous:
            raise ValueError("Non-monotonic observation timestamp")
        if added:
            self.events.append((now, added))
            self.token_ids.extend(output.outputs[0].token_ids[-added:])
            self.output_tokens += added
        if output.is_finished():
            self.finished_at = now

    def record(self, origin):
        first = self.events[0][0] if self.events else None
        last = self.events[-1][0] if self.events else None
        decode_s = last - first if len(self.events) > 1 else None
        tpot_s = (decode_s / (self.output_tokens - 1)
                  if decode_s is not None and self.output_tokens > 1 else None)
        return {
            "request_id": self.request_id, "cohort": self.cohort,
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "submitted_s": self.submitted_at - origin,
            "finished_s": None if self.finished_at is None else self.finished_at - origin,
            "events": [{"time_s": t - origin, "new_tokens": n} for t, n in self.events],
            "ttft_ms": None if first is None else 1000 * (first - self.submitted_at),
            "e2e_ms": None if self.finished_at is None else 1000 * (self.finished_at - self.submitted_at),
            "tpot_ms": None if tpot_s is None else 1000 * tpot_s,
            "decode_tokens_per_s": 1 / tpot_s if tpot_s and tpot_s > 0 else None,
            "handoff_ms": None if self.handoff_s is None else 1000 * self.handoff_s,
            "output_token_sha256": hashlib.sha256(json.dumps(self.token_ids).encode()).hexdigest(),
        }


def summarize(traces, windows):
    """Windows exclude model loading/warmup and include queueing and drain."""
    traces = list(traces)
    origin = min(start for start, _ in windows)
    records = [trace.record(origin) for trace in traces]
    elapsed = sum(end - start for start, end in windows)
    if elapsed <= 0:
        raise ValueError("Measurement duration must be positive")
    gaps = [1000 * (b[0] - a[0]) for t in traces
            for a, b in zip(t.events, t.events[1:])]
    burst = any(n != 1 for t in traces for _, n in t.events)
    metrics = {
        key: distribution(r[key] for r in records if r[key] is not None)
        for key in ("ttft_ms", "e2e_ms", "tpot_ms", "decode_tokens_per_s", "handoff_ms")
    }
    metrics.update({
        "requests": len(records),
        "completed_requests": sum(t.finished_at is not None for t in traces),
        "measurement_seconds": elapsed,
        "output_tokens": sum(t.output_tokens for t in traces),
        "output_tokens_per_s": sum(t.output_tokens for t in traces) / elapsed,
        "requests_per_s": sum(t.finished_at is not None for t in traces) / elapsed,
        "inter_update_ms": distribution(gaps),
        # Never turn a speculative burst into invented zero-latency tokens.
        "itl_ms": None if burst else distribution(gaps),
        "has_multi_token_updates": burst,
    })
    return metrics, records


def speculative_delta(before, after):
    totals = {name: 0 for name in (
        "verification_rounds", "verified_draft_tokens", "accepted_draft_tokens")}
    if len(before) != len(after):
        raise ValueError("Runtime snapshot counts differ")
    for start, end in zip(before, after):
        for name in totals:
            delta = end["speculative"][name] - start["speculative"][name]
            if delta < 0:
                raise ValueError("Runtime counters reset during measurement")
            totals[name] += delta
    rounds = totals["verification_rounds"]
    verified = totals["verified_draft_tokens"]
    accepted = totals["accepted_draft_tokens"]
    return {**totals,
            "accepted_tokens_per_round": accepted / rounds if rounds else None,
            "verified_tokens_per_round": verified / rounds if rounds else None,
            "acceptance_rate": accepted / verified if verified else None}
