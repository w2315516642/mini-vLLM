"""Sequential B1 HTTP comparison against an otherwise idle upstream vLLM."""

import argparse
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from benchmarks.benchmark_utils import WorkItem, prepare_prompts, workload_info
from benchmarks.streaming_metrics import distribution


def run_closed_loop(prompts, concurrency, generate):
    """Keep at most concurrency HTTP requests in flight, including final drain."""
    if concurrency < 1 or len(prompts) < concurrency:
        raise ValueError("Need at least concurrency prompts")
    rows, events, pending = [], [], {}
    next_index = 0
    origin = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        def submit(index):
            def task():
                submitted = time.perf_counter() - origin
                result = generate(prompts[index])
                return {"request_id": str(index), "submitted_s": submitted,
                        "finished_s": time.perf_counter() - origin, **result}
            pending[pool.submit(task)] = index

        while next_index < concurrency:
            submit(next_index)
            next_index += 1
        events.append({"time_s": time.perf_counter() - origin, "inflight": len(pending)})
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                rows.append(future.result())
                del pending[future]
                if next_index < len(prompts):
                    submit(next_index)
                    next_index += 1
            events.append({"time_s": time.perf_counter() - origin, "inflight": len(pending)})
    return sorted(rows, key=lambda r: int(r["request_id"])), time.perf_counter() - origin, events


COUNTERS = {
    "verification_rounds": "vllm:spec_decode_num_drafts_total",
    "verified_draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted_draft_tokens": "vllm:spec_decode_num_accepted_tokens_total",
}


def parse_counters(body, model):
    from prometheus_client.parser import text_string_to_metric_families

    values = {}
    for family in text_string_to_metric_families(body):
        for sample in family.samples:
            if sample.labels.get("model_name") != model:
                continue
            if sample.name in COUNTERS.values():
                key = next(k for k, v in COUNTERS.items() if v == sample.name)
            elif sample.name == "vllm:spec_decode_num_accepted_tokens_per_pos_total":
                key = "position_" + sample.labels["position"]
            else:
                continue
            values[key] = values.get(key, 0) + sample.value
    return values


def counter_delta(before, after, mode):
    delta = {k: after.get(k, 0) - before.get(k, 0) for k in before.keys() | after.keys()}
    if any(v < 0 for v in delta.values()):
        raise ValueError("Server counters reset during measurement")
    rounds = delta.get("verification_rounds", 0)
    proposed = delta.get("verified_draft_tokens", 0)
    accepted = delta.get("accepted_draft_tokens", 0)
    if mode == "dspark" and (rounds <= 0 or proposed <= 0):
        raise ValueError("No DSpark counter increments; check server mode and metrics delay")
    if mode == "target" and any(delta.values()):
        raise ValueError("Target-only run unexpectedly performed speculative decoding")
    if not 0 <= accepted <= proposed:
        raise ValueError("Inconsistent speculative counters")
    delta["accepted_tokens_per_round"] = accepted / rounds if rounds else None
    delta["acceptance_rate"] = accepted / proposed if proposed else None
    return delta


def consume_stream(lines, input_len, output_len, started, clock=time.perf_counter):
    tokens, text, updates = [], [], []
    usage, first, last, done, finish = None, None, None, False, None
    for line in lines:
        if not line or not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            done = True
            break
        data = json.loads(payload)
        if "error" in data:
            raise RuntimeError(data["error"])
        if data.get("usage"):
            usage = data["usage"]
        for choice in data.get("choices", []):
            ids = choice.get("token_ids") or []
            if choice.get("text") and not ids:
                raise ValueError("Content without token IDs; server must support return_token_ids")
            if ids:
                now = clock()
                first = now if first is None else first
                last = now
                tokens.extend(ids)
                text.append(choice.get("text", ""))
                updates.append({"elapsed_ms": (now - started) * 1000, "tokens": len(ids)})
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    if not done or first is None or finish != "length":
        raise ValueError("Incomplete stream or unexpected finish reason")
    if usage is None or usage["prompt_tokens"] != input_len or usage["completion_tokens"] != output_len:
        raise ValueError("Server usage differs from fixed workload")
    if len(tokens) != output_len:
        raise ValueError("Returned token IDs differ from usage count")
    return {"ttft_ms": (first - started) * 1000,
            "e2e_ms": (last - started) * 1000,
            "tpot_ms": (last - first) * 1000 / (output_len - 1),
            "token_ids": tokens, "text": "".join(text), "updates": updates}


def main():
    import requests
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="target")
    parser.add_argument("--mode", choices=("target", "dspark"), required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--concurrency", type=int, choices=(1, 16), default=1)
    parser.add_argument("--framework", choices=("vllm", "mini"), default="vllm")
    parser.add_argument("--profile", action="store_true", help="Instrumented run, not a performance baseline")
    parser.add_argument("--metrics-settle-seconds", type=float, default=10)
    args = parser.parse_args()
    if args.count < args.concurrency or args.metrics_settle_seconds < 0:
        parser.error("count must be positive and settle time nonnegative")
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    prompts = prepare_prompts(tokenizer, args.count, 512, dataset=args.dataset, seed=42)
    if len({tuple(p) for p in prompts}) != args.count:
        raise ValueError("Need count distinct prompts; refusing resampling")
    warmup = prepare_prompts(tokenizer, args.concurrency, 512, synthetic=True, seed=10042)
    url = args.url.rstrip("/")
    with requests.Session() as session:
        def get(path):
            response = session.get(url + path, timeout=30)
            response.raise_for_status()
            return response

        def snapshot():
            # Stats may reach Prometheus after the last HTTP response. Keep this
            # wait outside the timed workload; use a dedicated idle server.
            time.sleep(args.metrics_settle_seconds)
            return parse_counters(get("/metrics").text, args.model)

        def generate(ids):
            started = time.perf_counter()
            # A session per request avoids sharing mutable Session state across threads.
            with requests.post(url + "/v1/completions", json={
                "model": args.model, "prompt": ids, "temperature": 0,
                "top_p": 1, "top_k": -1, "max_tokens": 128, "ignore_eos": True,
                "seed": 42, "stream": True, "return_token_ids": True,
                "stream_options": {"include_usage": True},
            }, stream=True, timeout=(30, 600)) as response:
                response.raise_for_status()
                return consume_stream(response.iter_lines(chunk_size=None, decode_unicode=True),
                                      len(ids), 128, started)

        get("/health")
        version = get("/version").json()
        for _ in range(2):
            run_closed_loop(warmup, args.concurrency, generate)
        before = snapshot()
        if args.profile:
            response = session.post(url + "/start_profile", timeout=600)
            response.raise_for_status()
        try:
            rows, elapsed, occupancy = run_closed_loop(prompts, args.concurrency, generate)
        finally:
            if args.profile:
                response = session.post(url + "/stop_profile", timeout=600)
                response.raise_for_status()
        after = snapshot()
    result = {"config": vars(args), "server_version": version,
              "workload": workload_info([WorkItem(str(i), p, 128) for i, p in enumerate(prompts)]),
              "metrics": {k: distribution([r[k] for r in rows])
                          for k in ("ttft_ms", "e2e_ms", "tpot_ms")},
              "counters_before": before, "counters_after": after,
              "speculative": counter_delta(before, after, args.mode), "requests": rows,
              "client_occupancy": occupancy,
              "notes": ["Client inflight counts are not GPU active batch sizes.",
                        "Client HTTP timings include transport overhead; no per-token ITL inferred.",
                        "Requires an otherwise idle server; settle delay is not a stats flush barrier.",
                        "Mode is a client assertion, not a complete server configuration attestation."]}
    result["metrics"].update(output_tokens=args.count * 128, measurement_seconds=elapsed,
                             output_tokens_per_s=args.count * 128 / elapsed,
                             completed_requests=len(rows))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({"output": str(out), "metrics": result["metrics"],
                      "speculative": result["speculative"]}, indent=2))


if __name__ == "__main__":
    main()
