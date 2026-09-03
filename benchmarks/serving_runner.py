"""One step owner per engine, with independent P/D progress for benchmarks.

These pumps only drive existing RPC methods. They are not an HTTP service and
must have exclusive ownership of their engine endpoints for the entire run.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
import threading
import time

from benchmarks.streaming_metrics import RequestTrace


class Collector:
    def __init__(self, workload, origin):
        self.traces = {item.request_id: RequestTrace(
            item.request_id, len(item.token_ids), origin + item.arrival_s, item.cohort)
            for item in workload}
        if len(self.traces) != len(workload):
            raise ValueError("Workload request IDs must be unique")
        self.lock = threading.Lock()
        self.expected_tokens = {item.request_id: item.output_len for item in workload}

    def observe(self, outputs, timestamp):
        with self.lock:
            for output in outputs:
                if output.request_id not in self.traces:
                    raise RuntimeError("Another client is driving these benchmark engines")
                self.traces[output.request_id].observe(output, timestamp, cumulative=True)


def _submit(engine, item, params_factory):
    engine.add_request(item.request_id, None, params_factory(item.output_len), item.token_ids)


def _abort(engines, workload):
    # Preserve the original failure if a timed-out connection is already closed.
    for engine in engines:
        for item in workload:
            with suppress(Exception):
                engine.abort_request(item.request_id)


def run_unified(engines, workload, params_factory, timeout_s=600.):
    """Round-robin across equal replicas; each engine handles a request batch."""
    if not workload or not engines:
        raise ValueError("A workload and at least one engine are required")
    if any(engine.has_unfinished_requests() for engine in engines):
        raise RuntimeError("Benchmark engines must start idle")
    origin = time.perf_counter()
    deadline = origin + timeout_s
    collector = Collector(workload, origin)
    stop = threading.Event()

    def pump(engine, items):
        items = sorted(items, key=lambda item: item.arrival_s)
        index, pending = 0, set()
        try:
            while (index < len(items) or pending) and not stop.is_set():
                if time.perf_counter() >= deadline:
                    raise TimeoutError("Unified benchmark exceeded its total timeout")
                while index < len(items) and time.perf_counter() >= origin + items[index].arrival_s:
                    item = items[index]
                    _submit(engine, item, params_factory)
                    pending.add(item.request_id)
                    index += 1
                if pending:
                    outputs = engine.step()
                    collector.observe(outputs, time.perf_counter())
                    pending.difference_update(x.request_id for x in outputs if x.is_finished())
                elif index < len(items):
                    stop.wait(max(0, min(deadline, origin + items[index].arrival_s) - time.perf_counter()))
        except BaseException:
            stop.set()
            raise

    try:
        with ThreadPoolExecutor(max_workers=len(engines)) as pool:
            futures = [pool.submit(pump, engine, workload[i::len(engines)])
                       for i, engine in enumerate(engines)]
            try:
                for future in futures:
                    future.result()
            finally:
                stop.set()
    except BaseException:
        stop.set()
        _abort(engines, workload)
        raise
    return _finish(collector, origin)


def run_pd(prefill, decode, workload, params_factory, bridge, timeout_s=600.):
    """P can prefill/transfer while D decodes previously activated requests."""
    if not workload:
        raise ValueError("A non-empty workload is required")
    if prefill.has_unfinished_requests() or decode.has_unfinished_requests():
        raise RuntimeError("Benchmark engines must start idle")
    origin = time.perf_counter()
    deadline = origin + timeout_s
    collector = Collector(workload, origin)
    stop = threading.Event()
    condition = threading.Condition()
    # Register before activation, so a fast D completion cannot race ahead of
    # P's transfer return. False entries alone never wake an idle decode pump.
    active = {}
    prefill_done = False
    transferred = []

    def pump_prefill():
        nonlocal prefill_done
        items = sorted(workload, key=lambda item: item.arrival_s)
        index, pending = 0, set()
        try:
            while (index < len(items) or pending) and not stop.is_set():
                if time.perf_counter() >= deadline:
                    raise TimeoutError("PD benchmark exceeded its total timeout")
                while index < len(items) and time.perf_counter() >= origin + items[index].arrival_s:
                    item = items[index]
                    _submit(prefill, item, params_factory)
                    pending.add(item.request_id)
                    index += 1
                if not pending:
                    if index < len(items):
                        stop.wait(max(0, min(deadline, origin + items[index].arrival_s) - time.perf_counter()))
                    continue
                outputs = prefill.step()
                # Timestamp P's first token before reservation or transfer.
                collector.observe(outputs, time.perf_counter())
                pending.difference_update(x.request_id for x in outputs if x.is_finished())
                for handoff in prefill.pop_prefill_handoffs():
                    request_id = handoff.request_id
                    if request_id not in pending:
                        raise RuntimeError("Unexpected or duplicate handoff")
                    with condition:
                        active[request_id] = False
                    started = time.perf_counter()
                    bridge.transfer(handoff)
                    with collector.lock:
                        collector.traces[request_id].handoff_s = time.perf_counter() - started
                    transferred.append(request_id)
                    pending.remove(request_id)
                    with condition:
                        if request_id in active:
                            active[request_id] = True
                        condition.notify_all()
        except BaseException:
            stop.set()
            raise
        finally:
            with condition:
                prefill_done = True
                condition.notify_all()

    def pump_decode():
        try:
            while not stop.is_set():
                if time.perf_counter() >= deadline:
                    raise TimeoutError("PD benchmark exceeded its total timeout")
                with condition:
                    ready = condition.wait_for(
                        lambda: any(active.values()) or prefill_done or stop.is_set(),
                        timeout=max(0, deadline - time.perf_counter()))
                    if not ready:
                        raise TimeoutError("PD benchmark exceeded its total timeout")
                    if stop.is_set() or (prefill_done and not active):
                        return
                outputs = decode.step()
                collector.observe(outputs, time.perf_counter())
                with condition:
                    for output in outputs:
                        if output.is_finished():
                            active.pop(output.request_id, None)
        except BaseException:
            stop.set()
            with condition:
                condition.notify_all()
            raise

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(pump_prefill), pool.submit(pump_decode)]
            try:
                for future in futures:
                    future.result()
            finally:
                stop.set()
                with condition:
                    condition.notify_all()
        for request_id in transferred:
            bridge.coordinator.finish(request_id)
    except BaseException:
        stop.set()
        _abort([prefill, decode], workload)
        raise
    return _finish(collector, origin)


def _finish(collector, origin):
    traces = list(collector.traces.values())
    if any(t.finished_at is None or not t.events for t in traces):
        raise RuntimeError("Benchmark ended with missing outputs")
    if any(t.output_tokens != collector.expected_tokens[t.request_id] for t in traces):
        raise RuntimeError("Generated token count differs from the fixed benchmark workload")
    return traces, (origin, max(t.finished_at for t in traces))
