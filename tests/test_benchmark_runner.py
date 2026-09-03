import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from benchmarks.benchmark_generation import measure_batch
from benchmarks.benchmark_utils import WorkItem
from benchmarks.serving_runner import run_pd, run_unified


class FakeOutput:
    def __init__(self, request_id, count, finished=False):
        self.request_id = request_id
        self.prompt_token_ids = [1, 2]
        self.outputs = [SimpleNamespace(token_ids=list(range(count)))]
        self.finished = finished

    def is_finished(self):
        return self.finished


output = FakeOutput


class Engine:
    """Small stateful engine, optionally acting as P, with no GPU dependency."""
    def __init__(self, prefill=False):
        self.prefill = prefill
        self.pending = {}
        self.handoffs = []
        self.lock = threading.RLock()
        self.steps = 0
        self.aborted = []
        self.fail = False

    def has_unfinished_requests(self):
        with self.lock:
            return bool(self.pending)

    def add_request(self, request_id, prompt, params, token_ids):
        with self.lock:
            self.pending[request_id] = [0, params]

    def step(self):
        with self.lock:
            if self.fail:
                raise ValueError("worker failed")
            time.sleep(.001)
            self.steps += 1
            outputs = []
            for request_id, state in list(self.pending.items()):
                state[0] += 1
                finished = state[0] == state[1]
                outputs.append(output(request_id, state[0], finished))
                if self.prefill and not finished:
                    self.handoffs.append(SimpleNamespace(request_id=request_id, state=list(state)))
                if self.prefill or finished:
                    del self.pending[request_id]
            return outputs

    def pop_prefill_handoffs(self):
        with self.lock:
            result, self.handoffs = self.handoffs, []
            return result

    def abort_request(self, request_id):
        with self.lock:
            self.aborted.append(request_id)
            self.pending.pop(request_id, None)


def fake_bridge(prefill, decode):
    bridge = Mock()

    def transfer(handoff):
        time.sleep(.002)
        with decode.lock:
            decode.pending[handoff.request_id] = handoff.state
        # D can finish this request while P is still returning from transfer.
        time.sleep(.004)

    bridge.transfer.side_effect = transfer
    return bridge


class RunnerTest(unittest.TestCase):
    def test_unified_replicas_route_outputs_and_preserve_arrivals(self):
        engines = [Engine(), Engine()]
        workload = [WorkItem(str(i), [1, 2], 4, i * .003) for i in range(8)]
        traces, window = run_unified(engines, workload, int)
        self.assertEqual(len(traces), 8)
        self.assertTrue(all(t.output_tokens == 4 for t in traces))
        self.assertAlmostEqual(traces[3].submitted_at - window[0], .009)
        self.assertTrue(all(not engine.pending for engine in engines))

    def test_pd_routes_many_requests_and_counts_first_token_once(self):
        p, d = Engine(prefill=True), Engine()
        workload = [WorkItem(str(i), [1, 2], 3, i * .002) for i in range(8)]
        bridge = fake_bridge(p, d)
        traces, _ = run_pd(p, d, workload, int, bridge)
        self.assertTrue(all(t.output_tokens == 3 for t in traces))
        self.assertTrue(all(t.handoff_s > 0 for t in traces))
        self.assertTrue(all(t.events[0][1] == 1 for t in traces))
        self.assertEqual(bridge.coordinator.finish.call_count, 8)
        self.assertFalse(p.pending or d.pending)

    def test_decode_progresses_while_next_prefill_transfer_is_running(self):
        p, d = Engine(prefill=True), Engine()
        bridge = fake_bridge(p, d)
        original = bridge.transfer.side_effect
        observed = []

        def transfer(handoff):
            if handoff.request_id == "later":
                before = d.steps
                time.sleep(.02)
                observed.append(d.steps > before)
            original(handoff)

        bridge.transfer.side_effect = transfer
        traces, _ = run_pd(p, d, [WorkItem("first", [1], 100),
                                  WorkItem("later", [1], 2, .015)], int, bridge)
        self.assertEqual(observed, [True])
        self.assertEqual([t.output_tokens for t in traces], [100, 2])

    def test_one_token_request_finishes_on_p_without_handoff(self):
        p, d = Engine(prefill=True), Engine()
        bridge = fake_bridge(p, d)
        traces, _ = run_pd(p, d, [WorkItem("a", [1], 1)], int, bridge)
        self.assertIsNone(traces[0].handoff_s)
        bridge.transfer.assert_not_called()

    def test_failure_releases_requests_and_propagates(self):
        p, d = Engine(prefill=True), Engine()
        d.fail = True
        with self.assertRaisesRegex(ValueError, "worker failed"):
            run_pd(p, d, [WorkItem("a", [1], 3)], int, fake_bridge(p, d))
        self.assertIn("a", p.aborted)
        self.assertIn("a", d.aborted)
        self.assertFalse(d.pending)

    def test_transfer_failure_wakes_idle_decode_pump(self):
        p, d = Engine(prefill=True), Engine()
        bridge = Mock()
        bridge.transfer.side_effect = RuntimeError("transfer failed")
        with self.assertRaisesRegex(RuntimeError, "transfer failed"):
            run_pd(p, d, [WorkItem("a", [1], 3)], int, bridge)

    def test_streaming_batch_uses_delta_counts(self):
        class LLM:
            def generate_stream(self, **kwargs):
                yield output("a", 1)
                yield output("b", 1)
                yield output("a", 3, True)
                yield output("b", 3, True)

        traces, _ = measure_batch(LLM(), [[1], [2]], SimpleNamespace(max_tokens=4))
        self.assertEqual([t.output_tokens for t in traces], [4, 4])

    def test_early_finished_generation_does_not_count_as_a_faster_fixed_workload(self):
        class LLM:
            def generate_stream(self, **kwargs):
                yield output("a", 1, True)

        with self.assertRaisesRegex(RuntimeError, "fixed benchmark workload"):
            measure_batch(LLM(), [[1]], SimpleNamespace(max_tokens=8))

    def test_no_progress_times_out_and_releases_requests(self):
        engine = Engine()
        engine.step = lambda: []
        with self.assertRaisesRegex(TimeoutError, "total timeout"):
            run_unified([engine], [WorkItem("a", [1], 3)], int, timeout_s=.01)
        self.assertIn("a", engine.aborted)

    def test_pd_no_progress_wakes_waiting_decode_on_timeout(self):
        p, d = Engine(prefill=True), Engine()
        p.step = lambda: []
        with self.assertRaisesRegex(TimeoutError, "total timeout"):
            run_pd(p, d, [WorkItem("a", [1], 3)], int, fake_bridge(p, d), timeout_s=.01)
        self.assertIn("a", p.aborted)


if __name__ == "__main__":
    unittest.main()
