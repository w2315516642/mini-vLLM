import unittest
from unittest.mock import patch

from benchmarks.benchmark_generation import measure_continuous
from benchmarks.compare_results import compare_results
from minivllm.outputs import CompletionOutput, RequestOutput
from minivllm.sampling_params import SamplingParams


class FakeLLM:
    def __init__(self, fail=False):
        self.llm_engine = self
        self._generation_active = False
        self.active = {}
        self.admissions = []
        self.batches = []
        self.aborted = []
        self.fail = fail

    def _add_request(self, prompt, params, ids):
        rid = str(len(self.admissions))
        self.admissions.append((rid, len(self.batches)))
        self.active[rid] = (ids, 0)
        return rid

    def has_unfinished_requests(self):
        return bool(self.active)

    def abort_request(self, rid):
        self.aborted.append(rid)
        self.active.pop(rid, None)

    def step(self):
        self.batches.append(tuple(self.active))
        if self.fail:
            raise RuntimeError("injected failure")
        result = []
        for rid, (ids, count) in list(self.active.items()):
            # Request 0 emits a speculative burst; others need three steps.
            count = min(3, count + (3 if rid == "0" else 1))
            finished = count == 3
            completion = CompletionOutput(0, "x" * count, [7] * count, 0., None,
                                          "length" if finished else None)
            result.append(RequestOutput(rid, "", ids, [completion], finished))
            if finished:
                del self.active[rid]
            else:
                self.active[rid] = (ids, count)
        return result


class ContinuousBenchmarkTest(unittest.TestCase):
    def test_comparison_rejects_continuous_against_legacy_batch(self):
        baseline = {"schema_version": 1, "workload": {}, "config": {}}
        candidate = {**baseline, "config": {"load_mode": "continuous"}}
        with self.assertRaisesRegex(ValueError, "load_mode"):
            compare_results(baseline, candidate)

    def test_refill_before_slow_request_finishes_and_include_drain(self):
        llm = FakeLLM()
        with patch("benchmarks.benchmark_generation.time.perf_counter",
                   side_effect=range(100)):
            traces, (start, end) = measure_continuous(
                llm, [[1], [2], [3], [4], [5]], SamplingParams(max_tokens=3), 2)
        self.assertEqual(llm.batches[0], ("0", "1"))
        self.assertEqual(llm.batches[1], ("1", "2"))
        self.assertTrue(all(len(batch) <= 2 for batch in llm.batches))
        self.assertEqual(len(traces), 5)
        self.assertTrue(all(t.output_tokens == 3 for t in traces))
        self.assertEqual([n for _, n in traces[0].events], [3])
        self.assertEqual([n for _, n in traces[1].events], [1, 1, 1])
        self.assertGreater(traces[2].submitted_at, traces[0].finished_at)
        self.assertLess(start, min(t.submitted_at for t in traces))
        self.assertGreater(end, max(t.finished_at for t in traces))
        self.assertFalse(llm.active)
        self.assertFalse(llm._generation_active)

    def test_failure_aborts_outstanding_requests(self):
        llm = FakeLLM(fail=True)
        with self.assertRaisesRegex(RuntimeError, "injected failure"):
            measure_continuous(llm, [[1], [2]], SamplingParams(max_tokens=3), 2)
        self.assertEqual(set(llm.aborted), {"0", "1"})
        self.assertFalse(llm._generation_active)

    def test_wrong_output_length_fails(self):
        llm = FakeLLM()
        with self.assertRaisesRegex(RuntimeError, "token count differs"):
            measure_continuous(llm, [[1], [2]], SamplingParams(max_tokens=4), 2)
        self.assertFalse(llm.active)

    def test_busy_engine_is_not_modified(self):
        llm = FakeLLM()
        llm._generation_active = True
        with self.assertRaisesRegex(RuntimeError, "idle engine"):
            measure_continuous(llm, [[1]], SamplingParams(max_tokens=3), 1)
        self.assertTrue(llm._generation_active)
        self.assertFalse(llm.admissions)


if __name__ == "__main__":
    unittest.main()
