import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch


spec = importlib.util.spec_from_file_location("profiling", Path(__file__).parents[1]
                                            / "minivllm/profiling.py")
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def metadata(prefill=False, speculative=False):
    return NS(speculative_seq_ids=[1] if speculative else [],
              prompt_seq_ids=[1] if prefill or speculative else [],
              generation_seq_ids=[] if prefill or speculative else [1],
              num_valid_tokens=4 if speculative else 1,
              prompt_lens=[4] if prefill or speculative else [])


class DecodeCaptureTests(unittest.TestCase):
    def setUp(self):
        self.cuda = NS(synchronize=Mock(), profiler=NS(start=Mock(), stop=Mock()),
                       nvtx=NS(range_push=Mock(), range_pop=Mock()))
        self.patch = patch.dict("sys.modules", {"torch": NS(cuda=self.cuda)})
        self.patch.start()
        p._active = False

    def tearDown(self):
        p._active = False
        self.patch.stop()

    def test_window_skips_prefill_and_counts_verification(self):
        capture = p.DecodeCapture(1, 2)
        output = {1: NS(output_token_ids=[1, 2])}
        capture.begin_step(metadata(prefill=True))
        capture.end_step(output)
        capture.begin_step(metadata())
        capture.end_step(output)
        self.cuda.profiler.start.assert_not_called()
        for _ in range(3):
            capture.begin_step(metadata(speculative=True))
            capture.end_step(output)
        result = capture.close()
        self.assertTrue(result["complete"])
        self.assertEqual(len(result["records"]), 2)
        self.assertEqual(result["records"][0]["M"], 4)
        self.assertEqual(result["records"][0]["T"], 4)
        self.assertEqual(result["records"][0]["produced_tokens"], 2)
        self.cuda.profiler.start.assert_called_once()
        self.cuda.profiler.stop.assert_called_once()
        self.assertEqual(self.cuda.synchronize.call_count, 2)

    def test_exception_balances_ranges_and_stops(self):
        capture = p.DecodeCapture(0, 20)

        @p.capture_worker_step
        def step(worker):
            capture.begin_step(metadata())
            with p.nvtx_range("inner"):
                raise RuntimeError("failure")

        with self.assertRaises(RuntimeError):
            step(NS(_decode_capture=capture))
        self.assertFalse(p._active)
        self.assertEqual(self.cuda.nvtx.range_push.call_count, 2)
        self.assertEqual(self.cuda.nvtx.range_pop.call_count, 2)
        self.cuda.profiler.stop.assert_called_once()
        self.assertFalse(capture.close()["complete"])

    def test_new_prefill_closes_incomplete_window(self):
        capture = p.DecodeCapture(0, 20)
        capture.begin_step(metadata())
        capture.end_step({})
        capture.begin_step(metadata(prefill=True))
        capture.begin_step(metadata())
        self.assertEqual(len(capture.records), 1)
        self.assertFalse(capture.close()["complete"])

    def test_disabled_marker_does_not_inspect_arguments(self):
        class Argument:
            @property
            def shape(self):
                raise AssertionError("Must not inspect shapes when disabled")
        fn = p.nvtx_function("linear")(lambda x: x)
        value = Argument()
        self.assertIs(fn(value), value)
        self.cuda.nvtx.range_push.assert_not_called()

    def test_invalid_window(self):
        for skip, steps in [(-1, 1), (0, 0)]:
            with self.assertRaises(ValueError):
                p.DecodeCapture(skip, steps)


if __name__ == "__main__":
    unittest.main()
