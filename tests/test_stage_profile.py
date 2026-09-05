import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from benchmarks.summarize_stage_profile import summarize

# Load the standalone helper without importing the CUDA inference package.
spec = importlib.util.spec_from_file_location("stage_profile", Path(__file__).parents[1]
    / "minivllm/worker/stage_profile.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class StageProfileTests(unittest.TestCase):
    def test_events_are_deferred_and_bounded(self):
        cuda = SimpleNamespace(Event=Mock(side_effect=lambda **kw: Mock(
            elapsed_time=Mock(return_value=2.0))), synchronize=Mock())
        with patch.dict("sys.modules", {"torch": SimpleNamespace(cuda=cuda)}):
            profile = module.StageProfile(1)
            profile.begin()
            profile.counts(prefill_requests=0, rejected_requests=2)
            profile.mark("target_model")
            profile.end()
            profile.begin()
            profile.mark("ignored")
            profile.end()
            cuda.synchronize.assert_not_called()
            self.assertEqual(cuda.Event.call_count, 2)
            result = profile.finish()
            cuda.synchronize.assert_called_once()
        self.assertEqual(len(result["steps"]), 1)
        self.assertTrue(result["limit_reached"])
        self.assertEqual(result["steps"][0]["stages"]["target_model"]["stream_ms"], 2)
        self.assertEqual(result["steps"][0]["counts"]["rejected_requests"], 2)

    def test_empty_and_invalid(self):
        with self.assertRaises(ValueError):
            module.StageProfile(0)
        cuda = SimpleNamespace(synchronize=Mock())
        with patch.dict("sys.modules", {"torch": SimpleNamespace(cuda=cuda)}):
            self.assertEqual(module.StageProfile(2).finish()["steps"], [])

    def test_summary_separates_prefill(self):
        rows = [{"counts": {"prefill_requests": n, "replayed_requests": n},
                 "stages": {"target_model": {"host_ms": 1, "stream_ms": 2}}}
                for n in (1, 0, 0)]
        result = summarize(rows)
        self.assertEqual(result["decode"]["steps"], 2)
        self.assertEqual(result["decode"]["stage_totals"]["target_model"]["stream_ms"], 4)
        self.assertEqual(result["prefill_or_mixed"]["count_totals"]["replayed_requests"], 1)


if __name__ == "__main__":
    unittest.main()
