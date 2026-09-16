import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location("profiling_test", Path(__file__).parents[1]
                                            / "minivllm/profiling.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ExternalNvtxTest(unittest.TestCase):
    def tearDown(self):
        module.set_external_nvtx(False)
        module._active = False

    def test_disabled_does_not_touch_cuda(self):
        cuda = Mock()
        with patch.dict("sys.modules", {"torch": SimpleNamespace(cuda=cuda)}):
            fn = module.nvtx_function("linear")(lambda: 7)
            self.assertEqual(fn(), 7)
            self.assertEqual(cuda.mock_calls, [])

    def test_nested_ranges_balance_on_error(self):
        cuda = Mock()
        metadata = SimpleNamespace(prompt_seq_ids=[1], generation_seq_ids=[],
                                   speculative_seq_ids=[1], num_valid_tokens=4)
        @module.nvtx_function("target_model")
        def model(**kwargs):
            with module.nvtx_range("state_snapshot"):
                raise ValueError("test")
        with patch.dict("sys.modules", {"torch": SimpleNamespace(cuda=cuda)}):
            module.set_external_nvtx(True)
            with self.assertRaises(ValueError):
                model(input_metadata=metadata)
            self.assertEqual(cuda.nvtx.range_push.call_count, 2)
            self.assertEqual(cuda.nvtx.range_pop.call_count, 2)
            self.assertIn("B=1 M=4 verify_requests=1", cuda.nvtx.range_push.call_args_list[0].args[0])
            cuda.synchronize.assert_not_called()
            module.set_external_nvtx(False)
            with module.nvtx_range("off"):
                pass
            self.assertEqual(cuda.nvtx.range_push.call_count, 2)

    def test_external_worker_range(self):
        cuda = Mock()
        fn = module.capture_worker_step(lambda worker: 9)
        with patch.dict("sys.modules", {"torch": SimpleNamespace(cuda=cuda)}):
            module.set_external_nvtx(True)
            self.assertEqual(fn(SimpleNamespace()), 9)
            cuda.nvtx.range_push.assert_called_once_with("worker_step")
            cuda.nvtx.range_pop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
