"""Direct fused-kernel tests independent of model configuration and loading."""
import os
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from minivllm.model_executor.layers import fp8
from test_fp8_weights import block_oracle


RUN_CUDA = os.environ.get("MINIVLLM_RUN_CUDA_FP8_TESTS") == "1" and torch.cuda.is_available()


@unittest.skipUnless(RUN_CUDA, "enable MINIVLLM_RUN_CUDA_FP8_TESTS=1")
class FP8FusedLauncherTest(unittest.TestCase):
    def test_launch_forwards_strides_and_output_without_dense_weight(self):
        from minivllm.model_executor.layers import fp8_triton
        x = torch.zeros(17, 70, device="cuda", dtype=torch.float16)[:, ::2]
        w = torch.zeros(35, 33, device="cuda").to(torch.float8_e4m3fn).t()
        s = torch.ones(5, 5, device="cuda").t()
        y = torch.empty(17, 33, device="cuda", dtype=x.dtype)
        with patch.object(fp8_triton, "fp8_linear_kernel") as kernel:
            fp8_triton.launch_fp8_linear(x, w, s, y, (7, 7))
        self.assertEqual(kernel.__getitem__.call_args.args[0], (2, 2))
        call = kernel.__getitem__.return_value.call_args
        self.assertEqual(call.args[4:7], (17, 33, 35))
        self.assertEqual(call.args[7:13], (*x.stride(), *w.stride(), *s.stride()))
        self.assertEqual(call.args[13:], (7, 7))
        self.assertIs(call.args[0], x)
        self.assertIs(call.args[1], w)
        self.assertIs(call.args[2], s)
        self.assertIs(call.args[3], y)

    def test_empty_batch_does_not_launch(self):
        x = torch.empty(2, 0, 7, device="cuda", dtype=torch.bfloat16)
        w = torch.empty(5, 7, device="cuda", dtype=torch.float8_e4m3fn)
        s = torch.ones(3, 3, device="cuda")
        with patch("minivllm.model_executor.layers.fp8_triton.launch_fp8_linear") as launch:
            y = fp8.fused_fp8_linear(x, w, s, (2, 3))
        self.assertEqual(y.shape, (2, 0, 5))
        self.assertEqual(y.dtype, x.dtype)
        launch.assert_not_called()

    def test_invalid_activation_metadata_fails_before_launch(self):
        w = torch.empty(5, 7, device="cuda", dtype=torch.float8_e4m3fn)
        s = torch.ones(3, 3, device="cuda")
        for x in (torch.zeros(1, 7, device="cuda"),
                  torch.zeros(1, 8, device="cuda", dtype=torch.float16)):
            with self.subTest(dtype=x.dtype, shape=x.shape):
                with patch("minivllm.model_executor.layers.fp8_triton.launch_fp8_linear") as launch:
                    with self.assertRaises(ValueError):
                        fp8.fused_fp8_linear(x, w, s, (2, 3))
                    launch.assert_not_called()


@unittest.skipUnless(RUN_CUDA, "enable MINIVLLM_RUN_CUDA_FP8_TESTS=1")
class FP8FusedKernelTest(unittest.TestCase):
    def compare(self, x, w, s, blocks):
        restored = block_oracle(w, s, blocks).to(device=x.device, dtype=x.dtype)
        expected = F.linear(x, restored)
        old_w, old_s = w.float().clone(), s.clone()
        with patch.object(fp8, "dequantize_fp8_block_weight", side_effect=AssertionError("separate dequant")), \
             patch.object(F, "linear", side_effect=AssertionError("separate GEMM")):
            actual = fp8.fused_fp8_linear(x, w, s, blocks)
        tol = 3e-3 if x.dtype == torch.float16 else 2e-2
        torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
        torch.testing.assert_close(w.float(), old_w, rtol=0, atol=0)
        torch.testing.assert_close(s, old_s, rtol=0, atol=0)
        self.assertEqual(actual.dtype, x.dtype)
        self.assertEqual(actual.shape, (*x.shape[:-1], w.shape[0]))
        self.assertTrue(actual.is_contiguous())

    def test_decode_prefill_tails_and_quant_groups(self):
        torch.manual_seed(91)
        # GEMM tiles (16,32,32) and quantization groups are intentionally different.
        for m, n, k, blocks in ((1, 3, 5, (2, 3)), (7, 35, 67, (7, 13)),
                                 (33, 65, 129, (128, 128))):
            bn, bk = blocks
            for dtype in (torch.float16, torch.bfloat16):
                with self.subTest(m=m, n=n, k=k, dtype=dtype):
                    x = torch.randn(m, k, device="cuda", dtype=dtype)
                    w = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
                    s = torch.rand((n+bn-1)//bn, (k+bk-1)//bk, device="cuda") + .125
                    self.compare(x, w, s, blocks)

    def test_noncontiguous_operands_and_leading_dimensions(self):
        w = torch.randn(67, 35, device="cuda").to(torch.float8_e4m3fn).t()
        s = (torch.rand(6, 5, device="cuda") + .125).t()
        xs = [torch.randn(7, 134, device="cuda", dtype=torch.float16)[:, ::2],
              torch.randn(2, 67, 3, device="cuda", dtype=torch.bfloat16).transpose(1, 2)]
        for x in xs:
            with self.subTest(shape=x.shape):
                self.compare(x, w, s, (7, 13))

    def test_scale_in_fp32_before_half_cast(self):
        x = torch.full((1, 5), .125, device="cuda", dtype=torch.float16)
        w = torch.full((3, 5), 1/256, device="cuda").to(torch.float8_e4m3fn)
        s = torch.full((2, 3), 100000., device="cuda")
        self.compare(x, w, s, (2, 2))

    def test_no_full_weight_allocation_after_warmup(self):
        x = torch.ones(1, 1024, device="cuda", dtype=torch.float16)
        w = torch.ones(1024, 1024, device="cuda").to(torch.float8_e4m3fn)
        s = torch.ones(8, 8, device="cuda")
        warmup = fp8.fused_fp8_linear(x, w, s, (128, 128))
        torch.cuda.synchronize()
        del warmup
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        out = fp8.fused_fp8_linear(x, w, s, (128, 128))
        torch.cuda.synchronize()
        peak_extra = torch.cuda.max_memory_allocated() - baseline
        self.assertLess(peak_extra, w.numel(), "possible full floating weight intermediate")
        torch.testing.assert_close(out, torch.full_like(out, 1024))


if __name__ == "__main__":
    unittest.main()
