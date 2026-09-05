"""Numerical, state-continuation and buffer contracts for the inference kernels."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from minivllm.model_executor.parallel_utils.tensor_parallel.layers import (
    _linear, dequantize_fp8_block_weight,
)
from minivllm.model_executor.layers.gated_delta_net_cuda import (
    gated_delta_rule_prefill, gated_delta_rule_varlen, prepare_gated_delta_qk,
)


RUN = os.environ.get("MINIVLLM_RUN_FUSED_TESTS") == "1" and torch.cuda.is_available()


@unittest.skipUnless(RUN, "enable MINIVLLM_RUN_FUSED_TESTS on CUDA")
class FusedLinearTest(unittest.TestCase):
    def test_shapes_scales_bias_and_output_buffer(self):
        from minivllm.model_executor.layers.fp8_linear import fp8_linear
        torch.manual_seed(481)
        for dtype in (torch.float16, torch.bfloat16):
            for m, n, k, block in ((1, 129, 1025, (128, 128)),
                                    (16, 256, 2048, (128, 128)),
                                    (33, 65, 97, (32, 16)),
                                    (128, 128, 256, (128, 128))):
                with self.subTest(dtype=dtype, shape=(m, n, k)):
                    # Strided activations exercise TP-like input layouts.
                    x = torch.randn(m, k * 2, device="cuda", dtype=dtype)[:, ::2]
                    w = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
                    s = torch.rand((n + block[0] - 1) // block[0],
                                   (k + block[1] - 1) // block[1], device="cuda") * .1
                    bias = torch.randn(n, device="cuda", dtype=dtype)
                    expected = F.linear(x, dequantize_fp8_block_weight(w, s, block, dtype), bias)
                    out = torch.empty(m, n, device="cuda", dtype=dtype)
                    actual = fp8_linear(x, w, s, block, bias, out=out)
                    self.assertEqual(actual.data_ptr(), out.data_ptr())
                    self.assertEqual(w.dtype, torch.float8_e4m3fn)
                    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=3e-2)

    def test_all_finite_e4m3_encodings_and_tiny_scales(self):
        from minivllm.model_executor.layers.fp8_linear import fp8_linear
        # Decode subnormals, both signs and max finite values on Ampere too.
        bits = torch.arange(256, device="cuda").to(torch.uint8)
        bits[127] = 0
        bits[255] = 128
        w = bits.view(torch.float8_e4m3fn).reshape(256, 1).expand(256, 32).contiguous()
        x = torch.ones(1, 32, device="cuda", dtype=torch.bfloat16)
        scales = torch.full((256, 1), 2 ** -14, device="cuda")
        expected = F.linear(x, dequantize_fp8_block_weight(w, scales, (1, 32), x.dtype))
        torch.testing.assert_close(fp8_linear(x, w, scales, (1, 32)), expected,
                                   atol=0, rtol=0)

    def test_dispatch_does_not_materialize_weight_and_supports_leading_dims(self):
        module = SimpleNamespace(
            weight=torch.randn(128, 128, device="cuda").to(torch.float8_e4m3fn),
            weight_scale_inv=torch.ones(1, 1, device="cuda") * .01,
            weight_block_size=(128, 128))
        x = torch.randn(2, 3, 128, device="cuda", dtype=torch.bfloat16)
        expected = F.linear(x, dequantize_fp8_block_weight(
            module.weight, module.weight_scale_inv, (128, 128), x.dtype))
        with patch("minivllm.model_executor.parallel_utils.tensor_parallel.layers."
                   "dequantize_fp8_block_weight", side_effect=AssertionError):
            actual = _linear(module, x)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)

    def test_large_prefill_uses_cublas_without_changing_resident_weight(self):
        module = SimpleNamespace(
            weight=torch.randn(128, 128, device="cuda").to(torch.float8_e4m3fn),
            weight_scale_inv=torch.ones(1, 1, device="cuda") * .01,
            weight_block_size=(128, 128))
        x = torch.randn(513, 128, device="cuda", dtype=torch.bfloat16)
        expected = F.linear(x, dequantize_fp8_block_weight(
            module.weight, module.weight_scale_inv, (128, 128), x.dtype))
        pointer = module.weight.data_ptr()
        with patch("minivllm.model_executor.layers.fp8_linear.fp8_linear",
                   side_effect=AssertionError):
            actual = _linear(module, x)
        self.assertEqual(module.weight.data_ptr(), pointer)
        self.assertEqual(module.weight.dtype, torch.float8_e4m3fn)
        torch.testing.assert_close(actual, expected)

    def test_tp_row_linear_writes_the_collective_buffer(self):
        from minivllm.model_executor.parallel_utils import parallel_state
        from minivllm.model_executor.parallel_utils.tensor_parallel.layers import RowParallelLinear
        quant = {"quant_method": "fp8", "weight_block_size": [128, 128]}
        with patch.object(parallel_state, "_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE", 2), \
             patch.object(parallel_state, "_MPU_TENSOR_MODEL_PARALLEL_RANK", 0):
            layer = RowParallelLinear(256, 128, bias=False, input_is_parallel=True,
                                      use_cpu_initialization=True, quant_config=quant).cuda()
            with torch.no_grad():
                layer.weight.copy_(torch.randn(128, 128, device="cuda").to(torch.float8_e4m3fn))
                layer.weight_scale_inv.fill_(.02)
            x = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16)
            buffer = torch.empty(16, 128, device="cuda", dtype=x.dtype)
            seen = []

            def launch(output):
                seen.append(output.data_ptr())
                return output

            launcher = SimpleNamespace(get_buffer=lambda m: buffer, launch=launch)
            expected = F.linear(x, dequantize_fp8_block_weight(
                layer.weight, layer.weight_scale_inv, (128, 128), x.dtype))
            with patch("minivllm.model_executor.parallel_utils.tensor_parallel.layers."
                       "get_all_reduce_launcher", return_value=launcher):
                output, _ = layer(x)
            self.assertEqual(seen, [buffer.data_ptr()])
            torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-3)

    def test_graph_replay_and_current_stream(self):
        from minivllm.model_executor.layers.fp8_linear import fp8_linear
        x = torch.randn(16, 1024, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(128, 1024, device="cuda").to(torch.float8_e4m3fn)
        s = torch.ones(1, 8, device="cuda") * .02
        out = torch.empty(16, 128, device="cuda", dtype=x.dtype)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fp8_linear(x, w, s, (128, 128), out=out)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fp8_linear(x, w, s, (128, 128), out=out)
        x.mul_(.5)
        graph.replay()
        expected = F.linear(x, dequantize_fp8_block_weight(w, s, (128, 128), x.dtype))
        torch.testing.assert_close(out, expected, rtol=2e-2, atol=1e-2)

    def test_ampere_codegen_without_fp8_hardware(self):
        import triton
        from triton.backends.compiler import GPUTarget
        from minivllm.model_executor.layers.fp8_linear import _linear_kernel
        from minivllm.model_executor.layers.gdn_prefill import _prefill
        x = torch.empty(16, 128, device="cuda", dtype=torch.bfloat16)
        w = torch.empty(128, 128, device="cuda", dtype=torch.uint8)
        s = torch.ones(1, 1, device="cuda")
        y = torch.empty_like(x)
        linear = _linear_kernel.warmup(
            x, w, s, None, y, y, 16, 128, 128,
            *x.stride(), *w.stride(), *s.stride(), 128, 128,
            False, 1, 16, 64, 32, num_warps=4, num_stages=1, grid=(2, 1))
        state = torch.zeros(1, 1, 128, 128, device="cuda")
        g = torch.zeros(16, device="cuda")
        beta = torch.ones(16, device="cuda", dtype=x.dtype)
        gdn = _prefill.warmup(x, x, x, g, beta, state, y, None, None,
                             1, 128, 128, 16, False, False, 128, 16,
                             num_warps=4, enable_fp_fusion=False, grid=(1, 8))
        # Compile, but do not claim hardware validation on a local Ada GPU.
        for kernel in (linear, gdn):
            ampere = triton.compile(kernel.src, target=GPUTarget("cuda", 80, 32),
                                   options={"num_warps": 4, "num_stages": 1,
                                            "enable_fp_fusion": False})
            self.assertIn(".target sm_80", ampere.asm["ptx"])


@unittest.skipUnless(RUN, "enable MINIVLLM_RUN_FUSED_TESTS on CUDA")
class StateResidentGdnTest(unittest.TestCase):
    def inputs(self, lengths, dk=128, dv=128, dtype=torch.bfloat16):
        torch.manual_seed(482)
        total, h = sum(lengths), 2
        q, k = prepare_gated_delta_qk(
            torch.randn(total, h, dk, device="cuda", dtype=dtype),
            torch.randn(total, h, dk, device="cuda", dtype=dtype))
        v = torch.randn(total, h, dv, device="cuda", dtype=dtype)
        g = -torch.rand(total, h, device="cuda") * .01
        beta = torch.rand(total, h, device="cuda", dtype=dtype)
        state = torch.randn(len(lengths), h, dk, dv, device="cuda") * .1
        cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()),
                          device="cuda", dtype=torch.int32)
        return q, k, v, g, beta, state, cu

    def test_varlen_and_accepted_prefix_match_original_cuda(self):
        from minivllm import gated_delta_net_ops as ops
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for dk, dv in ((7, 11), (128, 128)):
                for limited in (False, True):
                    with self.subTest(dtype=dtype, dims=(dk, dv), limited=limited):
                        lengths = [0, 1, 15, 16, 65, 512]
                        q, k, v, g, b, state, cu = self.inputs(lengths, dk, dv, dtype)
                        limits = torch.tensor([0, 0, 3, 16, 64, 257], device="cuda",
                                              dtype=torch.int32) if limited else None
                        expected_state, expected = state.clone(), torch.empty_like(v)
                        ops.gated_delta_rule_varlen(expected, q, k, v, g, b,
                                                   expected_state, cu, 512, limits)
                        actual = gated_delta_rule_varlen(q, k, v, g, b, state, cu, 512, limits)
                        selected = torch.cat([torch.arange(int(cu[i]), int(cu[i]) + count)
                                              for i, count in enumerate(
                                                  limits.tolist() if limited else lengths)]).long()
                        torch.testing.assert_close(actual[selected], expected[selected],
                                                   rtol=1e-2, atol=2e-3)
                        torch.testing.assert_close(state, expected_state, rtol=2e-3, atol=2e-5)

    def test_rectangular_prefill_continues_across_calls(self):
        from minivllm import gated_delta_net_ops as ops
        q, k, v, g, b, state, _ = self.inputs([1024])
        q, k, v, g, b = (x.unsqueeze(0) for x in (q, k, v, g, b))
        expected_state, expected = state.clone(), torch.empty_like(v)
        ops.gated_delta_rule_prefill(expected, q, k, v, g, b, expected_state, 64)
        split = 513
        first = gated_delta_rule_prefill(q[:, :split].contiguous(), k[:, :split].contiguous(),
                                        v[:, :split].contiguous(), g[:, :split].contiguous(),
                                        b[:, :split].contiguous(), state)
        second = gated_delta_rule_prefill(q[:, split:].contiguous(), k[:, split:].contiguous(),
                                         v[:, split:].contiguous(), g[:, split:].contiguous(),
                                         b[:, split:].contiguous(), state)
        torch.testing.assert_close(torch.cat((first, second), dim=1), expected,
                                   rtol=1e-2, atol=2e-3)
        torch.testing.assert_close(state, expected_state, rtol=2e-3, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
