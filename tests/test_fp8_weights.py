"""Block-FP8 configuration, storage, independent oracle and forward contracts."""
import copy
import os
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from minivllm.model_executor.layers import fp8

from minivllm.model_executor.layers.fp8 import (
    FP8BlockConfig, Fp8Linear, dequantize_fp8_block_weight, load_fp8_shard,
)


RAW_CONFIG = {
    "quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
    "weight_block_size": [2, 2], "modules_to_not_convert": [],
}


def block_oracle(weight, scale, blocks):
    # An intentionally plain, per-element CPU oracle, not the production helper.
    restored = torch.empty(weight.shape, dtype=torch.float32)
    values, scales = weight.float().cpu(), scale.cpu()
    for i in range(weight.shape[0]):
        for j in range(weight.shape[1]):
            restored[i, j] = values[i, j] * scales[i // blocks[0], j // blocks[1]]
    return restored


class FP8ConfigTest(unittest.TestCase):
    def test_parse_without_mutating_config(self):
        raw = copy.deepcopy(RAW_CONFIG)
        raw["modules_to_not_convert"] = ["lm_head", "model.language_model.layers.0"]
        before = copy.deepcopy(raw)
        result = FP8BlockConfig.from_dict(raw)
        self.assertEqual(result.block_size, (2, 2))
        self.assertEqual(result.modules_to_not_convert, tuple(raw["modules_to_not_convert"]))
        self.assertEqual(raw, before)

    def test_optional_format_and_exclusions(self):
        raw = {k: v for k, v in RAW_CONFIG.items() if k not in ("fmt", "modules_to_not_convert")}
        self.assertEqual(FP8BlockConfig.from_dict(raw), FP8BlockConfig((2, 2)))

    def test_reject_invalid_schema(self):
        for key, value in (
            ("quant_method", "awq"), ("activation_scheme", "static"), ("fmt", "e5m2"),
            ("weight_block_size", None), ("weight_block_size", [2]),
            ("weight_block_size", [0, 2]), ("weight_block_size", [True, 2]),
            ("weight_block_size", [2.0, 2]), ("weight_per_tensor", True),
            ("act_per_tensor", True), ("modules_to_not_convert", "lm_head"),
            ("modules_to_not_convert", ["model..layers"]),
            ("modules_to_not_convert", [" "]), ("modules_to_not_convert", [4]),
        ):
            with self.subTest(key=key, value=value):
                with self.assertRaisesRegex(ValueError, key):
                    FP8BlockConfig.from_dict({**RAW_CONFIG, key: value})

    def test_normalized_exclusion_is_dot_delimited(self):
        config = FP8BlockConfig((2, 2), ("model.language_model.layers.1",))
        self.assertFalse(config.should_quantize(["model.layers.1.mlp.gate_proj"]))
        self.assertTrue(config.should_quantize(["model.language_model.layers.10.mlp.gate_proj"]))

    def test_packed_exclusion_is_all_or_none(self):
        names = ["model.layers.0.self_attn." + p + "_proj" for p in ("q", "k", "v")]
        self.assertTrue(FP8BlockConfig((2, 2)).should_quantize(names))
        self.assertFalse(FP8BlockConfig((2, 2), tuple(names)).should_quantize(names))
        with self.assertRaisesRegex(ValueError, "[Pp]acked|[Pp]artial|[Mm]ixed"):
            FP8BlockConfig((2, 2), (names[1],)).should_quantize(names)

    def test_empty_group_is_invalid(self):
        with self.assertRaises(ValueError):
            FP8BlockConfig((2, 2)).should_quantize([])


class FP8DequantTest(unittest.TestCase):
    def test_hand_computed_tail_blocks(self):
        w = torch.ones(3, 5).to(torch.float8_e4m3fn)
        s = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3)
        result = dequantize_fp8_block_weight(w, s, (2, 2), torch.float32)
        expected = torch.tensor([[1, 1, 2, 2, 3], [1, 1, 2, 2, 3], [4, 4, 5, 5, 6.]])
        torch.testing.assert_close(result, expected)

    def test_dtypes_noncontiguous_and_unchanged_inputs(self):
        w = (torch.arange(30).reshape(5, 6).float() / 8).to(torch.float8_e4m3fn).t()
        s = torch.tensor([[.125, 2., 4.], [.5, 1., .25]]).t()
        w_before, s_before = w.float().clone(), s.clone()
        expected = block_oracle(w, s, (2, 3))
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                out = dequantize_fp8_block_weight(w, s, (2, 3), dtype)
                self.assertEqual(out.dtype, dtype)
                self.assertTrue(out.is_contiguous())
                torch.testing.assert_close(out, expected.to(dtype))
        torch.testing.assert_close(w.float(), w_before)
        torch.testing.assert_close(s, s_before)

    def test_scale_multiply_happens_before_half_cast(self):
        w = torch.full((2, 2), 1 / 256).to(torch.float8_e4m3fn)
        scale = torch.tensor([[100000.0]])
        actual = dequantize_fp8_block_weight(w, scale, (2, 2), torch.float16)
        torch.testing.assert_close(actual, (w.float() * scale).half())
        self.assertTrue(torch.isfinite(actual).all())

    def test_metadata_errors(self):
        w = torch.ones(3, 5).to(torch.float8_e4m3fn)
        s = torch.ones(2, 3)
        for weight, scale, blocks, dtype in (
            (w.float(), s, (2, 2), torch.float32),
            (w, s.half(), (2, 2), torch.float32),
            (w.flatten(), s, (2, 2), torch.float32),
            (w, s[:1], (2, 2), torch.float32),
            (w, s, (0, 2), torch.float32),
            (w, s, (2, 2), torch.int8),
        ):
            with self.subTest(shape=weight.shape, dtype=dtype):
                with self.assertRaises(ValueError):
                    dequantize_fp8_block_weight(weight, scale, blocks, dtype)


class FP8StorageTest(unittest.TestCase):
    def test_scale_first_and_packed_offsets(self):
        layer = Fp8Linear(5, 7, (2, 2))
        # Internal boundaries 0,4,6 align; the final one-row shard is a tail.
        for offset, rows, factor in ((0, 4, 1.), (4, 2, 3.), (6, 1, 5.)):
            scale = torch.full(((rows + 1) // 2, 3), factor)
            weight = torch.full((rows, 5), factor / 2).to(torch.float8_e4m3fn)
            load_fp8_shard(layer, scale, kind="scale", row_offset=offset,
                          num_rows=rows, source_name="proj.weight_scale_inv")
            load_fp8_shard(layer, weight, kind="weight", row_offset=offset,
                          num_rows=rows, source_name="proj.weight")
            torch.testing.assert_close(layer.weight[offset:offset+rows].float(), weight.float())
            torch.testing.assert_close(layer.weight_scale_inv[offset//2:(offset+rows+1)//2], scale)

    def test_bad_load_does_not_mutate_destination(self):
        layer = Fp8Linear(4, 8, (2, 2))
        layer.weight.copy_(torch.ones(8, 4).to(torch.float8_e4m3fn))
        layer.weight_scale_inv.fill_(2)
        for kind, offset, rows, tensor in (
            ("weight", 1, 2, torch.ones(2, 4).to(torch.float8_e4m3fn)),
            ("weight", 0, 3, torch.ones(3, 4).to(torch.float8_e4m3fn)),
            ("weight", 8, 2, torch.ones(2, 4).to(torch.float8_e4m3fn)),
            ("weight", 0, 2, torch.ones(2, 4)),
            ("weight", 0, 2, torch.ones(3, 4).to(torch.float8_e4m3fn)),
            ("weight", 0, 2, torch.full((2, 4), float("nan")).to(torch.float8_e4m3fn)),
            ("scale", 0, 2, torch.ones(1, 2).half()),
            ("scale", 0, 2, torch.zeros(1, 2)),
            ("scale", 0, 2, torch.full((1, 2), float("nan"))),
            ("scale", 0, 2, torch.full((1, 2), float("inf"))),
            ("scale", 0, 2, -torch.ones(1, 2)),
            ("other", 0, 2, torch.ones(1, 2)),
        ):
            with self.subTest(kind=kind, offset=offset, shape=tensor.shape):
                with self.assertRaisesRegex(ValueError, "bad_source"):
                    load_fp8_shard(layer, tensor, kind=kind, row_offset=offset,
                                  num_rows=rows, source_name="bad_source")
                torch.testing.assert_close(layer.weight.float(), torch.ones(8, 4))
                torch.testing.assert_close(layer.weight_scale_inv, torch.full((4, 2), 2.))

class FP8ForwardContractTest(unittest.TestCase):
    def test_forward_delegates_without_materializing_weight(self):
        for returns_tuple in (False, True):
            layer = Fp8Linear(5, 3, (2, 2), returns_tuple=returns_tuple)
            w = (torch.arange(15).reshape(3, 5).float() / 4).to(torch.float8_e4m3fn)
            s = torch.tensor([[.25, 1, 2], [4., .5, .125]])
            layer.weight.copy_(w)
            layer.weight_scale_inv.copy_(s)
            before = {name: p.data_ptr() for name, p in layer.named_parameters()}
            x = torch.arange(40).float().reshape(2, 5, 4).transpose(1, 2) / 40
            expected = F.linear(x, block_oracle(w, s, (2, 2)))
            for _ in range(2):
                with patch.object(fp8, "fused_fp8_linear", return_value=expected) as fused, \
                     patch.object(fp8, "dequantize_fp8_block_weight", side_effect=AssertionError("not production")), \
                     patch.object(F, "linear", side_effect=AssertionError("not production")):
                    out = layer(x)
                args = fused.call_args.args
                self.assertIs(args[0], x)
                self.assertIs(args[1], layer.weight)
                self.assertIs(args[2], layer.weight_scale_inv)
                self.assertEqual(args[3], (2, 2))
                if returns_tuple:
                    self.assertIsNone(out[1])
                    out = out[0]
                torch.testing.assert_close(out, expected)
            self.assertEqual({n: p.data_ptr() for n, p in layer.named_parameters()}, before)
            self.assertEqual(set(layer.state_dict()), {"weight", "weight_scale_inv"})
            self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
            self.assertEqual(layer.weight_scale_inv.dtype, torch.float32)
            self.assertFalse(layer.weight.requires_grad)

    def test_cpu_execution_does_not_fall_back_to_reference(self):
        layer = Fp8Linear(5, 3, (2, 2))
        with patch.object(fp8, "dequantize_fp8_block_weight", side_effect=AssertionError("not production")):
            with self.assertRaisesRegex(ValueError, "CUDA"):
                layer(torch.zeros(2, 5, dtype=torch.float16))


@unittest.skipUnless(os.environ.get("MINIVLLM_RUN_CUDA_FP8_TESTS") == "1"
                     and torch.cuda.is_available(), "enable MINIVLLM_RUN_CUDA_FP8_TESTS=1")
class FP8CudaTest(unittest.TestCase):
    def test_cuda_linear_half_and_bfloat16(self):
        w = (torch.arange(35).reshape(5, 7).float() / 16).to(torch.float8_e4m3fn)
        s = torch.arange(1, 10, dtype=torch.float32).reshape(3, 3) / 8
        restored = block_oracle(w, s, (2, 3))
        layer = Fp8Linear(7, 5, (2, 3)).cuda()
        load_fp8_shard(layer, w, kind="weight", row_offset=0, num_rows=5, source_name="cuda.weight")
        load_fp8_shard(layer, s, kind="scale", row_offset=0, num_rows=5, source_name="cuda.weight_scale_inv")
        for dtype in (torch.float16, torch.bfloat16):
            x = torch.randn(2, 3, 7, device="cuda", dtype=dtype)
            torch.testing.assert_close(layer(x), F.linear(x, restored.cuda().to(dtype)))
        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)


if __name__ == "__main__":
    unittest.main()
