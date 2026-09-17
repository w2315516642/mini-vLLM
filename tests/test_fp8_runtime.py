"""FP8 model wiring, packed storage and checkpoint integration tests."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from safetensors.torch import save_file

import test_qwen_stage8 as stage8
from minivllm.model_executor.layers import fp8
from minivllm.model_executor import model_loader
from minivllm.model_executor.weight_utils import hf_model_weights_iterator
from test_fp8_weights import RAW_CONFIG, block_oracle


def make_checkpoint():
    """Synthetic, inspectable Qwen checkpoint; not an official model export."""
    model, original, _ = stage8.Stage8WeightTest().make_loader()
    result = []
    for name, weight in original:
        # Only the same bias-free decoder projections selected by configure_fp8.
        quantized = weight.ndim == 2 and ".layers." in name and name.endswith(".weight")
        if quantized:
            rows, cols = weight.shape
            scale = torch.full(((rows + 1)//2, (cols + 1)//2), .125)
            result.extend([(name, (weight / .125).to(torch.float8_e4m3fn)),
                           (name.removesuffix(".weight") + ".weight_scale_inv", scale)])
        else:
            result.append((name, weight))
    return model, result


class FP8PlumbingTest(unittest.TestCase):
    def test_root_text_resolution_and_conflicts(self):
        root, text = SimpleNamespace(), SimpleNamespace()
        self.assertIsNone(fp8.resolve_fp8_config(root, text))
        root.quantization_config = RAW_CONFIG
        sentinel = fp8.FP8BlockConfig((2, 2))
        with patch.object(fp8.FP8BlockConfig, "from_dict", return_value=sentinel) as parse:
            self.assertIs(fp8.resolve_fp8_config(root, text), sentinel)
            parse.assert_called_once_with(RAW_CONFIG)
        self.assertFalse(hasattr(text, "quantization_config"))
        text.quantization_config = {**RAW_CONFIG, "weight_block_size": [4, 4]}
        with self.assertRaisesRegex(ValueError, "disagree"):
            fp8.resolve_fp8_config(root, text)

    def test_loader_configures_before_weights_and_preserves_text(self):
        root, text = SimpleNamespace(quantization_config=RAW_CONFIG), SimpleNamespace()
        config = SimpleNamespace(
            architecture=SimpleNamespace(root_config=root, text_config=text, architectures=("Qwen",)),
            dtype=torch.float32, use_dummy_weights=False, model="unused",
            download_dir=None, use_np_weights=False,
        )
        events = []
        class RecordingModel(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                events.append(("construct", cfg))
            def configure_fp8(self, cfg):
                events.append(("fp8", cfg))
            def load_weights(self, *args):
                events.append(("load", args))
            def cuda(self):
                events.append(("cuda", None))
                return self
        parsed = fp8.FP8BlockConfig((2, 2))
        with patch.object(fp8.FP8BlockConfig, "from_dict", return_value=parsed), \
             patch.object(model_loader, "_get_model_architecture", return_value=RecordingModel), \
             patch.object(model_loader.torch, "set_default_dtype"):
            model_loader.get_model(config)
        self.assertEqual([e[0] for e in events], ["construct", "fp8", "load", "cuda"])
        self.assertIs(events[0][1], text)
        self.assertIs(events[1][1], parsed)

    def test_model_replacement_and_scale_routes_with_mock_loader(self):
        model, checkpoint = make_checkpoint()
        with stage8.cpu_model_layers(), patch.object(fp8.FP8BlockConfig, "should_quantize", return_value=True):
            model.configure_fp8(fp8.FP8BlockConfig((2, 2)))
        with patch.object(stage8.qwen, "load_fp8_shard") as load:
            model.load_weights_from_iterator(reversed(checkpoint))
        calls = [c.kwargs for c in load.call_args_list if ".self_attn." in c.kwargs["source_name"]]
        k_scale = next(c for c in calls if c["source_name"].endswith("k_proj.weight_scale_inv"))
        self.assertEqual(k_scale["kind"], "scale")
        self.assertEqual(k_scale["row_offset"], 256)
        self.assertEqual(k_scale["num_rows"], 64)
        self.assertTrue(model.model.layers[0].self_attn.qkv_gate_proj.returns_tuple)
        self.assertFalse(model.model.layers[1].linear_attn.in_proj_z.returns_tuple)
        self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)
        self.assertEqual(model.model.embed_tokens.weight.dtype, torch.float32)

    def test_missing_duplicate_unknown_scales_are_loader_errors(self):
        for issue in ("missing", "duplicate", "unknown"):
            with self.subTest(issue=issue):
                model, checkpoint = make_checkpoint()
                with stage8.cpu_model_layers(), patch.object(fp8.FP8BlockConfig, "should_quantize", return_value=True):
                    model.configure_fp8(fp8.FP8BlockConfig((2, 2)))
                scale = next(item for item in checkpoint if item[0].endswith("k_proj.weight_scale_inv"))
                if issue == "missing":
                    checkpoint = [item for item in checkpoint if item[0] != scale[0]]
                elif issue == "duplicate":
                    checkpoint.append(scale)
                else:
                    checkpoint.append(("model.layers.0.typo.weight_scale_inv", torch.ones(1)))
                with patch.object(stage8.qwen, "load_fp8_shard"), self.assertRaises(ValueError):
                    model.load_weights_from_iterator(checkpoint)

    def test_plain_model_does_not_silently_cast_fp8(self):
        model, checkpoint = make_checkpoint()
        with self.assertRaisesRegex(ValueError, "no configured scale"):
            model.load_weights_from_iterator(checkpoint)

    def test_packed_boundary_rejected_before_replacement(self):
        model, _ = make_checkpoint()
        old = model.model.layers[0].self_attn.qkv_gate_proj
        with stage8.cpu_model_layers(), patch.object(fp8.FP8BlockConfig, "should_quantize", return_value=True):
            with self.assertRaisesRegex(ValueError, "block-aligned"):
                model.configure_fp8(fp8.FP8BlockConfig((3, 2)))
        self.assertIs(model.model.layers[0].self_attn.qkv_gate_proj, old)


class FP8ModelIntegrationTest(unittest.TestCase):
    def test_sharded_checkpoint_scales_first_and_packed_storage(self):
        model, checkpoint = make_checkpoint()
        with stage8.cpu_model_layers():
            model.configure_fp8(fp8.FP8BlockConfig.from_dict(RAW_CONFIG))
        with tempfile.TemporaryDirectory() as folder:
            scales = {n: w for n, w in checkpoint if n.endswith("weight_scale_inv")}
            weights = {n: w.contiguous() for n, w in checkpoint if n not in scales}
            save_file(scales, str(Path(folder) / "a.safetensors"))
            save_file(weights, str(Path(folder) / "b.safetensors"))
            model.load_weights_from_iterator(hf_model_weights_iterator(folder))
        source = dict(checkpoint)
        prefix = "model.language_model.layers.0.self_attn."
        restored = [block_oracle(source[prefix + p + "_proj.weight"],
                                source[prefix + p + "_proj.weight_scale_inv"], (2, 2))
                    for p in ("q", "k", "v")]
        layer = model.model.layers[0].self_attn.qkv_gate_proj
        # CPU validates checkpoint layout only, not a hidden CPU forward fallback.
        actual = block_oracle(layer.weight, layer.weight_scale_inv, (2, 2))
        torch.testing.assert_close(actual, torch.cat(restored))
        prefix = "model.language_model.layers.1.mlp."
        expected = torch.cat([block_oracle(source[prefix + p + "_proj.weight"],
                                           source[prefix + p + "_proj.weight_scale_inv"], (2, 2))
                              for p in ("gate", "up")])
        layer = model.model.layers[1].mlp.gate_up_proj
        actual = block_oracle(layer.weight, layer.weight_scale_inv, (2, 2))
        torch.testing.assert_close(actual, expected)

    def test_excluded_gdn_projection_stays_floating(self):
        model, _ = make_checkpoint()
        name = "model.language_model.layers.1.linear_attn.in_proj_a"
        with stage8.cpu_model_layers():
            model.configure_fp8(fp8.FP8BlockConfig((2, 2), (name,)))
        self.assertIsInstance(model.model.layers[1].linear_attn.in_proj_z, fp8.Fp8Linear)
        self.assertNotIsInstance(model.model.layers[1].linear_attn.in_proj_a, fp8.Fp8Linear)


if __name__ == "__main__":
    unittest.main()
