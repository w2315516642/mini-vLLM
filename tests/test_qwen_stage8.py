"""Learner acceptance tests: TODO errors are expected before stage 8 submission."""
from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from minivllm.model_executor.models import qwen3_5 as qwen
from minivllm.model_executor.models import llama
from minivllm.model_executor.layers.gated_delta_net import Qwen3_5GatedDeltaNetReference
from minivllm.worker.hybrid_cache import HybridCache, GatedDeltaNetStateSpec


def tiny_config(**overrides):
    values = dict(
        hidden_size=16, intermediate_size=32, vocab_size=32,
        num_hidden_layers=2, layer_types=["linear_attention", "full_attention"],
        num_attention_heads=2, num_key_value_heads=1, head_dim=64,
        hidden_act="silu", rms_norm_eps=1e-6, max_position_embeddings=128,
        rope_parameters=dict(rope_theta=10000000.0, partial_rotary_factor=0.25),
        tie_word_embeddings=True, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=4, linear_value_head_dim=4, linear_conv_kernel_dim=3,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class CpuParallelLinear(nn.Linear):
    def __init__(self, input_size, output_size, bias=False, **kwargs):
        super().__init__(input_size, output_size, bias=bias)

    def forward(self, x):
        return super().forward(x), None


@contextmanager
def cpu_model_layers():
    # Keep real shapes and checkpoint parameters without initializing NCCL/GPU.
    with patch.object(qwen, "get_tensor_model_parallel_world_size", return_value=1), \
         patch.object(qwen, "ColumnParallelLinear", CpuParallelLinear), \
         patch.object(qwen, "RowParallelLinear", CpuParallelLinear), \
         patch.object(llama, "ColumnParallelLinear", CpuParallelLinear), \
         patch.object(llama, "RowParallelLinear", CpuParallelLinear):
        yield


def make_cache(config, device="cpu"):
    full = {
        i: (torch.empty(0, device=device), torch.empty(0, device=device))
        for i, kind in enumerate(config.layer_types) if kind == "full_attention"
    }
    return HybridCache(config.layer_types, full,
                       GatedDeltaNetStateSpec.from_text_config(config), 1, device)


class Stage8DecoderTest(unittest.TestCase):
    def test_constructor_selects_mixer_and_zero_centered_norms(self):
        with cpu_model_layers():
            linear = qwen.Qwen3_5DecoderLayer(tiny_config(), 0)
            full = qwen.Qwen3_5DecoderLayer(tiny_config(), 1)
        self.assertIsInstance(linear.linear_attn, qwen.Qwen3_5GatedDeltaNet)
        self.assertIsInstance(full.self_attn, qwen.Qwen3_5Attention)
        self.assertFalse(hasattr(linear, "self_attn"))
        self.assertFalse(hasattr(full, "linear_attn"))
        self.assertIsInstance(linear.mlp, llama.LlamaMLP)
        for layer in (linear, full):
            self.assertIsInstance(layer.input_layernorm, qwen.Qwen3_5RMSNorm)
            self.assertIsInstance(layer.post_attention_layernorm, qwen.Qwen3_5RMSNorm)

    def test_linear_forward_orders_residuals_and_writes_both_states(self):
        layer = qwen.Qwen3_5DecoderLayer.__new__(qwen.Qwen3_5DecoderLayer)
        nn.Module.__init__(layer)
        layer.layer_idx, layer.layer_type = 0, "linear_attention"
        layer.input_layernorm = nn.Identity()
        layer.post_attention_layernorm = nn.Identity()
        layer.mlp = nn.Identity()

        class Mixer(nn.Module):
            def forward(self, x, state):
                state.conv_state.add_(2)
                state.recurrent_state.add_(3)
                return x * 4, state

        layer.linear_attn = Mixer()
        cache = make_cache(tiny_config())
        slots = cache.acquire([10])
        x = torch.ones(3, 16)
        actual = layer(torch.arange(3), x, cache,
                       SimpleNamespace(state_slot_ids=slots))
        torch.testing.assert_close(actual, x * 10)
        state = cache.read_state(0, slots)
        self.assertTrue(torch.all(state.conv_state == 2))
        self.assertTrue(torch.all(state.recurrent_state == 3))

    def test_full_forward_uses_global_layer_kv_and_event(self):
        layer = qwen.Qwen3_5DecoderLayer.__new__(qwen.Qwen3_5DecoderLayer)
        nn.Module.__init__(layer)
        layer.layer_idx, layer.layer_type = 1, "full_attention"
        layer.input_layernorm = nn.Identity()
        layer.post_attention_layernorm = nn.Identity()
        layer.mlp = nn.Identity()
        cache = make_cache(tiny_config())
        event = object()
        seen = {}

        class Attention(nn.Module):
            def forward(self, positions, hidden_states, kv_cache,
                        input_metadata, cache_event):
                seen.update(kv=kv_cache, event=cache_event)
                return hidden_states * 2

        layer.self_attn = Attention()
        x = torch.ones(3, 16)
        output = layer(torch.arange(3), x, cache, SimpleNamespace(), event)
        torch.testing.assert_close(output, x * 6)
        self.assertIs(seen["kv"], cache.get_kv_cache(1))
        self.assertIs(seen["event"], event)

    def test_model_discards_padding_before_embedding_and_preserves_layer_order(self):
        model = qwen.Qwen3_5Model.__new__(qwen.Qwen3_5Model)
        nn.Module.__init__(model)
        model.embed_tokens = nn.Embedding(4, 2)
        model.embed_tokens.weight.data.fill_(1)
        calls = []

        class Layer(nn.Module):
            def __init__(self, index):
                super().__init__()
                self.index = index

            def forward(self, positions, hidden_states, hybrid_cache,
                        input_metadata, cache_event=None):
                calls.append((self.index, positions.tolist(), cache_event))
                return hidden_states * 2 + self.index

        model.layers = nn.ModuleList([Layer(0), Layer(1)])
        model.norm = nn.Identity()
        # Invalid embedding IDs in padding make late trimming observable.
        result = model(torch.tensor([1, 2, 999, 999]), torch.arange(4), object(),
                       SimpleNamespace(num_valid_tokens=2), ["event0", "event1"])
        torch.testing.assert_close(result, torch.full((2, 2), 5.0))
        self.assertEqual(calls, [(0, [0, 1], "event0"), (1, [0, 1], "event1")])


class Stage8WeightTest(unittest.TestCase):
    def make_loader(self, tied=True):
        # Real loader on a tiny parameter tree, independent of decoder TODOs.
        model = qwen.Qwen3_5ForConditionalGeneration.__new__(
            qwen.Qwen3_5ForConditionalGeneration)
        nn.Module.__init__(model)
        model.config = tiny_config(tie_word_embeddings=tied)
        model.model = nn.Module()
        model.model.embed_tokens = nn.Embedding(32, 16)
        layer = nn.Module()
        with cpu_model_layers():
            layer.self_attn = qwen.Qwen3_5Attention(model.config)
            layer.mlp = llama.LlamaMLP(16, 32, "silu")
            linear = nn.Module()
            linear.linear_attn = qwen.Qwen3_5GatedDeltaNet(model.config)
            linear.mlp = llama.LlamaMLP(16, 32, "silu")
        for decoder in (layer, linear):
            decoder.input_layernorm = qwen.Qwen3_5RMSNorm(16)
            decoder.post_attention_layernorm = qwen.Qwen3_5RMSNorm(16)
        model.model.layers = nn.ModuleList([layer, linear])
        model.model.norm = qwen.Qwen3_5RMSNorm(16)
        model.lm_head = nn.Linear(16, 32, bias=False)
        if tied:
            model.lm_head.weight = model.model.embed_tokens.weight
        expected = {}
        checkpoint = []
        for i, (name, param) in enumerate(model.named_parameters()):
            value = torch.arange(param.numel(), dtype=param.dtype).reshape_as(param)
            value = value / max(param.numel(), 1) + i
            expected[name] = value
            external = name.replace("model.", "model.language_model.", 1)
            if "qkv_gate_proj" in name:
                sizes = [2 * layer.self_attn.q_size, layer.self_attn.kv_size,
                         layer.self_attn.kv_size]
                for part, tensor in zip(("q_proj", "k_proj", "v_proj"), value.split(sizes)):
                    checkpoint.append((external.replace("qkv_gate_proj", part), tensor))
            elif "gate_up_proj" in name:
                for part, tensor in zip(("gate_proj", "up_proj"), value.chunk(2)):
                    checkpoint.append((external.replace("gate_up_proj", part), tensor))
            else:
                checkpoint.append((external, value))
        return model, checkpoint, expected

    def test_load_packed_shards_tied_and_untied_embedding(self):
        for tied in (True, False):
            with self.subTest(tied=tied), cpu_model_layers():
                model, checkpoint, expected = self.make_loader(tied)
                checkpoint += [("model.visual.fake.weight", torch.ones(1)),
                               ("mtp.fake.weight", torch.ones(1))]
                model.load_weights_from_iterator(reversed(checkpoint))
                for name, param in model.named_parameters():
                    torch.testing.assert_close(param, expected[name])
                if tied:
                    self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)

    def test_missing_qkv_shard_is_not_hidden_by_packed_parameter_coverage(self):
        with cpu_model_layers():
            model, checkpoint, _ = self.make_loader()
            checkpoint = [(n, w) for n, w in checkpoint if ".k_proj." not in n]
            with self.assertRaises(ValueError):
                model.load_weights_from_iterator(checkpoint)

    def test_unknown_language_weight_duplicate_and_bad_shape_are_rejected(self):
        for issue in ("unknown", "duplicate", "shape"):
            with self.subTest(issue=issue), cpu_model_layers():
                model, checkpoint, _ = self.make_loader()
                if issue == "unknown":
                    checkpoint.append(("model.language_model.typo.weight", torch.ones(1)))
                elif issue == "duplicate":
                    checkpoint.append(checkpoint[0])
                else:
                    name, weight = checkpoint[0]
                    checkpoint[0] = (name, weight[:1])
                with self.assertRaises((ValueError, KeyError, AssertionError)):
                    model.load_weights_from_iterator(checkpoint)

    def test_local_prefix_and_destination_dtype(self):
        model, checkpoint, expected = self.make_loader()
        model.half()
        checkpoint = [(n.replace("model.language_model.", "model."), w)
                      for n, w in checkpoint]
        checkpoint.append(("model.layers.0.self_attn.rotary_emb.inv_freq", torch.ones(1)))
        model.load_weights_from_iterator(checkpoint)
        for name, param in model.named_parameters():
            torch.testing.assert_close(param, expected[name].half())

    def test_missing_mlp_gdn_or_embedding_is_reported(self):
        for suffix in ("mlp.up_proj.weight", "linear_attn.A_log", "embed_tokens.weight"):
            with self.subTest(suffix=suffix):
                model, checkpoint, _ = self.make_loader()
                checkpoint = [(n, w) for n, w in checkpoint if not n.endswith(suffix)]
                with self.assertRaisesRegex(ValueError, "Missing"):
                    model.load_weights_from_iterator(checkpoint)

    def test_oversized_q_shard_is_not_silently_truncated(self):
        model, checkpoint, _ = self.make_loader()
        checkpoint = [(n, torch.cat([w, w[:1]]) if n.endswith(".q_proj.weight") else w)
                      for n, w in checkpoint]
        with self.assertRaisesRegex(ValueError, "expected"):
            model.load_weights_from_iterator(checkpoint)

    def test_alias_prefix_does_not_hide_duplicate_shard(self):
        model, checkpoint, _ = self.make_loader()
        name, weight = next((n, w) for n, w in checkpoint if n.endswith(".k_proj.weight"))
        checkpoint.append((name.replace("model.language_model.", "model."), weight))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            model.load_weights_from_iterator(checkpoint)

    def test_tied_head_is_optional_but_must_agree_when_present(self):
        for head_first in (True, False):
            with self.subTest(head_first=head_first):
                model, checkpoint, expected = self.make_loader()
                head = ("lm_head.weight", expected["model.embed_tokens.weight"].clone())
                checkpoint = ([head] + checkpoint if head_first else checkpoint + [head])
                model.load_weights_from_iterator(checkpoint)
                self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)
                torch.testing.assert_close(model.lm_head.weight, head[1])

    def test_conflicting_tied_head_and_head_without_embedding_are_rejected(self):
        model, checkpoint, expected = self.make_loader()
        head = ("lm_head.weight", expected["model.embed_tokens.weight"] + 1)
        with self.assertRaisesRegex(ValueError, "disagree"):
            model.load_weights_from_iterator(checkpoint + [head])
        model, checkpoint, expected = self.make_loader()
        checkpoint = [(n, w) for n, w in checkpoint if not n.endswith("embed_tokens.weight")]
        checkpoint.append(("lm_head.weight", expected["model.embed_tokens.weight"]))
        with self.assertRaisesRegex(ValueError, "Missing.*embed_tokens"):
            model.load_weights_from_iterator(checkpoint)


@unittest.skipUnless(torch.cuda.is_available(), "requires existing stage 6 CUDA ops")
class Stage8GdnCudaTest(unittest.TestCase):
    @torch.inference_mode()
    def test_prefill_and_decode_match_reference_and_continue_state(self):
        torch.manual_seed(8)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                config = tiny_config()
                reference = Qwen3_5GatedDeltaNetReference(config).cuda().to(dtype)
                actual = qwen.Qwen3_5GatedDeltaNet(config).cuda().to(dtype)
                actual.load_state_dict(reference.state_dict())
                x = torch.randn(1, 6, 16, device="cuda", dtype=dtype)
                expected, expected_state = reference(x)
                first, state = actual(x[:, :5], actual.empty_state(1, x.device))
                last, state = actual(x[:, 5:], state)
                tol = 2e-2 if dtype == torch.bfloat16 else 3e-3
                torch.testing.assert_close(torch.cat([first, last], 1), expected,
                                           rtol=tol, atol=tol)
                torch.testing.assert_close(state.conv_state, expected_state.conv_state,
                                           rtol=tol, atol=tol)
                torch.testing.assert_close(state.recurrent_state, expected_state.recurrent_state,
                                           rtol=tol, atol=tol)


if __name__ == "__main__":
    unittest.main()
