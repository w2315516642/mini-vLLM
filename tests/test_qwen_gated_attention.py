import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from minivllm.model_executor.layers.layer_norm import Qwen3_5RMSNorm
from minivllm.model_executor.models import qwen3_5 as qwen_module
from minivllm.model_executor.models.qwen3_5 import (
    Qwen3_5Attention,
    _load_qkv_gate_weight,
    _split_q_gate_kv,
)


class RecordingColumnParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.kwargs = kwargs


class RecordingRowParallelLinear(RecordingColumnParallelLinear):
    pass


class RecordingPagedAttention(nn.Module):
    def __init__(
        self,
        num_heads,
        head_size,
        scale,
        rotary_dim,
        max_position,
        base,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.scale = scale
        self.rotary_dim = rotary_dim
        self.max_position = max_position
        self.base = base


class QwenRMSNormTest(unittest.TestCase):
    def test_zero_centered_weight_matches_fp32_reference(self):
        norm = Qwen3_5RMSNorm(4, eps=1e-5)
        with torch.no_grad():
            norm.weight.copy_(torch.tensor([0.0, 0.5, -0.25, 1.0]))
        x = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [4.0, -3.0, 2.0, -1.0]],
                [[0.5, 1.5, -2.0, 3.0], [-4.0, 1.0, 0.5, 2.0]],
            ],
            dtype=torch.float16,
        )

        actual = norm(x)
        x_fp32 = x.float()
        expected = x_fp32 * torch.rsqrt(
            x_fp32.pow(2).mean(dim=-1, keepdim=True) + 1e-5
        )
        expected = expected * (1.0 + norm.weight.float())

        self.assertEqual(actual.dtype, x.dtype)
        torch.testing.assert_close(
            actual,
            expected.to(x.dtype),
            rtol=2e-3,
            atol=2e-3,
        )

    def test_each_head_is_normalized_independently(self):
        norm = Qwen3_5RMSNorm(2, eps=0.0)
        x = torch.tensor([[[3.0, 4.0], [6.0, 8.0]]])

        actual = norm(x)

        self.assertEqual(actual.shape, (1, 2, 2))
        torch.testing.assert_close(
            actual.pow(2).mean(dim=-1),
            torch.ones(1, 2),
        )


class QwenPackedProjectionTest(unittest.TestCase):
    def test_q_and_gate_are_split_inside_each_query_head(self):
        # q_gate is [head0 q, head0 gate, head1 q, head1 gate]. Values are
        # intentionally distinct so a flat chunk(2) cannot pass accidentally.
        q_gate = torch.tensor(
            [[10, 11, 20, 21, 30, 31, 40, 41]],
            dtype=torch.float32,
        )
        key = torch.tensor([[50, 51]], dtype=torch.float32)
        value = torch.tensor([[60, 61]], dtype=torch.float32)
        packed = torch.cat((q_gate, key, value), dim=-1)

        query, gate, actual_key, actual_value = _split_q_gate_kv(
            packed,
            num_query_heads=2,
            head_dim=2,
            kv_size=2,
        )

        self.assertTrue(torch.equal(query, torch.tensor([[10, 11, 30, 31]])))
        self.assertTrue(torch.equal(gate, torch.tensor([[20, 21, 40, 41]])))
        self.assertTrue(torch.equal(actual_key, key))
        self.assertTrue(torch.equal(actual_value, value))

    def test_tp_checkpoint_shards_land_in_local_packed_segments(self):
        q_size = 4
        kv_size = 2
        param = torch.zeros(2 * q_size + 2 * kv_size, 1)
        query_gate_weight = torch.arange(16, dtype=torch.float32).reshape(-1, 1)
        key_weight = torch.arange(4, dtype=torch.float32).reshape(-1, 1) + 100
        value_weight = torch.arange(4, dtype=torch.float32).reshape(-1, 1) + 200

        for shard_id, loaded_weight in (
            ("q", query_gate_weight),
            ("k", key_weight),
            ("v", value_weight),
        ):
            _load_qkv_gate_weight(
                param,
                loaded_weight,
                shard_id,
                tensor_model_parallel_rank=1,
                q_size=q_size,
                kv_size=kv_size,
            )

        expected = torch.tensor(
            [
                [8], [9], [10], [11], [12], [13], [14], [15],
                [102], [103],
                [202], [203],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(param, expected))

    def test_unknown_checkpoint_shard_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "shard"):
            _load_qkv_gate_weight(
                torch.zeros(12, 1),
                torch.zeros(8, 1),
                "gate",
                tensor_model_parallel_rank=0,
                q_size=4,
                kv_size=2,
            )


class RecordingNorm(nn.Module):
    def __init__(self, increment):
        super().__init__()
        self.increment = increment
        self.seen_shape = None

    def forward(self, value):
        self.seen_shape = tuple(value.shape)
        return value + self.increment


class FixedProjection(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, hidden_states):
        return self.value, None


class QwenAttentionIntegrationTest(unittest.TestCase):
    @staticmethod
    def _config(**overrides):
        values = {
            "hidden_size": 5120,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "partial_rotary_factor": 0.25,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 262_144,
            "rope_theta": 10_000_000.0,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_constructor_uses_qwen38_tp_and_partial_rope_dimensions(self):
        with patch.object(
            qwen_module,
            "get_tensor_model_parallel_world_size",
            return_value=2,
        ), patch.object(
            qwen_module,
            "ColumnParallelLinear",
            RecordingColumnParallelLinear,
        ), patch.object(
            qwen_module,
            "RowParallelLinear",
            RecordingRowParallelLinear,
        ), patch.object(
            qwen_module,
            "PagedAttentionWithRoPE",
            RecordingPagedAttention,
        ):
            attention = Qwen3_5Attention(self._config())

        self.assertEqual(attention.num_heads, 12)
        self.assertEqual(attention.num_kv_heads, 2)
        self.assertEqual(attention.q_size, 3072)
        self.assertEqual(attention.kv_size, 512)
        self.assertEqual(attention.rotary_dim, 64)
        self.assertEqual(attention.qkv_gate_proj.output_size, 14_336)
        self.assertEqual(attention.o_proj.input_size, 6144)
        self.assertEqual(tuple(attention.q_norm.weight.shape), (256,))
        self.assertEqual(tuple(attention.k_norm.weight.shape), (256,))
        self.assertEqual(attention.attn.num_heads, 12)
        self.assertEqual(attention.attn.num_kv_heads, 2)
        self.assertEqual(attention.attn.rotary_dim, 64)
        self.assertEqual(attention.attn.base, 10_000_000.0)

    def test_invalid_partial_rotary_dimension_is_rejected(self):
        with patch.object(
            qwen_module,
            "get_tensor_model_parallel_world_size",
            return_value=1,
        ):
            with self.assertRaisesRegex(ValueError, "rotary_dim"):
                Qwen3_5Attention(
                    self._config(head_dim=6, partial_rotary_factor=0.5)
                )

    def test_projection_normalizes_q_and_k_per_head(self):
        attention = object.__new__(Qwen3_5Attention)
        nn.Module.__init__(attention)
        attention.num_heads = 2
        attention.num_kv_heads = 1
        attention.head_dim = 2
        attention.q_size = 4
        attention.kv_size = 2
        packed = torch.tensor(
            [[10, 11, 20, 21, 30, 31, 40, 41, 50, 51, 60, 61]],
            dtype=torch.float32,
        )
        attention.qkv_gate_proj = FixedProjection(packed)
        attention.q_norm = RecordingNorm(100)
        attention.k_norm = RecordingNorm(200)

        query, key, value, gate = attention._project_qkv(torch.zeros(1, 3))

        self.assertEqual(attention.q_norm.seen_shape, (1, 2, 2))
        self.assertEqual(attention.k_norm.seen_shape, (1, 1, 2))
        self.assertTrue(torch.equal(query, torch.tensor([[110, 111, 130, 131]])))
        self.assertTrue(torch.equal(key, torch.tensor([[250, 251]])))
        self.assertTrue(torch.equal(value, torch.tensor([[60, 61]])))
        self.assertTrue(torch.equal(gate, torch.tensor([[20, 21, 40, 41]])))
        for tensor in (query, key, value, gate):
            self.assertTrue(tensor.is_contiguous())


class RecordingAttentionCall(nn.Module):
    def __init__(self):
        super().__init__()
        self.arguments = None

    def forward(self, *arguments):
        self.arguments = arguments
        return arguments[1] + 10


class RecordingGateCall(nn.Module):
    def __init__(self):
        super().__init__()
        self.arguments = None

    def forward(self, value, gate):
        self.arguments = (value.clone(), gate.clone())
        return value * gate


class RecordingOutputProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = None

    def forward(self, value):
        self.input = value.clone()
        return value + 1000, None


class ForwardHarness(Qwen3_5Attention):
    def __init__(self):
        nn.Module.__init__(self)
        self.query = torch.tensor([[1.0, 2.0]])
        self.key = torch.tensor([[3.0]])
        self.value = torch.tensor([[4.0]])
        self.gate = torch.tensor([[0.5, 2.0]])
        self.attn = RecordingAttentionCall()
        self.gate_fn = RecordingGateCall()
        self.o_proj = RecordingOutputProjection()

    def _project_qkv(self, hidden_states):
        return self.query, self.key, self.value, self.gate


class QwenAttentionForwardTest(unittest.TestCase):
    def test_forward_gates_attention_output_before_output_projection(self):
        attention = ForwardHarness()
        positions = torch.tensor([7])
        hidden_states = torch.tensor([[9.0]])
        key_cache = torch.tensor([1.0])
        value_cache = torch.tensor([2.0])
        input_metadata = SimpleNamespace(is_profile_run=False)
        cache_event = object()

        output = attention(
            positions,
            hidden_states,
            (key_cache, value_cache),
            input_metadata,
            cache_event,
        )

        expected_arguments = (
            positions,
            attention.query,
            attention.key,
            attention.value,
            key_cache,
            value_cache,
            input_metadata,
            cache_event,
        )
        self.assertEqual(len(attention.attn.arguments), len(expected_arguments))
        for actual, expected in zip(
            attention.attn.arguments,
            expected_arguments,
        ):
            self.assertIs(actual, expected)
        expected_attention = attention.query + 10
        self.assertTrue(torch.equal(attention.gate_fn.arguments[0], expected_attention))
        self.assertTrue(torch.equal(attention.gate_fn.arguments[1], attention.gate))
        expected_gated = expected_attention * attention.gate
        self.assertTrue(torch.equal(attention.o_proj.input, expected_gated))
        self.assertTrue(torch.equal(output, expected_gated + 1000))


if __name__ == "__main__":
    unittest.main()
