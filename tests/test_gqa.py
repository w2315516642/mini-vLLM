import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from minivllm.model_executor.layers.attention import PagedAttention
from minivllm.model_executor.models import llama as llama_module
from minivllm.model_executor.models.llama import (
    LlamaAttention,
    _load_qkv_weight,
    _split_qkv,
)
from minivllm.worker.cache_engine import CacheEngine


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
        num_kv_heads=None,
        **kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.scale = scale
        self.rotary_dim = rotary_dim


class GQALayoutTest(unittest.TestCase):
    def setUp(self):
        self.attention = PagedAttention(
            num_heads=8,
            num_kv_heads=2,
            head_size=64,
            scale=64 ** -0.5,
        )

    def test_compact_qkv_uses_independent_head_dimensions(self):
        num_tokens = 3
        query = torch.arange(num_tokens * 8 * 64).reshape(num_tokens, -1)
        key = torch.arange(num_tokens * 2 * 64).reshape(num_tokens, -1)
        value = key + 10_000

        reshaped_q, reshaped_k, reshaped_v = self.attention._reshape_qkv(
            query,
            key,
            value,
        )

        self.assertEqual(reshaped_q.shape, (num_tokens, 8, 64))
        self.assertEqual(reshaped_k.shape, (num_tokens, 2, 64))
        self.assertEqual(reshaped_v.shape, (num_tokens, 2, 64))
        self.assertTrue(torch.equal(reshaped_q.reshape_as(query), query))
        self.assertTrue(torch.equal(reshaped_k.reshape_as(key), key))
        self.assertTrue(torch.equal(reshaped_v.reshape_as(value), value))

    def test_grouped_prefill_broadcasts_compact_kv(self):
        query = torch.randn(5, 8, 64)
        key = torch.randn(5, 2, 64)
        value = torch.randn(5, 2, 64)

        grouped_q, grouped_k, grouped_v = (
            self.attention._grouped_prefill_inputs(query, key, value)
        )

        self.assertEqual(grouped_q.shape, (1, 5, 2, 4, 64))
        self.assertEqual(grouped_k.shape, (1, 5, 2, 4, 64))
        self.assertEqual(grouped_v.shape, (1, 5, 2, 4, 64))
        self.assertEqual(grouped_k.stride(-2), 0)
        self.assertEqual(grouped_v.stride(-2), 0)
        self.assertTrue(torch.equal(grouped_k[0, :, :, 0], key))
        self.assertTrue(torch.equal(grouped_k[0, :, :, 3], key))

    def test_mha_remains_the_default_layout(self):
        attention = PagedAttention(
            num_heads=4,
            head_size=64,
            scale=64 ** -0.5,
        )

        self.assertEqual(attention.num_kv_heads, 4)
        self.assertEqual(attention.num_queries_per_kv, 1)

    def test_invalid_gqa_ratio_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "grouped"):
            PagedAttention(
                num_heads=6,
                num_kv_heads=4,
                head_size=64,
                scale=64 ** -0.5,
            )


class PackedQKVTest(unittest.TestCase):
    def test_packed_projection_is_split_by_q_and_kv_widths(self):
        q_size = 8
        kv_size = 2
        packed = torch.arange(2 * (q_size + 2 * kv_size)).reshape(2, -1)

        query, key, value = _split_qkv(packed, q_size, kv_size)

        self.assertTrue(torch.equal(query, packed[:, :q_size]))
        self.assertTrue(
            torch.equal(key, packed[:, q_size:q_size + kv_size])
        )
        self.assertTrue(torch.equal(value, packed[:, -kv_size:]))

    def test_tp_weight_shards_land_in_their_local_packed_segments(self):
        q_size = 4
        kv_size = 2
        param = torch.zeros(q_size + 2 * kv_size, 1)
        query_weight = torch.arange(8, dtype=torch.float32).reshape(-1, 1)
        key_weight = (
            torch.arange(4, dtype=torch.float32).reshape(-1, 1) + 100
        )
        value_weight = (
            torch.arange(4, dtype=torch.float32).reshape(-1, 1) + 200
        )

        for shard_id, loaded_weight in (
            ("q", query_weight),
            ("k", key_weight),
            ("v", value_weight),
        ):
            _load_qkv_weight(
                param,
                loaded_weight,
                shard_id,
                tensor_model_parallel_rank=1,
                q_size=q_size,
                kv_size=kv_size,
            )

        expected = torch.tensor(
            [[4], [5], [6], [7], [102], [103], [202], [203]],
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(param, expected))


class GQAIntegrationTest(unittest.TestCase):
    def test_llama_projection_and_attention_receive_local_gqa_sizes(self):
        with patch.object(
            llama_module,
            "get_tensor_model_parallel_world_size",
            return_value=2,
        ), patch.object(
            llama_module,
            "ColumnParallelLinear",
            RecordingColumnParallelLinear,
        ), patch.object(
            llama_module,
            "RowParallelLinear",
            RecordingRowParallelLinear,
        ), patch.object(
            llama_module,
            "PagedAttentionWithRoPE",
            RecordingPagedAttention,
        ):
            attention = LlamaAttention(
                hidden_size=5120,
                num_heads=24,
                num_kv_heads=4,
                head_dim=256,
            )

        self.assertEqual(attention.num_heads, 12)
        self.assertEqual(attention.num_kv_heads, 2)
        self.assertEqual(attention.q_size, 3072)
        self.assertEqual(attention.kv_size, 512)
        self.assertEqual(attention.qkv_proj.output_size, 8192)
        self.assertEqual(attention.o_proj.input_size, 6144)
        self.assertEqual(attention.attn.num_heads, 12)
        self.assertEqual(attention.attn.num_kv_heads, 2)

    def test_cache_block_size_uses_kv_heads_instead_of_query_heads(self):
        model_config = SimpleNamespace(
            dtype=torch.float16,
            get_head_size=lambda: 64,
            get_num_layers=lambda parallel_config: 3,
            get_num_heads=lambda parallel_config: 8,
            get_num_kv_heads=lambda parallel_config: 2,
        )
        parallel_config = SimpleNamespace(tensor_parallel_size=1)

        size_bytes = CacheEngine.get_cache_block_size(
            block_size=4,
            model_config=model_config,
            parallel_config=parallel_config,
        )

        expected = 3 * 2 * 4 * 2 * 64 * torch.float16.itemsize
        self.assertEqual(size_bytes, expected)


if __name__ == "__main__":
    unittest.main()
