import os
import unittest
from unittest.mock import patch

import torch

from minivllm import attention_ops, cache_ops
from minivllm.model_executor.layers.dspark_attention import DraftPagedAttention
from minivllm.spec_decode.draft_metadata import DraftAttentionMetadata
from minivllm.worker.draft_cache import DraftCacheEngine


RUN_CUDA_DSPARK_TESTS = (
    os.environ.get("MINIVLLM_RUN_CUDA_DSPARK_TESTS") == "1"
    and torch.cuda.is_available()
)


def metadata():
    return DraftAttentionMetadata(
        query_lens=[2, 1],
        cu_seqlens_q=torch.tensor([0, 2, 3], dtype=torch.int32),
        context_lens=torch.tensor([5, 4], dtype=torch.int32),
        block_tables=torch.tensor([[0], [1]], dtype=torch.int32),
        slot_mapping=torch.tensor([3, 4, 11], dtype=torch.int32),
    )


class DraftAttentionMetadataTest(unittest.TestCase):
    def test_lengths_and_derived_values(self):
        value = metadata()
        self.assertEqual(value.num_tokens, 3)
        self.assertEqual(value.max_query_len, 2)
        self.assertEqual(value.max_context_len, 5)

    def test_slot_count_must_match_queries(self):
        with self.assertRaisesRegex(ValueError, "slot_mapping"):
            DraftAttentionMetadata(
                query_lens=[2],
                cu_seqlens_q=torch.tensor([0, 2]),
                context_lens=torch.tensor([2]),
                block_tables=torch.tensor([[0]]),
                slot_mapping=torch.tensor([0]),
            )


class DraftPagedAttentionContractTest(unittest.TestCase):
    def test_launcher_uses_noncausal_query_mode(self):
        operation = DraftPagedAttention(2, 1, 4, 0.5)
        query = torch.randn(3, 2, 4)
        key = torch.randn(3, 1, 4)
        value = torch.randn(3, 1, 4)
        key_cache = torch.empty(2, 1, 1, 8, 4)
        value_cache = torch.empty(2, 1, 4, 8)

        with patch(
            "minivllm.model_executor.layers.dspark_attention."
            "cache_ops.reshape_and_cache"
        ) as write, patch(
            "minivllm.model_executor.layers.dspark_attention."
            "attention_ops.varlen_query_cached_kv_attention"
        ) as attention:
            output = operation(
                query, key, value, (key_cache, value_cache), metadata()
            )

        self.assertEqual(output.shape, query.shape)
        write.assert_called_once()
        self.assertIs(attention.call_args.args[-1], False)


class DraftCacheEngineTest(unittest.TestCase):
    def test_cache_mirrors_target_physical_blocks(self):
        cache = DraftCacheEngine(
            num_layers=2,
            num_gpu_blocks=3,
            num_cpu_blocks=0,
            block_size=8,
            num_kv_heads=1,
            head_dim=8,
            dtype=torch.float16,
            device="cpu",
        )

        self.assertEqual(len(cache.gpu_cache), 2)
        self.assertEqual(cache.gpu_cache[0][0].shape, (3, 1, 1, 8, 8))
        self.assertEqual(cache.gpu_cache[0][1].shape, (3, 1, 8, 8))
        self.assertEqual(cache.bytes_per_block, 512)


@unittest.skipUnless(
    RUN_CUDA_DSPARK_TESTS,
    "set MINIVLLM_RUN_CUDA_DSPARK_TESTS=1 after rebuilding extensions",
)
class DraftPagedAttentionCudaTest(unittest.TestCase):
    def test_noncausal_block_reads_every_current_position(self):
        torch.manual_seed(11)
        dtype = torch.float16
        num_heads = 4
        num_kv_heads = 2
        head_dim = 64
        block_size = 8
        context_len = 5
        query_len = 3
        scale = head_dim ** -0.5
        keys = torch.randn(
            context_len, num_kv_heads, head_dim, dtype=dtype, device="cuda"
        )
        values = torch.randn_like(keys)
        x = 16 // keys.element_size()
        key_cache = torch.zeros(
            1, num_kv_heads, head_dim // x, block_size, x,
            dtype=dtype, device="cuda",
        )
        value_cache = torch.zeros(
            1, num_kv_heads, head_dim, block_size,
            dtype=dtype, device="cuda",
        )
        cache_ops.reshape_and_cache(
            keys,
            values,
            key_cache,
            value_cache,
            torch.arange(context_len, dtype=torch.int32, device="cuda"),
        )
        query = torch.randn(
            query_len, num_heads, head_dim, dtype=dtype, device="cuda"
        )
        output = torch.empty_like(query)

        attention_ops.varlen_query_cached_kv_attention(
            output,
            query,
            key_cache,
            value_cache,
            torch.tensor([0, query_len], dtype=torch.int32, device="cuda"),
            query_len,
            scale,
            torch.tensor([[0]], dtype=torch.int32, device="cuda"),
            torch.tensor([context_len], dtype=torch.int32, device="cuda"),
            block_size,
            context_len,
            False,
        )
        torch.cuda.synchronize()

        expected = torch.empty_like(output)
        queries_per_kv = num_heads // num_kv_heads
        for token_index in range(query_len):
            for query_head in range(num_heads):
                kv_head = query_head // queries_per_kv
                scores = torch.matmul(
                    query[token_index, query_head].float(),
                    keys[:, kv_head].float().T,
                ) * scale
                expected[token_index, query_head] = torch.matmul(
                    torch.softmax(scores, dim=-1),
                    values[:, kv_head].float(),
                ).to(dtype)
        torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
