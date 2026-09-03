import os
import unittest
from itertools import product
from unittest.mock import patch

import torch

from minivllm import attention_ops, cache_ops, pos_encoding_ops
from minivllm.model_executor.layers import attention as attention_module
from minivllm.model_executor.layers.attention import PagedAttention
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask


RUN_CUDA_GQA_TESTS = (
    os.environ.get("MINIVLLM_RUN_CUDA_GQA_TESTS") == "1"
    and torch.cuda.is_available()
)


def _reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2))
    probs = torch.softmax(scores * scale, dim=-1)
    return torch.matmul(probs, value.float()).to(query.dtype)


def _pack_cache(
    keys: torch.Tensor,
    values: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_blocks: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_kv_heads = keys.shape[1]
    head_size = keys.shape[2]
    x = 16 // keys.element_size()
    key_cache = torch.zeros(
        num_blocks,
        num_kv_heads,
        head_size // x,
        block_size,
        x,
        dtype=keys.dtype,
        device="cuda",
    )
    value_cache = torch.zeros(
        num_blocks,
        num_kv_heads,
        head_size,
        block_size,
        dtype=values.dtype,
        device="cuda",
    )
    cache_ops.reshape_and_cache(
        keys,
        values,
        key_cache,
        value_cache,
        slot_mapping,
    )
    return key_cache, value_cache


@unittest.skipUnless(
    RUN_CUDA_GQA_TESTS,
    "set MINIVLLM_RUN_CUDA_GQA_TESTS=1 after rebuilding CUDA extensions",
)
class GQACudaKernelTest(unittest.TestCase):
    def test_fresh_prefill_without_cutlass_matches_reference(self):
        torch.manual_seed(4)
        dtypes = [torch.float16]
        if torch.cuda.is_bf16_supported():
            dtypes.append(torch.bfloat16)

        # Simulate CUTLASS being unavailable without pretending this GPU is SM120.
        # The automatic dispatcher must choose another real CUDA implementation.
        with patch.object(
            attention_module.xops.fmha.cutlass.FwOp,
            "not_supported_reasons",
            return_value=["CUTLASS disabled for dispatch regression test"],
        ):
            for dtype, head_size, heads, prompt_lens in product(
                dtypes, (64, 256), ((2, 2), (2, 1), (8, 2)), ((4,), (3, 5)),
            ):
                with self.subTest(
                    dtype=dtype, head_size=head_size,
                    heads=heads, prompt_lens=prompt_lens,
                ):
                    num_heads, num_kv_heads = heads
                    scale = head_size ** -0.5
                    attention = PagedAttention(
                        num_heads, head_size, scale, num_kv_heads,
                    )
                    query = torch.randn(
                        sum(prompt_lens), num_heads, head_size,
                        device="cuda", dtype=dtype,
                    )
                    key = torch.randn(
                        sum(prompt_lens), num_kv_heads, head_size,
                        device="cuda", dtype=dtype,
                    )
                    value = torch.randn_like(key)
                    output = torch.empty_like(query)
                    attention.multi_query_kv_attention(
                        output, query, key, value,
                        BlockDiagonalCausalMask.from_seqlens(list(prompt_lens)),
                    )

                    # Each prompt starts a new causal context; no cross-prompt KV.
                    expected = torch.empty_like(output)
                    start = 0
                    for length in prompt_lens:
                        for offset in range(length):
                            token = start + offset
                            for head in range(num_heads):
                                kv_head = head // (num_heads // num_kv_heads)
                                expected[token, head] = _reference_attention(
                                    query[token, head].unsqueeze(0),
                                    key[start:token + 1, kv_head],
                                    value[start:token + 1, kv_head],
                                    scale,
                                ).squeeze(0)
                        start += length
                    torch.testing.assert_close(
                        output, expected, rtol=2e-2, atol=2e-2,
                    )

    def test_decode_reads_compact_kv_cache(self):
        torch.manual_seed(1)
        dtype = torch.float16
        num_query_heads = 4
        num_kv_heads = 2
        head_size = 64
        block_size = 8
        context_lens_list = [5, 7]
        scale = head_size ** -0.5

        sequence_keys = [
            torch.randn(
                length,
                num_kv_heads,
                head_size,
                dtype=dtype,
                device="cuda",
            )
            for length in context_lens_list
        ]
        sequence_values = [torch.randn_like(keys) for keys in sequence_keys]
        keys = torch.cat(sequence_keys)
        values = torch.cat(sequence_values)
        slot_mapping = torch.tensor(
            list(range(5)) + list(range(block_size, block_size + 7)),
            dtype=torch.int32,
            device="cuda",
        )
        key_cache, value_cache = _pack_cache(
            keys,
            values,
            slot_mapping,
            num_blocks=2,
            block_size=block_size,
        )

        query = torch.randn(
            2,
            num_query_heads,
            head_size,
            dtype=dtype,
            device="cuda",
        )
        output = torch.empty_like(query)
        block_tables = torch.tensor(
            [[0], [1]], dtype=torch.int32, device="cuda"
        )
        context_lens = torch.tensor(
            context_lens_list, dtype=torch.int32, device="cuda"
        )

        attention_ops.single_query_cached_kv_attention(
            output,
            query,
            key_cache,
            value_cache,
            scale,
            block_tables,
            context_lens,
            block_size,
            max(context_lens_list),
        )
        torch.cuda.synchronize()

        expected = torch.empty_like(output)
        queries_per_kv = num_query_heads // num_kv_heads
        for seq_idx, (seq_key, seq_value) in enumerate(
            zip(sequence_keys, sequence_values)
        ):
            for query_head in range(num_query_heads):
                kv_head = query_head // queries_per_kv
                expected[seq_idx, query_head] = _reference_attention(
                    query[seq_idx, query_head].reshape(1, head_size),
                    seq_key[:, kv_head],
                    seq_value[:, kv_head],
                    scale,
                ).squeeze(0)

        torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)

    def test_cached_prefill_reads_compact_kv_cache(self):
        torch.manual_seed(2)
        dtype = torch.float16
        num_query_heads = 4
        num_kv_heads = 2
        head_size = 64
        block_size = 8
        context_len = 5
        query_len = 2
        scale = head_size ** -0.5

        keys = torch.randn(
            context_len,
            num_kv_heads,
            head_size,
            dtype=dtype,
            device="cuda",
        )
        values = torch.randn_like(keys)
        key_cache, value_cache = _pack_cache(
            keys,
            values,
            torch.arange(context_len, dtype=torch.int32, device="cuda"),
            num_blocks=1,
            block_size=block_size,
        )
        query = torch.randn(
            query_len,
            num_query_heads,
            head_size,
            dtype=dtype,
            device="cuda",
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
            True,
        )
        torch.cuda.synchronize()

        expected = torch.empty_like(output)
        queries_per_kv = num_query_heads // num_kv_heads
        query_start = context_len - query_len
        for token_idx in range(query_len):
            visible_tokens = query_start + token_idx + 1
            for query_head in range(num_query_heads):
                kv_head = query_head // queries_per_kv
                expected[token_idx, query_head] = _reference_attention(
                    query[token_idx, query_head].reshape(1, head_size),
                    keys[:visible_tokens, kv_head],
                    values[:visible_tokens, kv_head],
                    scale,
                ).squeeze(0)

        torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)

    def test_rope_supports_different_query_and_kv_strides(self):
        torch.manual_seed(3)
        dtype = torch.float16
        num_tokens = 3
        num_query_heads = 4
        num_kv_heads = 2
        head_size = 64
        positions = torch.tensor([0, 2, 5], dtype=torch.int64, device="cuda")

        query = torch.randn(
            num_tokens,
            num_query_heads * head_size,
            dtype=dtype,
            device="cuda",
        )
        key = torch.randn(
            num_tokens,
            num_kv_heads * head_size,
            dtype=dtype,
            device="cuda",
        )
        inv_freq = 1.0 / (
            10_000
            ** (torch.arange(0, head_size, 2, device="cuda") / head_size)
        )
        frequency = torch.einsum(
            "i,j->ij",
            torch.arange(8, device="cuda").float(),
            inv_freq.float(),
        )
        cos_sin_cache = torch.cat(
            (frequency.cos(), frequency.sin()), dim=-1
        ).to(dtype)

        expected_query = self._rope_reference(
            query,
            positions,
            cos_sin_cache,
            head_size,
        )
        expected_key = self._rope_reference(
            key,
            positions,
            cos_sin_cache,
            head_size,
        )

        pos_encoding_ops.rotary_embedding_neox(
            positions,
            query,
            key,
            head_size,
            cos_sin_cache,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(query, expected_query, rtol=2e-3, atol=2e-3)
        torch.testing.assert_close(key, expected_key, rtol=2e-3, atol=2e-3)

    @staticmethod
    def _rope_reference(
        tensor: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        head_size: int,
    ) -> torch.Tensor:
        shaped = tensor.reshape(tensor.shape[0], -1, head_size).float()
        half = head_size // 2
        cos_sin = cos_sin_cache[positions].float()
        cos = cos_sin[:, None, :half]
        sin = cos_sin[:, None, half:]
        first = shaped[..., :half]
        second = shaped[..., half:]
        rotated = torch.cat(
            (first * cos - second * sin, second * cos + first * sin),
            dim=-1,
        )
        return rotated.reshape_as(tensor).to(tensor.dtype)


if __name__ == "__main__":
    unittest.main()
