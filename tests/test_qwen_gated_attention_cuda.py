import os
import unittest

import torch

from minivllm import activation_ops, attention_ops, cache_ops
from minivllm.model_executor.layers.activation import SigmoidAndMul


RUN_CUDA_QWEN_ATTENTION_TESTS = (
    os.environ.get("MINIVLLM_RUN_CUDA_QWEN_ATTENTION_TESTS") == "1"
    and torch.cuda.is_available()
)


def _reference_attention(query, key, value, scale):
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2))
    probs = torch.softmax(scores * scale, dim=-1)
    return torch.matmul(probs, value.float()).to(query.dtype)


@unittest.skipUnless(
    RUN_CUDA_QWEN_ATTENTION_TESTS,
    "set MINIVLLM_RUN_CUDA_QWEN_ATTENTION_TESTS=1 after rebuilding extensions",
)
class QwenGatedAttentionCudaTest(unittest.TestCase):
    def test_sigmoid_and_mul_matches_pytorch(self):
        torch.manual_seed(41)
        dtypes = [torch.float16, torch.float32]
        if torch.cuda.get_device_capability()[0] >= 8:
            dtypes.append(torch.bfloat16)

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                value = torch.randn(3, 513, device="cuda", dtype=dtype)
                gate = torch.randn_like(value) * 3

                actual = SigmoidAndMul()(value, gate)
                expected = (
                    value.float() * torch.sigmoid(gate.float())
                ).to(dtype)
                torch.cuda.synchronize()

                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=2e-3,
                    atol=2e-3,
                )

    def test_sigmoid_and_mul_rejects_mismatched_shapes(self):
        value = torch.ones(2, 8, device="cuda", dtype=torch.float16)
        gate = torch.ones(2, 7, device="cuda", dtype=torch.float16)

        with self.assertRaisesRegex(RuntimeError, "shape"):
            SigmoidAndMul()(value, gate)

        out = torch.empty(2, 7, device="cuda", dtype=torch.float16)
        with self.assertRaisesRegex(RuntimeError, "shape"):
            activation_ops.sigmoid_and_mul(out, value, value)

    def test_cached_attention_launchers_accept_qwen_head_size(self):
        torch.manual_seed(42)
        dtype = torch.float16
        num_query_heads = 2
        num_kv_heads = 1
        head_size = 256
        block_size = 8
        context_len = 5
        scale = head_size ** -0.5

        key = torch.randn(
            context_len,
            num_kv_heads,
            head_size,
            device="cuda",
            dtype=dtype,
        )
        value = torch.randn_like(key)
        x = 16 // key.element_size()
        key_cache = torch.zeros(
            1,
            num_kv_heads,
            head_size // x,
            block_size,
            x,
            device="cuda",
            dtype=dtype,
        )
        value_cache = torch.zeros(
            1,
            num_kv_heads,
            head_size,
            block_size,
            device="cuda",
            dtype=dtype,
        )
        cache_ops.reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            torch.arange(context_len, device="cuda", dtype=torch.int32),
        )

        query = torch.randn(
            1,
            num_query_heads,
            head_size,
            device="cuda",
            dtype=dtype,
        )
        output = torch.empty_like(query)
        attention_ops.single_query_cached_kv_attention(
            output,
            query,
            key_cache,
            value_cache,
            scale,
            torch.tensor([[0]], device="cuda", dtype=torch.int32),
            torch.tensor([context_len], device="cuda", dtype=torch.int32),
            block_size,
            context_len,
        )
        torch.cuda.synchronize()

        expected = torch.empty_like(output)
        for query_head in range(num_query_heads):
            expected[0, query_head] = _reference_attention(
                query[0, query_head].reshape(1, head_size),
                key[:, 0],
                value[:, 0],
                scale,
            ).squeeze(0)
        torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)

        query_len = 2
        prefill_query = torch.randn(
            query_len,
            num_query_heads,
            head_size,
            device="cuda",
            dtype=dtype,
        )
        prefill_output = torch.empty_like(prefill_query)
        attention_ops.varlen_query_cached_kv_attention(
            prefill_output,
            prefill_query,
            key_cache,
            value_cache,
            torch.tensor(
                [0, query_len], device="cuda", dtype=torch.int32
            ),
            query_len,
            scale,
            torch.tensor([[0]], device="cuda", dtype=torch.int32),
            torch.tensor([context_len], device="cuda", dtype=torch.int32),
            block_size,
            context_len,
        )
        torch.cuda.synchronize()

        expected_prefill = torch.empty_like(prefill_output)
        query_start = context_len - query_len
        for token_idx in range(query_len):
            visible_tokens = query_start + token_idx + 1
            for query_head in range(num_query_heads):
                expected_prefill[token_idx, query_head] = _reference_attention(
                    prefill_query[token_idx, query_head].reshape(
                        1, head_size
                    ),
                    key[:visible_tokens, 0],
                    value[:visible_tokens, 0],
                    scale,
                ).squeeze(0)
        torch.testing.assert_close(
            prefill_output,
            expected_prefill,
            rtol=2e-2,
            atol=2e-2,
        )


if __name__ == "__main__":
    unittest.main()
