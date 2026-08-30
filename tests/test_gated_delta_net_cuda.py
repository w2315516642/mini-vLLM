import os
import unittest

import torch

from minivllm.model_executor.layers.gated_delta_net import (
    causal_depthwise_conv1d_reference,
)
from minivllm.model_executor.layers.gated_delta_net_cuda import (
    causal_conv1d_update,
    gated_delta_rule_decode,
    gated_delta_rule_prefill,
    prepare_gated_delta_qk,
)


RUN_CUDA_GDN_TESTS = (
    os.environ.get("MINIVLLM_RUN_CUDA_GDN_TESTS") == "1"
    and torch.cuda.is_available()
)


def _prepared_delta_reference(
    query,
    key,
    value,
    log_decay,
    beta,
    initial_state,
):
    state = initial_state.clone().float()
    output = torch.empty_like(value, dtype=torch.float32)
    for token_idx in range(query.shape[1]):
        q = query[:, token_idx].float()
        k = key[:, token_idx].float()
        v = value[:, token_idx].float()
        decay = log_decay[:, token_idx].float().exp()[..., None, None]
        state *= decay
        old_value = torch.einsum("bhd,bhdv->bhv", k, state)
        delta = beta[:, token_idx].float()[..., None] * (v - old_value)
        state += torch.einsum("bhd,bhv->bhdv", k, delta)
        output[:, token_idx] = torch.einsum("bhd,bhdv->bhv", q, state)
    return output.to(value.dtype), state


def _cuda_dtypes():
    dtypes = [torch.float16, torch.float32]
    if torch.cuda.get_device_capability()[0] >= 8:
        dtypes.append(torch.bfloat16)
    return dtypes


@unittest.skipUnless(
    RUN_CUDA_GDN_TESTS,
    "set MINIVLLM_RUN_CUDA_GDN_TESTS=1 after rebuilding extensions",
)
class GatedDeltaNetCudaNumericalTest(unittest.TestCase):
    def test_causal_conv_update_matches_reference(self):
        torch.manual_seed(62)
        for dtype in _cuda_dtypes():
            with self.subTest(dtype=dtype):
                projected = torch.randn(2, 17, device="cuda", dtype=dtype)
                weight = torch.randn(17, 4, device="cuda", dtype=dtype)
                initial_state = torch.randn(
                    2,
                    17,
                    4,
                    device="cuda",
                    dtype=torch.float32,
                )
                actual_state = initial_state.clone()

                actual = causal_conv1d_update(
                    projected,
                    actual_state,
                    weight,
                )
                expected, expected_state = causal_depthwise_conv1d_reference(
                    projected[:, None, :],
                    weight,
                    initial_state,
                )
                torch.cuda.synchronize()

                torch.testing.assert_close(
                    actual,
                    expected[:, 0],
                    rtol=3e-2,
                    atol=3e-2,
                )
                torch.testing.assert_close(
                    actual_state,
                    expected_state,
                    rtol=1e-5,
                    atol=1e-5,
                )

    def test_decode_matches_prepared_reference(self):
        torch.manual_seed(63)
        for dtype in _cuda_dtypes():
            with self.subTest(dtype=dtype):
                query = torch.randn(2, 3, 7, device="cuda", dtype=dtype)
                key = torch.randn_like(query)
                query, key = prepare_gated_delta_qk(query, key)
                value = torch.randn(2, 3, 19, device="cuda", dtype=dtype)
                log_decay = -torch.rand(
                    2,
                    3,
                    device="cuda",
                    dtype=torch.float32,
                )
                beta = torch.rand(2, 3, device="cuda", dtype=dtype)
                initial_state = torch.randn(
                    2,
                    3,
                    7,
                    19,
                    device="cuda",
                    dtype=torch.float32,
                )
                actual_state = initial_state.clone()

                actual = gated_delta_rule_decode(
                    query,
                    key,
                    value,
                    log_decay,
                    beta,
                    actual_state,
                )
                expected, expected_state = _prepared_delta_reference(
                    query[:, None],
                    key[:, None],
                    value[:, None],
                    log_decay[:, None],
                    beta[:, None],
                    initial_state,
                )
                torch.cuda.synchronize()

                torch.testing.assert_close(
                    actual,
                    expected[:, 0],
                    rtol=4e-2,
                    atol=4e-2,
                )
                torch.testing.assert_close(
                    actual_state,
                    expected_state,
                    rtol=2e-3,
                    atol=2e-3,
                )

    def test_prefill_matches_reference_for_partial_final_chunks(self):
        torch.manual_seed(64)
        dtype = torch.float16
        query = torch.randn(2, 9, 3, 5, device="cuda", dtype=dtype)
        key = torch.randn_like(query)
        query, key = prepare_gated_delta_qk(query, key)
        value = torch.randn(2, 9, 3, 13, device="cuda", dtype=dtype)
        log_decay = -torch.rand(
            2,
            9,
            3,
            device="cuda",
            dtype=torch.float32,
        )
        beta = torch.rand(2, 9, 3, device="cuda", dtype=dtype)
        initial_state = torch.randn(
            2,
            3,
            5,
            13,
            device="cuda",
            dtype=torch.float32,
        )
        expected, expected_state = _prepared_delta_reference(
            query,
            key,
            value,
            log_decay,
            beta,
            initial_state,
        )

        for chunk_size in (1, 4, 16):
            with self.subTest(chunk_size=chunk_size):
                actual_state = initial_state.clone()
                actual = gated_delta_rule_prefill(
                    query,
                    key,
                    value,
                    log_decay,
                    beta,
                    actual_state,
                    chunk_size,
                )
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=4e-2,
                    atol=4e-2,
                )
                torch.testing.assert_close(
                    actual_state,
                    expected_state,
                    rtol=2e-3,
                    atol=2e-3,
                )

    def test_prefill_final_state_continues_into_decode(self):
        torch.manual_seed(65)
        dtype = torch.float32
        query = torch.randn(1, 6, 2, 4, device="cuda", dtype=dtype)
        key = torch.randn_like(query)
        query, key = prepare_gated_delta_qk(query, key)
        value = torch.randn(1, 6, 2, 6, device="cuda", dtype=dtype)
        log_decay = -torch.rand(1, 6, 2, device="cuda", dtype=dtype)
        beta = torch.rand(1, 6, 2, device="cuda", dtype=dtype)
        initial_state = torch.randn(
            1,
            2,
            4,
            6,
            device="cuda",
            dtype=torch.float32,
        )

        actual_state = initial_state.clone()
        prefill = gated_delta_rule_prefill(
            query[:, :-1],
            key[:, :-1],
            value[:, :-1],
            log_decay[:, :-1],
            beta[:, :-1],
            actual_state,
            chunk_size=3,
        )
        decode = gated_delta_rule_decode(
            query[:, -1],
            key[:, -1],
            value[:, -1],
            log_decay[:, -1],
            beta[:, -1],
            actual_state,
        )
        expected, expected_state = _prepared_delta_reference(
            query,
            key,
            value,
            log_decay,
            beta,
            initial_state,
        )
        torch.cuda.synchronize()

        actual = torch.cat((prefill, decode[:, None]), dim=1)
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(
            actual_state,
            expected_state,
            rtol=2e-4,
            atol=2e-4,
        )

    def test_binding_rejects_non_fp32_state(self):
        projected = torch.randn(1, 4, device="cuda", dtype=torch.float16)
        state = torch.zeros(1, 4, 3, device="cuda", dtype=torch.float16)
        weight = torch.randn(4, 3, device="cuda", dtype=torch.float16)

        with self.assertRaisesRegex(RuntimeError, "conv_state.*float32"):
            causal_conv1d_update(projected, state, weight)


if __name__ == "__main__":
    unittest.main()
