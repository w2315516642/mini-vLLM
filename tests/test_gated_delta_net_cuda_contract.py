import unittest
from unittest.mock import patch

import torch

import minivllm.model_executor.layers.gated_delta_net_cuda as gdn_cuda


class _FakeGatedDeltaNetOps:
    def __init__(self):
        self.calls = []

    def causal_conv1d_update(self, output, projected, state, weight):
        self.calls.append(("conv", state, weight))
        output.copy_(projected + 1)
        state.add_(1)

    def gated_delta_rule_decode(
        self,
        output,
        query,
        key,
        value,
        log_decay,
        beta,
        state,
    ):
        self.calls.append(("decode", state, query, key, log_decay, beta))
        output.copy_(value + 2)
        state.add_(2)

    def gated_delta_rule_prefill(
        self,
        output,
        query,
        key,
        value,
        log_decay,
        beta,
        state,
        chunk_size,
    ):
        self.calls.append(("prefill", state, chunk_size))
        output.copy_(value + 3)
        state.add_(3)


class GatedDeltaNetCudaWrapperTest(unittest.TestCase):
    def test_prepare_qk_adds_epsilon_to_squared_norm_near_zero(self):
        query = torch.tensor([[[1e-4, 0.0], [0.0, 0.0]]])
        actual_q, actual_k = gdn_cuda.prepare_gated_delta_qk(query, query)
        expected_k = query / torch.sqrt(torch.tensor([[[1.01e-6], [1e-6]]]))
        torch.testing.assert_close(actual_k, expected_k)
        torch.testing.assert_close(actual_q, expected_k / (2 ** 0.5))
        self.assertLess(actual_k[0, 0, 0].item(), 0.1)

    def test_prepare_qk_matches_fp32_normalization_and_query_scale(self):
        torch.manual_seed(61)
        query = torch.randn(2, 3, 4, 5, dtype=torch.float16)
        key = torch.randn_like(query)

        actual_query, actual_key = gdn_cuda.prepare_gated_delta_qk(
            query,
            key,
        )

        query_fp32 = query.float()
        key_fp32 = key.float()
        expected_query = query_fp32 * torch.rsqrt(
            query_fp32.square().sum(dim=-1, keepdim=True) + 1e-6)
        expected_query *= query.shape[-1] ** -0.5
        expected_key = key_fp32 * torch.rsqrt(
            key_fp32.square().sum(dim=-1, keepdim=True) + 1e-6)
        torch.testing.assert_close(actual_query, expected_query.half())
        torch.testing.assert_close(actual_key, expected_key.half())
        self.assertTrue(actual_query.is_contiguous())
        self.assertTrue(actual_key.is_contiguous())

    def test_prepare_qk_rejects_incompatible_inputs(self):
        query = torch.randn(2, 3, 4)

        with self.assertRaisesRegex(ValueError, "same shape"):
            gdn_cuda.prepare_gated_delta_qk(query, torch.randn(2, 3, 5))
        with self.assertRaisesRegex(ValueError, "layouts"):
            gdn_cuda.prepare_gated_delta_qk(query[0], query[0])
        with self.assertRaisesRegex(ValueError, "eps"):
            gdn_cuda.prepare_gated_delta_qk(query, query, eps=0)

    def test_conv_wrapper_preserves_in_place_state_contract(self):
        fake_ops = _FakeGatedDeltaNetOps()
        projected = torch.randn(2, 4)
        state = torch.zeros(2, 4, 3)
        weight = torch.randn(4, 3)

        with patch.object(gdn_cuda, "_load_ops", return_value=fake_ops):
            output = gdn_cuda.causal_conv1d_update(
                projected,
                state,
                weight,
            )

        torch.testing.assert_close(output, projected + 1)
        torch.testing.assert_close(state, torch.ones_like(state))
        self.assertIs(fake_ops.calls[0][1], state)

    def test_decode_wrapper_preserves_in_place_state_contract(self):
        fake_ops = _FakeGatedDeltaNetOps()
        query = torch.randn(2, 3, 4)
        key = torch.randn_like(query)
        value = torch.randn(2, 3, 5)
        log_decay = torch.randn(2, 3)
        beta = torch.randn(2, 3)
        state = torch.zeros(2, 3, 4, 5)

        with patch.object(gdn_cuda, "_load_ops", return_value=fake_ops):
            output = gdn_cuda.gated_delta_rule_decode(
                query,
                key,
                value,
                log_decay,
                beta,
                state,
            )

        torch.testing.assert_close(output, value + 2)
        torch.testing.assert_close(state, torch.full_like(state, 2))
        self.assertIs(fake_ops.calls[0][1], state)

    def test_prefill_wrapper_forwards_chunk_size_and_state(self):
        fake_ops = _FakeGatedDeltaNetOps()
        query = torch.randn(2, 7, 3, 4)
        key = torch.randn_like(query)
        value = torch.randn(2, 7, 3, 5)
        log_decay = torch.randn(2, 7, 3)
        beta = torch.randn(2, 7, 3)
        state = torch.zeros(2, 3, 4, 5)

        with patch.object(gdn_cuda, "_load_ops", return_value=fake_ops):
            output = gdn_cuda.gated_delta_rule_prefill(
                query,
                key,
                value,
                log_decay,
                beta,
                state,
                chunk_size=4,
            )

        torch.testing.assert_close(output, value + 3)
        torch.testing.assert_close(state, torch.full_like(state, 3))
        self.assertEqual(fake_ops.calls[0][2], 4)
        self.assertIs(fake_ops.calls[0][1], state)

    def test_prefill_rejects_invalid_chunk_size_before_loading_ops(self):
        inputs = (
            torch.randn(1, 2, 1, 3),
            torch.randn(1, 2, 1, 3),
            torch.randn(1, 2, 1, 4),
            torch.randn(1, 2, 1),
            torch.randn(1, 2, 1),
            torch.zeros(1, 1, 3, 4),
        )

        with patch.object(gdn_cuda, "_load_ops") as load_ops:
            with self.assertRaisesRegex(ValueError, "positive integer"):
                gdn_cuda.gated_delta_rule_prefill(*inputs, chunk_size=0)
            load_ops.assert_not_called()

    def test_extension_is_loaded_only_when_an_operator_runs(self):
        with patch.object(gdn_cuda, "import_module") as import_module:
            gdn_cuda._load_ops()

        import_module.assert_called_once_with("minivllm.gated_delta_net_ops")


if __name__ == "__main__":
    unittest.main()
