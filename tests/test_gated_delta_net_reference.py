import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

import minivllm.model_executor.layers.gated_delta_net as gated_delta_net
from minivllm.model_executor.layers.gated_delta_net import (
    GatedDeltaNetState,
    Qwen3_5GatedDeltaNetReference,
    RMSNormGated,
    causal_depthwise_conv1d_reference,
    recurrent_gated_delta_rule_reference,
)


def _tiny_config(**overrides):
    values = {
        "hidden_size": 8,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 3,
        "linear_value_head_dim": 2,
        "linear_conv_kernel_dim": 3,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _manual_delta_step(query, key, value, log_decay, beta, state):
    query = F.normalize(query.float(), p=2, dim=-1, eps=1e-6)
    query = query * (query.shape[-1] ** -0.5)
    key = F.normalize(key.float(), p=2, dim=-1, eps=1e-6)
    state = state.float() * log_decay.float().exp()[..., None, None]
    old_value = torch.einsum("bhd,bhdv->bhv", key, state)
    delta = (value.float() - old_value) * beta.float()[..., None]
    state = state + torch.einsum("bhd,bhv->bhdv", key, delta)
    output = torch.einsum("bhd,bhdv->bhv", query, state)
    return output, state


class GatedDeltaNetConfigurationTest(unittest.TestCase):
    def test_constructor_exposes_qwen_projection_and_state_dimensions(self):
        layer = Qwen3_5GatedDeltaNetReference(_tiny_config())

        self.assertEqual(layer.key_dim, 6)
        self.assertEqual(layer.value_dim, 8)
        self.assertEqual(layer.conv_dim, 20)
        self.assertEqual(layer.in_proj_qkv.weight.shape, (20, 8))
        self.assertEqual(layer.in_proj_z.weight.shape, (8, 8))
        self.assertEqual(layer.in_proj_b.weight.shape, (4, 8))
        self.assertEqual(layer.in_proj_a.weight.shape, (4, 8))
        self.assertEqual(layer.conv1d.weight.shape, (20, 1, 3))

        state = layer.empty_state(2, torch.device("cpu"))
        self.assertEqual(state.conv_state.shape, (2, 20, 3))
        self.assertEqual(state.recurrent_state.shape, (2, 4, 3, 2))
        self.assertEqual(state.conv_state.dtype, torch.float32)
        self.assertEqual(state.recurrent_state.dtype, torch.float32)

    def test_value_heads_must_be_grouped_by_key_heads(self):
        with self.assertRaisesRegex(ValueError, "value_heads"):
            Qwen3_5GatedDeltaNetReference(
                _tiny_config(
                    linear_num_key_heads=3,
                    linear_num_value_heads=4,
                )
            )


class CausalDepthwiseConvolutionTest(unittest.TestCase):
    def test_small_sequence_matches_manual_causal_windows(self):
        projected = torch.tensor([[[1.0], [2.0], [-1.0]]])
        weight = torch.tensor([[0.5, -1.0, 2.0]])

        output, final_state = causal_depthwise_conv1d_reference(
            projected,
            weight,
        )

        raw = torch.tensor([2.0, 3.0, -3.5]).reshape(1, 3, 1)
        torch.testing.assert_close(output, F.silu(raw))
        torch.testing.assert_close(
            final_state,
            torch.tensor([[[1.0, 2.0, -1.0]]]),
        )

    def test_prefill_matches_tokenwise_continuation(self):
        torch.manual_seed(51)
        projected = torch.randn(2, 5, 4)
        weight = torch.randn(4, 3)
        initial_state = torch.randn(2, 4, 3)
        original_state = initial_state.clone()

        prefill_output, prefill_state = causal_depthwise_conv1d_reference(
            projected,
            weight,
            initial_state,
        )

        step_state = initial_state
        step_outputs = []
        for token_idx in range(projected.shape[1]):
            step_output, step_state = causal_depthwise_conv1d_reference(
                projected[:, token_idx:token_idx + 1],
                weight,
                step_state,
            )
            step_outputs.append(step_output)

        torch.testing.assert_close(
            prefill_output,
            torch.cat(step_outputs, dim=1),
        )
        torch.testing.assert_close(prefill_state, step_state)
        torch.testing.assert_close(initial_state, original_state)


class RecurrentGatedDeltaRuleTest(unittest.TestCase):
    def _make_inputs(self, sequence_length=4):
        torch.manual_seed(52)
        query = torch.randn(2, sequence_length, 3, 4)
        key = torch.randn_like(query)
        value = torch.randn(2, sequence_length, 3, 2)
        log_decay = -F.softplus(torch.randn(2, sequence_length, 3))
        beta = torch.sigmoid(torch.randn(2, sequence_length, 3))
        state = torch.randn(2, 3, 4, 2)
        return query, key, value, log_decay, beta, state

    def test_single_token_matches_manual_update(self):
        inputs = self._make_inputs(sequence_length=1)
        query, key, value, log_decay, beta, initial_state = inputs

        output, final_state = recurrent_gated_delta_rule_reference(*inputs)
        expected_output, expected_state = _manual_delta_step(
            query[:, 0],
            key[:, 0],
            value[:, 0],
            log_decay[:, 0],
            beta[:, 0],
            initial_state,
        )

        torch.testing.assert_close(output[:, 0], expected_output)
        torch.testing.assert_close(final_state, expected_state)

    def test_prefill_matches_tokenwise_continuation(self):
        inputs = self._make_inputs(sequence_length=5)
        query, key, value, log_decay, beta, initial_state = inputs
        original_state = initial_state.clone()

        prefill_output, prefill_state = recurrent_gated_delta_rule_reference(
            *inputs
        )

        step_state = initial_state
        step_outputs = []
        for token_idx in range(query.shape[1]):
            step_output, step_state = recurrent_gated_delta_rule_reference(
                query[:, token_idx:token_idx + 1],
                key[:, token_idx:token_idx + 1],
                value[:, token_idx:token_idx + 1],
                log_decay[:, token_idx:token_idx + 1],
                beta[:, token_idx:token_idx + 1],
                step_state,
            )
            step_outputs.append(step_output)

        torch.testing.assert_close(
            prefill_output,
            torch.cat(step_outputs, dim=1),
        )
        torch.testing.assert_close(prefill_state, step_state)
        torch.testing.assert_close(initial_state, original_state)

    def test_future_inputs_do_not_change_past_outputs(self):
        inputs = list(self._make_inputs(sequence_length=4))
        baseline_output, _ = recurrent_gated_delta_rule_reference(*inputs)

        for tensor_idx in range(5):
            changed = [tensor.clone() for tensor in inputs]
            changed[tensor_idx][:, -1].add_(100.0)
            actual_output, _ = recurrent_gated_delta_rule_reference(*changed)
            torch.testing.assert_close(
                actual_output[:, :-1],
                baseline_output[:, :-1],
            )

    def test_zero_beta_only_decays_existing_state(self):
        query, key, value, log_decay, _, initial_state = (
            self._make_inputs(sequence_length=1)
        )
        beta = torch.zeros(log_decay.shape)

        output, final_state = recurrent_gated_delta_rule_reference(
            query,
            key,
            value,
            log_decay,
            beta,
            initial_state,
        )

        expected_state = (
            initial_state
            * log_decay[:, 0].exp()[..., None, None]
        )
        normalized_query = F.normalize(
            query[:, 0], p=2, dim=-1, eps=1e-6
        ) * (query.shape[-1] ** -0.5)
        expected_output = torch.einsum(
            "bhd,bhdv->bhv",
            normalized_query,
            expected_state,
        )
        torch.testing.assert_close(final_state, expected_state)
        torch.testing.assert_close(output[:, 0], expected_output)

    def test_rejects_incompatible_parameter_and_state_shapes(self):
        query, key, value, log_decay, beta, initial_state = self._make_inputs()

        with self.assertRaisesRegex(ValueError, "query and key"):
            recurrent_gated_delta_rule_reference(
                query,
                key[..., :-1],
                value,
                log_decay,
                beta,
                initial_state,
            )
        with self.assertRaisesRegex(ValueError, "log_decay"):
            recurrent_gated_delta_rule_reference(
                query,
                key,
                value,
                log_decay[..., :-1],
                beta,
                initial_state,
            )
        with self.assertRaisesRegex(ValueError, "initial_state"):
            recurrent_gated_delta_rule_reference(
                query,
                key,
                value,
                log_decay,
                beta,
                initial_state[..., :-1],
            )


class RMSNormGatedTest(unittest.TestCase):
    def test_matches_fp32_norm_then_silu_gate(self):
        torch.manual_seed(53)
        module = RMSNormGated(5, eps=1e-5)
        module.weight.data.copy_(torch.linspace(0.5, 1.5, 5))
        hidden_states = torch.randn(2, 3, 4, 5)
        gate = torch.randn_like(hidden_states)

        actual = module(hidden_states, gate)
        hidden_fp32 = hidden_states.float()
        expected = hidden_fp32 * torch.rsqrt(
            hidden_fp32.square().mean(dim=-1, keepdim=True) + 1e-5
        )
        expected = expected * module.weight.float() * F.silu(gate.float())

        torch.testing.assert_close(actual, expected.to(hidden_states.dtype))

    def test_rejects_gate_or_hidden_dimension_mismatch(self):
        module = RMSNormGated(5)
        hidden_states = torch.randn(2, 3, 4, 5)

        with self.assertRaisesRegex(ValueError, "same shape"):
            module(hidden_states, torch.randn(2, 3, 4, 4))
        with self.assertRaisesRegex(ValueError, "last dimension"):
            narrowed = hidden_states[..., :-1]
            module(narrowed, torch.randn_like(narrowed))


class GatedDeltaNetReferenceIntegrationTest(unittest.TestCase):
    def test_forward_builds_qwen_gates_and_repeats_key_heads(self):
        torch.manual_seed(54)
        layer = Qwen3_5GatedDeltaNetReference(_tiny_config())
        layer.conv1d.weight.data.fill_(1.0)
        hidden_states = torch.randn(2, 3, 8)
        captured = {}

        def fake_recurrence(
            query,
            key,
            value,
            log_decay,
            beta,
            initial_state=None,
            **kwargs,
        ):
            captured.update(
                query=query.detach(),
                key=key.detach(),
                value=value.detach(),
                log_decay=log_decay.detach(),
                beta=beta.detach(),
            )
            final_state = torch.zeros(
                query.shape[0],
                query.shape[2],
                query.shape[3],
                value.shape[3],
            )
            return torch.zeros_like(value), final_state

        with patch.object(
            gated_delta_net,
            "recurrent_gated_delta_rule_reference",
            side_effect=fake_recurrence,
        ):
            output, state = layer(hidden_states)

        projected_qkv = layer.in_proj_qkv(hidden_states)
        convolved_qkv, _ = causal_depthwise_conv1d_reference(
            projected_qkv,
            layer.conv1d.weight.squeeze(1),
        )
        expected_q, expected_k, expected_v = torch.split(
            convolved_qkv,
            [layer.key_dim, layer.key_dim, layer.value_dim],
            dim=-1,
        )
        expected_q = expected_q.reshape(2, 3, 2, 3)
        expected_k = expected_k.reshape(2, 3, 2, 3)
        expected_q = expected_q.repeat_interleave(2, dim=2)
        expected_k = expected_k.repeat_interleave(2, dim=2)

        torch.testing.assert_close(captured["query"], expected_q)
        torch.testing.assert_close(captured["key"], expected_k)
        torch.testing.assert_close(
            captured["value"], expected_v.reshape(2, 3, 4, 2)
        )
        expected_beta = torch.sigmoid(layer.in_proj_b(hidden_states))
        expected_log_decay = -layer.A_log.float().exp() * F.softplus(
            layer.in_proj_a(hidden_states).float() + layer.dt_bias
        )
        torch.testing.assert_close(captured["beta"], expected_beta)
        torch.testing.assert_close(
            captured["log_decay"], expected_log_decay
        )
        self.assertEqual(output.shape, hidden_states.shape)
        self.assertIsInstance(state, GatedDeltaNetState)

    def test_full_prefill_matches_tokenwise_module_calls(self):
        torch.manual_seed(55)
        layer = Qwen3_5GatedDeltaNetReference(_tiny_config())
        hidden_states = torch.randn(2, 4, 8)

        prefill_output, prefill_state = layer(hidden_states)

        step_state = None
        step_outputs = []
        for token_idx in range(hidden_states.shape[1]):
            step_output, step_state = layer(
                hidden_states[:, token_idx:token_idx + 1],
                step_state,
            )
            step_outputs.append(step_output)

        torch.testing.assert_close(
            prefill_output,
            torch.cat(step_outputs, dim=1),
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            prefill_state.conv_state,
            step_state.conv_state,
        )
        torch.testing.assert_close(
            prefill_state.recurrent_state,
            step_state.recurrent_state,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_forward_rejects_invalid_hidden_shape_and_state_type(self):
        layer = Qwen3_5GatedDeltaNetReference(_tiny_config())

        with self.assertRaisesRegex(ValueError, "hidden_states"):
            layer(torch.randn(2, 3, 7))
        with self.assertRaisesRegex(TypeError, "GatedDeltaNetState"):
            layer(torch.randn(2, 3, 8), state=object())


if __name__ == "__main__":
    unittest.main()
