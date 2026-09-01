import unittest

import torch

from minivllm.model_executor.layers.gated_delta_net import (
    causal_depthwise_conv1d_reference,
)
from minivllm.model_executor.models.qwen3_5 import _causal_conv1d_prefill


class QwenCausalConvPrefillTest(unittest.TestCase):
    def test_prefill_matches_reference_and_updates_state(self):
        torch.manual_seed(101)
        projected = torch.randn(2, 7, 5, dtype=torch.float16)
        weight = torch.randn(5, 4, dtype=torch.float16)
        initial_state = torch.randn(2, 5, 4, dtype=torch.float32)

        expected, expected_state = causal_depthwise_conv1d_reference(
            projected,
            weight,
            initial_state,
        )
        actual_state = initial_state.clone()
        actual = _causal_conv1d_prefill(projected, actual_state, weight)

        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
        torch.testing.assert_close(actual_state, expected_state)

    def test_split_continuation_matches_one_prefill(self):
        torch.manual_seed(102)
        projected = torch.randn(1, 9, 6, dtype=torch.float32)
        weight = torch.randn(6, 3, dtype=torch.float32)

        full_state = torch.zeros(1, 6, 3, dtype=torch.float32)
        full = _causal_conv1d_prefill(projected, full_state, weight)

        split_state = torch.zeros_like(full_state)
        first = _causal_conv1d_prefill(
            projected[:, :4], split_state, weight
        )
        second = _causal_conv1d_prefill(
            projected[:, 4:], split_state, weight
        )

        torch.testing.assert_close(torch.cat((first, second), dim=1), full)
        torch.testing.assert_close(split_state, full_state)


if __name__ == "__main__":
    unittest.main()
