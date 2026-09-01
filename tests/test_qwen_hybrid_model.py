import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from minivllm.model_executor.layers.gated_delta_net import (
    causal_depthwise_conv1d_reference,
)
from minivllm.model_executor.models import qwen3_5 as qwen_module
from minivllm.model_executor.models.qwen3_5 import (
    Qwen3_5GatedDeltaNet,
    _causal_conv1d_prefill,
    _load_gdn_qkv_shard,
)
from minivllm.worker.hybrid_cache import GatedDeltaNetStateSpec


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


class RecordingParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.kwargs = kwargs


class QwenGatedDeltaNetTPTest(unittest.TestCase):
    @staticmethod
    def _config():
        return SimpleNamespace(
            hidden_size=5120,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            linear_num_key_heads=16,
            linear_num_value_heads=48,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
        )

    def test_27b_dimensions_are_partitioned_by_head(self):
        with patch.object(
            qwen_module,
            "get_tensor_model_parallel_world_size",
            return_value=2,
        ), patch.object(
            qwen_module,
            "ColumnParallelLinear",
            RecordingParallelLinear,
        ), patch.object(
            qwen_module,
            "RowParallelLinear",
            RecordingParallelLinear,
        ):
            layer = Qwen3_5GatedDeltaNet(self._config(), layer_idx=0)

        self.assertEqual(layer.num_k_heads, 8)
        self.assertEqual(layer.num_v_heads, 24)
        self.assertEqual(layer.key_dim, 1024)
        self.assertEqual(layer.value_dim, 3072)
        self.assertEqual(layer.conv_dim, 5120)
        self.assertEqual(layer.total_conv_dim, 10240)
        self.assertEqual(layer.in_proj_qkv.output_size, 10240)
        self.assertEqual(layer.in_proj_z.output_size, 6144)
        self.assertEqual(layer.out_proj.input_size, 6144)
        self.assertEqual(layer.conv1d.weight.shape, (5120, 1, 4))
        self.assertEqual(layer.dt_bias.shape, (24,))
        self.assertEqual(layer.A_log.shape, (24,))

        spec = GatedDeltaNetStateSpec.from_text_config(
            self._config(), tensor_parallel_size=2
        )
        self.assertEqual(spec.conv_state_shape, (5120, 4))
        self.assertEqual(spec.recurrent_state_shape, (24, 128, 128))

    def test_combined_qkv_checkpoint_is_sharded_per_segment(self):
        checkpoint = torch.arange(14, dtype=torch.float32).reshape(14, 1)
        local = torch.zeros(7, 1)

        _load_gdn_qkv_shard(
            local,
            checkpoint,
            tensor_model_parallel_rank=1,
            tensor_model_parallel_world_size=2,
            global_key_dim=4,
            global_value_dim=6,
        )

        expected = checkpoint[[2, 3, 6, 7, 11, 12, 13]]
        torch.testing.assert_close(local, expected)

    def test_combined_qkv_fp8_scales_follow_the_same_partition(self):
        checkpoint_scale = torch.arange(8, dtype=torch.float32).reshape(8, 1)
        local_scale = torch.zeros(4, 1)

        _load_gdn_qkv_shard(
            local_scale,
            checkpoint_scale,
            tensor_model_parallel_rank=1,
            tensor_model_parallel_world_size=2,
            global_key_dim=4,
            global_value_dim=8,
            block_n=2,
        )

        expected = checkpoint_scale[[1, 3, 6, 7]]
        torch.testing.assert_close(local_scale, expected)


if __name__ == "__main__":
    unittest.main()
