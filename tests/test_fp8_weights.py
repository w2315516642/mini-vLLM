import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from minivllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    _load_merged_column_shard,
    _load_qkv_gate_scale,
)
from minivllm.model_executor.parallel_utils import parallel_state
from minivllm.model_executor.parallel_utils.tensor_parallel.layers import (
    ColumnParallelLinear,
    dequantize_fp8_block_weight,
    get_fp8_block_size,
)
from minivllm.model_executor.weight_utils import hf_model_weights_iterator


FP8_CONFIG = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [2, 2],
}


class FP8BlockWeightTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = 1
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK = 0

    @classmethod
    def tearDownClass(cls):
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK = None

    def test_config_and_tail_blocks_define_scale_shape(self):
        self.assertEqual(get_fp8_block_size(FP8_CONFIG), (2, 2))
        weight = torch.ones(3, 5, dtype=torch.float8_e4m3fn)
        scale = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3)

        actual = dequantize_fp8_block_weight(
            weight, scale, (2, 2), torch.float32
        )

        expected_scale = torch.tensor(
            [
                [1, 1, 2, 2, 3],
                [1, 1, 2, 2, 3],
                [4, 4, 5, 5, 6],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(actual, expected_scale)

    def test_column_linear_keeps_fp8_resident_and_computes_in_input_dtype(self):
        layer = ColumnParallelLinear(
            4,
            4,
            bias=False,
            gather_output=False,
            use_cpu_initialization=True,
            quant_config=FP8_CONFIG,
        )
        base_weight = torch.tensor(
            [
                [1.0, 0.5, -1.0, 2.0],
                [0.0, 1.0, 0.5, -0.5],
                [2.0, -1.0, 1.0, 0.0],
                [0.5, 0.5, -2.0, 1.0],
            ]
        )
        with torch.no_grad():
            layer.weight.copy_(base_weight.to(torch.float8_e4m3fn))
            layer.weight_scale_inv.copy_(
                torch.tensor([[0.5, 1.5], [2.0, 0.25]])
            )
        inputs = torch.tensor([[1.0, 2.0, -1.0, 0.5]])

        actual, _ = layer(inputs)
        dequantized = dequantize_fp8_block_weight(
            layer.weight,
            layer.weight_scale_inv,
            (2, 2),
            inputs.dtype,
        )
        expected = F.linear(inputs, dequantized)

        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(actual.dtype, inputs.dtype)
        torch.testing.assert_close(actual, expected)

    def test_packed_qkv_and_gate_up_scales_use_local_segments(self):
        qkv_scale = torch.zeros(4, 3)
        _load_qkv_gate_scale(
            qkv_scale,
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "q",
            tensor_model_parallel_rank=0,
            q_size=2,
            kv_size=2,
            block_n=2,
        )
        _load_qkv_gate_scale(
            qkv_scale,
            torch.tensor([[7.0, 8.0, 9.0]]),
            "k",
            tensor_model_parallel_rank=0,
            q_size=2,
            kv_size=2,
            block_n=2,
        )
        _load_qkv_gate_scale(
            qkv_scale,
            torch.tensor([[10.0, 11.0, 12.0]]),
            "v",
            tensor_model_parallel_rank=0,
            q_size=2,
            kv_size=2,
            block_n=2,
        )
        torch.testing.assert_close(
            qkv_scale,
            torch.tensor(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0],
                ]
            ),
        )

        gate_up_scale = torch.zeros(4, 2)
        _load_merged_column_shard(
            gate_up_scale,
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            shard_idx=0,
            tensor_model_parallel_rank=0,
        )
        _load_merged_column_shard(
            gate_up_scale,
            torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
            shard_idx=1,
            tensor_model_parallel_rank=0,
        )
        torch.testing.assert_close(
            gate_up_scale,
            torch.tensor(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [5.0, 6.0],
                    [7.0, 8.0],
                ]
            ),
        )


class SafetensorsIteratorTest(unittest.TestCase):
    def test_shards_are_streamed_in_stable_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            save_file(
                {"second.weight": torch.tensor([2.0])},
                root / "model-00002-of-00002.safetensors",
            )
            save_file(
                {"first.weight": torch.tensor([1.0])},
                root / "model-00001-of-00002.safetensors",
            )

            values = list(hf_model_weights_iterator(tmpdir))

        self.assertEqual([name for name, _ in values], [
            "first.weight",
            "second.weight",
        ])
        self.assertEqual([value.item() for _, value in values], [1.0, 2.0])


class _QwenLoaderHarness:
    load_weights = Qwen3_5ForConditionalGeneration.load_weights

    def __init__(self, state_dict):
        self.config = type(
            "Config",
            (),
            {
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 2,
                "quantization_config": FP8_CONFIG,
            },
        )()
        self._state_dict = state_dict

    def state_dict(self):
        return self._state_dict


class QwenSafetensorsLoaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = 1
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK = 0

    @classmethod
    def tearDownClass(cls):
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK = None

    def test_official_prefix_and_packed_fp8_weights_are_mapped(self):
        q_weight = torch.arange(1, 33, dtype=torch.float32).reshape(8, 4)
        k_weight = torch.arange(41, 49, dtype=torch.float32).reshape(2, 4)
        v_weight = torch.arange(51, 59, dtype=torch.float32).reshape(2, 4)
        gate_weight = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)
        up_weight = torch.arange(21, 37, dtype=torch.float32).reshape(4, 4)
        checkpoint = {
            "model.language_model.layers.0.self_attn.q_proj.weight": (
                q_weight.to(torch.float8_e4m3fn)
            ),
            "model.language_model.layers.0.self_attn.k_proj.weight": (
                k_weight.to(torch.float8_e4m3fn)
            ),
            "model.language_model.layers.0.self_attn.v_proj.weight": (
                v_weight.to(torch.float8_e4m3fn)
            ),
            "model.language_model.layers.0.self_attn.q_proj.weight_scale_inv": (
                torch.arange(1, 9, dtype=torch.float32).reshape(4, 2)
            ),
            "model.language_model.layers.0.self_attn.k_proj.weight_scale_inv": (
                torch.tensor([[9.0, 10.0]])
            ),
            "model.language_model.layers.0.self_attn.v_proj.weight_scale_inv": (
                torch.tensor([[11.0, 12.0]])
            ),
            "model.language_model.layers.0.mlp.gate_proj.weight": (
                gate_weight.to(torch.float8_e4m3fn)
            ),
            "model.language_model.layers.0.mlp.up_proj.weight": (
                up_weight.to(torch.float8_e4m3fn)
            ),
            "model.language_model.layers.0.mlp.gate_proj.weight_scale_inv": (
                torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            ),
            "model.language_model.layers.0.mlp.up_proj.weight_scale_inv": (
                torch.tensor([[5.0, 6.0], [7.0, 8.0]])
            ),
        }
        targets = {
            "model.layers.0.self_attn.qkv_gate_proj.weight": torch.zeros(
                12, 4, dtype=torch.float8_e4m3fn
            ),
            "model.layers.0.self_attn.qkv_gate_proj.weight_scale_inv": (
                torch.zeros(6, 2)
            ),
            "model.layers.0.mlp.gate_up_proj.weight": torch.zeros(
                8, 4, dtype=torch.float8_e4m3fn
            ),
            "model.layers.0.mlp.gate_up_proj.weight_scale_inv": torch.zeros(
                4, 2
            ),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            save_file(checkpoint, Path(tmpdir) / "model.safetensors")
            _QwenLoaderHarness(targets).load_weights(tmpdir)

        torch.testing.assert_close(
            targets["model.layers.0.self_attn.qkv_gate_proj.weight"],
            torch.cat(
                (
                    q_weight.to(torch.float8_e4m3fn),
                    k_weight.to(torch.float8_e4m3fn),
                    v_weight.to(torch.float8_e4m3fn),
                )
            ),
        )
        torch.testing.assert_close(
            targets[
                "model.layers.0.self_attn.qkv_gate_proj.weight_scale_inv"
            ],
            torch.arange(1, 13, dtype=torch.float32).reshape(6, 2),
        )
        torch.testing.assert_close(
            targets["model.layers.0.mlp.gate_up_proj.weight"],
            torch.cat(
                (
                    gate_weight.to(torch.float8_e4m3fn),
                    up_weight.to(torch.float8_e4m3fn),
                )
            ),
        )


if __name__ == "__main__":
    unittest.main()
