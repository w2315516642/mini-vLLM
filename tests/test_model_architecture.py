import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from minivllm.configs.config import ModelConfig, ParallelConfig
from minivllm.configs.model_architecture import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    ModelArchitecture,
)


def llama_config(**overrides):
    values = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "dtype": torch.float16,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def qwen_config(**text_overrides):
    layer_types = (
        [LINEAR_ATTENTION, LINEAR_ATTENTION, LINEAR_ATTENTION, FULL_ATTENTION]
        * 2
    )
    text_values = {
        "hidden_size": 5120,
        "num_hidden_layers": 8,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "layer_types": layer_types,
        "dtype": torch.bfloat16,
    }
    text_values.update(text_overrides)
    return SimpleNamespace(
        architectures=["Qwen3_5ForConditionalGeneration"],
        model_type="qwen3_5",
        text_config=SimpleNamespace(**text_values),
    )


class ModelArchitectureTest(unittest.TestCase):

    def test_flat_config_uses_root_as_text_config(self):
        config = llama_config()
        architecture = ModelArchitecture.from_hf_config(config)

        self.assertIs(architecture.root_config, config)
        self.assertIs(architecture.text_config, config)
        self.assertEqual(architecture.architectures, ("LlamaForCausalLM",))

    def test_nested_config_uses_text_config_dimensions(self):
        config = qwen_config()
        architecture = ModelArchitecture.from_hf_config(config)

        self.assertIs(architecture.root_config, config)
        self.assertIs(architecture.text_config, config.text_config)
        self.assertEqual(architecture.hidden_size, 5120)
        self.assertEqual(architecture.num_hidden_layers, 8)
        self.assertEqual(architecture.num_attention_heads, 24)
        self.assertEqual(architecture.num_key_value_heads, 4)

    def test_explicit_head_dim_takes_precedence(self):
        architecture = ModelArchitecture.from_hf_config(qwen_config())
        self.assertEqual(architecture.head_size, 256)

    def test_head_size_falls_back_to_hidden_size_division(self):
        config = llama_config()
        del config.num_key_value_heads
        architecture = ModelArchitecture.from_hf_config(config)

        self.assertEqual(architecture.head_size, 128)
        self.assertEqual(architecture.num_key_value_heads, 32)

    def test_flat_model_defaults_to_full_attention_layers(self):
        architecture = ModelArchitecture.from_hf_config(llama_config())

        self.assertEqual(
            architecture.layer_types,
            (FULL_ATTENTION,) * architecture.num_hidden_layers,
        )
        self.assertEqual(architecture.num_full_attention_layers, 32)
        self.assertEqual(architecture.num_linear_attention_layers, 0)
        self.assertFalse(architecture.is_hybrid)

    def test_hybrid_layer_counts_are_reported(self):
        architecture = ModelArchitecture.from_hf_config(qwen_config())

        self.assertEqual(architecture.num_full_attention_layers, 2)
        self.assertEqual(architecture.num_linear_attention_layers, 6)
        self.assertTrue(architecture.is_hybrid)

    def test_parallel_partition_helpers(self):
        architecture = ModelArchitecture.from_hf_config(qwen_config())

        self.assertEqual(architecture.get_num_attention_heads(2), 12)
        self.assertEqual(architecture.get_num_kv_heads(2), 2)
        self.assertEqual(architecture.get_num_layers(2), 4)

    def test_parallelism_rejects_non_positive_sizes(self):
        architecture = ModelArchitecture.from_hf_config(qwen_config())

        with self.assertRaisesRegex(ValueError, "tensor_parallel_size"):
            architecture.verify_parallelism(0, 1)
        with self.assertRaisesRegex(ValueError, "pipeline_parallel_size"):
            architecture.verify_parallelism(1, 0)

    def test_parallelism_rejects_indivisible_query_heads(self):
        architecture = ModelArchitecture.from_hf_config(qwen_config())
        with self.assertRaisesRegex(ValueError, "attention heads"):
            architecture.verify_parallelism(5, 1)

    def test_parallelism_rejects_indivisible_kv_heads(self):
        architecture = ModelArchitecture.from_hf_config(qwen_config())
        with self.assertRaisesRegex(ValueError, "KV heads"):
            architecture.verify_parallelism(3, 1)

    def test_parallelism_rejects_indivisible_layers(self):
        architecture = ModelArchitecture.from_hf_config(qwen_config())
        with self.assertRaisesRegex(ValueError, "hidden layers"):
            architecture.verify_parallelism(1, 3)

    def test_query_heads_must_be_grouped_by_kv_heads(self):
        with self.assertRaisesRegex(ValueError, "query heads"):
            ModelArchitecture.from_hf_config(
                qwen_config(num_attention_heads=10, num_key_value_heads=4)
            )

    def test_layer_type_count_must_match_layer_count(self):
        with self.assertRaisesRegex(ValueError, "layer_types"):
            ModelArchitecture.from_hf_config(qwen_config(layer_types=[]))

    def test_unknown_layer_type_is_rejected(self):
        layer_types = [LINEAR_ATTENTION] * 7 + ["sliding_attention"]
        with self.assertRaisesRegex(ValueError, "sliding_attention"):
            ModelArchitecture.from_hf_config(
                qwen_config(layer_types=layer_types)
            )

    def test_missing_required_field_has_clear_error(self):
        config = llama_config()
        del config.hidden_size
        with self.assertRaisesRegex(ValueError, "hidden_size"):
            ModelArchitecture.from_hf_config(config)

    def test_inferred_head_size_requires_exact_division(self):
        with self.assertRaisesRegex(ValueError, "hidden_size"):
            ModelArchitecture.from_hf_config(
                llama_config(hidden_size=4097)
            )


class ModelConfigIntegrationTest(unittest.TestCase):

    @patch("minivllm.configs.config.AutoConfig.from_pretrained")
    def test_model_config_exposes_normalized_architecture(self, from_pretrained):
        from_pretrained.return_value = qwen_config()
        with patch("torch.cuda.get_device_capability", return_value=(8, 9)):
            model_config = ModelConfig(
                "Qwen/Qwen3.8-27B",
                download_dir=None,
                use_np_weights=False,
                use_dummy_weights=False,
                dtype="auto",
                seed=42,
            )

        self.assertIsInstance(model_config.architecture, ModelArchitecture)
        self.assertEqual(model_config.dtype, torch.bfloat16)
        self.assertEqual(model_config.get_hidden_size(), 5120)
        self.assertEqual(model_config.get_head_size(), 256)

    @patch("minivllm.configs.config.AutoConfig.from_pretrained")
    def test_model_config_delegates_parallel_values(self, from_pretrained):
        from_pretrained.return_value = qwen_config()
        with patch("torch.cuda.get_device_capability", return_value=(8, 9)):
            model_config = ModelConfig(
                "Qwen/Qwen3.8-27B",
                download_dir=None,
                use_np_weights=False,
                use_dummy_weights=False,
                dtype="auto",
                seed=42,
            )
        parallel_config = ParallelConfig(
            pipeline_parallel_size=1,
            tensor_parallel_size=2,
            worker_use_ray=False,
        )

        model_config.verify_with_parallel_config(parallel_config)
        self.assertEqual(model_config.get_num_heads(parallel_config), 12)
        self.assertEqual(model_config.get_num_kv_heads(parallel_config), 2)
        self.assertEqual(model_config.get_num_layers(parallel_config), 8)


if __name__ == "__main__":
    unittest.main()
