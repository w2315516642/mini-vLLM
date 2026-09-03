import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from minivllm.multimodal import MultiModalInputs
from minivllm.entrypoints.llm import LLM
from minivllm.sampling_params import SamplingParams
from minivllm.model_executor.models.qwen3_5 import (
    _apply_interleaved_mrope,
)
from minivllm.model_executor.models.qwen3_5_vision import (
    Qwen3_5VisionModel,
)

llm_module = importlib.import_module("minivllm.entrypoints.llm")


class MultiModalInputsTest(unittest.TestCase):
    def test_processor_output_builds_image_and_video_positions(self):
        token_types = [0, 1, 1, 1, 1, 0, 2, 0, 2, 0]
        output = {
            "input_ids": torch.arange(10).unsqueeze(0),
            "mm_token_type_ids": torch.tensor([token_types]),
            "pixel_values": torch.randn(16, 8),
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
            "pixel_values_videos": torch.randn(8, 8),
            "video_grid_thw": torch.tensor([[2, 2, 2]]),
        }

        prompt_token_ids, inputs = MultiModalInputs.from_processor_output(
            output
        )
        inputs = inputs.with_positions(prompt_token_ids, spatial_merge_size=2)

        self.assertEqual(prompt_token_ids, tuple(range(10)))
        self.assertEqual(
            inputs.position_ids,
            (
                (0, 1, 1, 1, 1, 3, 4, 5, 6, 7),
                (0, 1, 1, 2, 2, 3, 4, 5, 6, 7),
                (0, 1, 2, 1, 2, 3, 4, 5, 6, 7),
            ),
        )
        self.assertEqual(inputs.rope_delta, -2)
        lightweight = inputs.positions_only()
        self.assertIsNone(lightweight.pixel_values)
        self.assertIsNone(lightweight.pixel_values_videos)
        self.assertEqual(lightweight.position_ids, inputs.position_ids)
        self.assertEqual(lightweight.rope_delta, inputs.rope_delta)

    def test_visual_grid_and_tensor_must_be_paired(self):
        with self.assertRaisesRegex(ValueError, "provided together"):
            MultiModalInputs.from_processor_output({
                "input_ids": [[1, 2]],
                "mm_token_type_ids": [[0, 1]],
                "image_grid_thw": [[1, 2, 2]],
            })


class MultiModalLLMApiTest(unittest.TestCase):
    def test_chat_uses_processor_and_forwards_thinking_mode(self):
        llm = object.__new__(LLM)
        processor = Mock()
        processor.apply_chat_template.return_value = {"processed": True}
        llm.get_processor = Mock(return_value=processor)
        llm.generate = Mock(return_value=[])
        messages = [{"role": "user", "content": "hello"}]

        output = llm.chat(
            messages, enable_thinking=False, use_tqdm=False
        )

        self.assertEqual(output, [])
        processor.apply_chat_template.assert_called_once_with(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        llm.generate.assert_called_once_with(
            sampling_params=None,
            multi_modal_inputs=[{"processed": True}],
            use_tqdm=False,
        )

    def test_generate_accepts_one_processor_output_per_request(self):
        llm = object.__new__(LLM)
        llm._generation_active = False
        llm.llm_engine = Mock()
        recorded = []
        llm._add_request = lambda *args: recorded.append(args)
        llm._run_engine = lambda use_tqdm: (item for item in [])
        processor_output = {
            "input_ids": [[3, 120, 4]],
            "mm_token_type_ids": [[0, 1, 0]],
            "pixel_values": torch.randn(4, 8),
            "image_grid_thw": [[1, 2, 2]],
        }

        default_params = SamplingParams()
        with patch.object(llm_module, "SamplingParams") as params:
            params.return_value = default_params
            llm.generate(
                multi_modal_inputs=[processor_output], use_tqdm=False
            )

        self.assertEqual(len(recorded), 1)
        prompt, sampling_params, token_ids, multimodal = recorded[0]
        self.assertIsNone(prompt)
        self.assertIs(sampling_params, params.return_value)
        self.assertEqual(token_ids, [3, 120, 4])
        self.assertIsInstance(multimodal, MultiModalInputs)

    def test_generate_keeps_use_tqdm_as_fourth_positional_argument(self):
        llm = object.__new__(LLM)
        llm._generation_active = False
        llm.llm_engine = Mock()
        llm._add_request = Mock()
        llm._run_engine = Mock(return_value=(item for item in []))
        sampling_params = SamplingParams()

        llm.generate(None, sampling_params, [[3, 4]], False)

        llm._run_engine.assert_called_once_with(False)
        llm._add_request.assert_called_once_with(
            None, sampling_params, [3, 4], None
        )


class QwenVisionModelTest(unittest.TestCase):
    @staticmethod
    def _config():
        return SimpleNamespace(
            hidden_size=16,
            intermediate_size=32,
            num_heads=4,
            in_channels=3,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=8,
            num_position_embeddings=16,
            depth=2,
        )

    def test_same_tower_encodes_image_and_video_patch_grids(self):
        torch.manual_seed(211)
        model = Qwen3_5VisionModel(self._config()).eval()
        patch_width = 3 * 2 * 2 * 2
        with torch.inference_mode():
            image = model(
                torch.randn(4, patch_width),
                torch.tensor([[1, 2, 2]]),
            )
            video = model(
                torch.randn(8, patch_width),
                torch.tensor([[2, 2, 2]]),
            )

        self.assertEqual(image.shape, (1, 8))
        self.assertEqual(video.shape, (2, 8))
        self.assertTrue(torch.isfinite(image).all())
        self.assertTrue(torch.isfinite(video).all())


class QwenMRotaryTest(unittest.TestCase):
    def test_equal_3d_positions_reduce_to_one_dimensional_neox_rope(self):
        torch.manual_seed(223)
        head_dim = 8
        rotary_dim = 6
        query = torch.randn(3, 2 * head_dim)
        key = torch.randn(3, head_dim)
        expected_query = query.clone().view(3, 2, head_dim)
        expected_key = key.clone().view(3, 1, head_dim)
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, rotary_dim, 2) / rotary_dim)
        )
        positions = torch.tensor([1, 3, 5])
        frequencies = torch.outer(positions.float(), inv_freq)
        cache = torch.cat((
            torch.outer(torch.arange(8).float(), inv_freq).cos(),
            torch.outer(torch.arange(8).float(), inv_freq).sin(),
        ), dim=-1)
        cos = frequencies.cos().unsqueeze(1)
        sin = frequencies.sin().unsqueeze(1)

        for tensor in (expected_query, expected_key):
            first = tensor[..., : rotary_dim // 2].clone()
            second = tensor[..., rotary_dim // 2:rotary_dim].clone()
            tensor[..., : rotary_dim // 2] = first * cos - second * sin
            tensor[..., rotary_dim // 2:rotary_dim] = second * cos + first * sin

        _apply_interleaved_mrope(
            query,
            key,
            positions.repeat(3, 1),
            head_dim,
            cache,
            (1, 1, 1),
        )

        torch.testing.assert_close(query, expected_query.flatten(1))
        torch.testing.assert_close(key, expected_key.flatten(1))


if __name__ == "__main__":
    unittest.main()
