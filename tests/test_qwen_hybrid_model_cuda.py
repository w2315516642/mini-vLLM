import os
import unittest
from types import SimpleNamespace

import torch

from minivllm.configs.model_architecture import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
)
from minivllm.multimodal import MultiModalInputs
from minivllm.model_executor.input_metadata import InputMetadata
from minivllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
)
from minivllm.model_executor.parallel_utils import parallel_state
from minivllm.sampling_params import SamplingParams
from minivllm.sequence import SequenceData
from minivllm.sequence import SequenceGroupMetadata
from minivllm.model_executor.weight_utils import initialize_dummy_weights
from minivllm.worker.hybrid_cache import (
    GatedDeltaNetStateSpec,
    HybridCache,
)
from minivllm.worker.worker import Worker


RUN_CUDA_TESTS = (
    os.environ.get("MINIVLLM_RUN_CUDA_QWEN_HYBRID_TESTS") == "1"
    and torch.cuda.is_available()
)


def _config():
    return SimpleNamespace(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        partial_rotary_factor=0.5,
        rms_norm_eps=1e-6,
        max_position_embeddings=256,
        rope_theta=10_000.0,
        hidden_act="silu",
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        layer_types=(LINEAR_ATTENTION, FULL_ATTENTION),
        mtp_num_hidden_layers=1,
        mtp_use_dedicated_embeddings=False,
        eos_token_id=127,
    )


def _multimodal_config():
    config = _config()
    config.rope_parameters = {
        "rope_theta": 10_000.0,
        "mrope_section": [6, 5, 5],
    }
    config.vision_config = SimpleNamespace(
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=128,
        num_position_embeddings=16,
        depth=2,
    )
    config.image_token_id = 120
    config.video_token_id = 121
    return config


def _kv_cache(dtype):
    block_size = 8
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key = torch.zeros(
        1, 1, 64 // x, block_size, x, device="cuda", dtype=dtype
    )
    value = torch.zeros(
        1, 1, 64, block_size, device="cuda", dtype=dtype
    )
    return key, value


def _metadata(
    query_len,
    start,
    *,
    is_prompt,
    state_cache,
    enable_mtp=False,
):
    seq_id = 10
    sampling_params = SamplingParams(temperature=0.0)
    seq_data = {seq_id: SequenceData(list(range(start + query_len)))}
    if is_prompt:
        prompt_lens = [query_len]
        context_lens = torch.empty(0, dtype=torch.int32, device="cuda")
        block_tables = torch.empty((0, 0), dtype=torch.int32, device="cuda")
        prompt_seq_ids = [seq_id]
        generation_seq_ids = []
        fresh_prompt_lens = [query_len] if start == 0 else []
        cached_query_lens = [] if start == 0 else [query_len]
        cached_cu = torch.tensor(
            [0, query_len], dtype=torch.int32, device="cuda"
        )
        cached_context = torch.tensor(
            [start + query_len], dtype=torch.int32, device="cuda"
        )
        cached_tables = torch.tensor(
            [[0]], dtype=torch.int32, device="cuda"
        )
    else:
        prompt_lens = []
        context_lens = torch.tensor(
            [start + query_len], dtype=torch.int32, device="cuda"
        )
        block_tables = torch.tensor(
            [[0]], dtype=torch.int32, device="cuda"
        )
        prompt_seq_ids = []
        generation_seq_ids = [seq_id]
        fresh_prompt_lens = []
        cached_query_lens = []
        cached_cu = torch.tensor([0], dtype=torch.int32, device="cuda")
        cached_context = torch.empty(0, dtype=torch.int32, device="cuda")
        cached_tables = torch.empty((0, 0), dtype=torch.int32, device="cuda")

    metadata = InputMetadata(
        seq_groups=[([seq_id], sampling_params)],
        seq_data=seq_data,
        prompt_lens=prompt_lens,
        slot_mapping=torch.arange(
            start, start + query_len, dtype=torch.int32, device="cuda"
        ),
        context_lens=context_lens,
        max_context_len=start + query_len if not is_prompt else 0,
        block_tables=block_tables,
        fresh_prompt_lens=fresh_prompt_lens,
        cached_prompt_query_lens=cached_query_lens,
        cached_prompt_cu_seqlens=cached_cu,
        cached_prompt_context_lens=cached_context,
        cached_prompt_block_tables=cached_tables,
        max_cached_prompt_context_len=(
            start + query_len if start and is_prompt else 0
        ),
        prompt_seq_ids=prompt_seq_ids,
        generation_seq_ids=generation_seq_ids,
        enable_mtp=enable_mtp,
    )
    metadata.state_slot_mapping = state_cache.acquire([seq_id])
    return metadata


def _hybrid_cache(config, kv_cache):
    return HybridCache(
        layer_types=config.layer_types,
        full_attention_caches={1: kv_cache},
        state_spec=GatedDeltaNetStateSpec.from_text_config(config),
        max_num_seqs=1,
        device="cuda",
    )


@unittest.skipUnless(
    RUN_CUDA_TESTS,
    "set MINIVLLM_RUN_CUDA_QWEN_HYBRID_TESTS=1 after rebuilding extensions",
)
class QwenHybridModelCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = 1
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK = 0

    @classmethod
    def tearDownClass(cls):
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK = None

    def test_prefill_then_decode_matches_single_prefill(self):
        torch.manual_seed(103)
        config = _config()
        split_model = Qwen3_5Model(config).cuda().half().eval()
        for parameter in split_model.parameters():
            if parameter.numel():
                torch.nn.init.normal_(parameter, mean=0.0, std=0.02)
        full_model = Qwen3_5Model(config).cuda().half().eval()
        full_model.load_state_dict(split_model.state_dict())

        split_cache = _hybrid_cache(config, _kv_cache(torch.float16))
        full_cache = _hybrid_cache(config, _kv_cache(torch.float16))
        tokens = torch.tensor([3, 5, 7, 9, 11], device="cuda")

        with torch.inference_mode():
            first_tokens = torch.cat(
                (tokens[:4], torch.zeros(4, dtype=torch.long, device="cuda"))
            )
            first_positions = torch.tensor(
                [0, 1, 2, 3, 0, 0, 0, 0], device="cuda"
            )
            split_model(
                first_tokens,
                first_positions,
                split_cache,
                _metadata(4, 0, is_prompt=True, state_cache=split_cache),
                None,
            )

            decode_tokens = torch.tensor(
                [11, 0, 0, 0, 0, 0, 0, 0], device="cuda"
            )
            decode_positions = torch.tensor(
                [4, 0, 0, 0, 0, 0, 0, 0], device="cuda"
            )
            split_output = split_model(
                decode_tokens,
                decode_positions,
                split_cache,
                _metadata(1, 4, is_prompt=False, state_cache=split_cache),
                None,
            )[0]

            full_tokens = torch.cat(
                (tokens, torch.zeros(3, dtype=torch.long, device="cuda"))
            )
            full_positions = torch.tensor(
                [0, 1, 2, 3, 4, 0, 0, 0], device="cuda"
            )
            full_output = full_model(
                full_tokens,
                full_positions,
                full_cache,
                _metadata(5, 0, is_prompt=True, state_cache=full_cache),
                None,
            )[4]

        self.assertTrue(torch.isfinite(split_output).all())
        torch.testing.assert_close(
            split_output,
            full_output,
            rtol=3e-2,
            atol=3e-2,
        )

    def test_chunked_prefill_matches_single_prefill(self):
        torch.manual_seed(107)
        config = _config()
        chunked_model = Qwen3_5Model(config).cuda().half().eval()
        for parameter in chunked_model.parameters():
            if parameter.numel():
                torch.nn.init.normal_(parameter, mean=0.0, std=0.02)
        full_model = Qwen3_5Model(config).cuda().half().eval()
        full_model.load_state_dict(chunked_model.state_dict())
        chunked_cache = _hybrid_cache(config, _kv_cache(torch.float16))
        full_cache = _hybrid_cache(config, _kv_cache(torch.float16))
        tokens = torch.tensor([3, 5, 7, 9, 11], device="cuda")

        with torch.inference_mode():
            start = 0
            chunked_output = None
            for query_len in (2, 2, 1):
                chunk_tokens = torch.zeros(8, dtype=torch.long, device="cuda")
                chunk_positions = torch.zeros(
                    8, dtype=torch.long, device="cuda"
                )
                chunk_tokens[:query_len] = tokens[start:start + query_len]
                chunk_positions[:query_len] = torch.arange(
                    start, start + query_len, device="cuda"
                )
                chunked_output = chunked_model(
                    chunk_tokens,
                    chunk_positions,
                    chunked_cache,
                    _metadata(
                        query_len,
                        start,
                        is_prompt=True,
                        state_cache=chunked_cache,
                    ),
                    None,
                )
                start += query_len

            full_tokens = torch.cat(
                (tokens, torch.zeros(3, dtype=torch.long, device="cuda"))
            )
            full_positions = torch.tensor(
                [0, 1, 2, 3, 4, 0, 0, 0], device="cuda"
            )
            full_output = full_model(
                full_tokens,
                full_positions,
                full_cache,
                _metadata(5, 0, is_prompt=True, state_cache=full_cache),
                None,
            )

        assert chunked_output is not None
        torch.testing.assert_close(
            chunked_output[0], full_output[4], rtol=3e-2, atol=3e-2
        )

    def test_fp8_weights_remain_compressed_during_hybrid_forward(self):
        config = _config()
        config.quantization_config = {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [16, 16],
        }
        old_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float16)
            model = Qwen3_5Model(config).cuda().eval()
        finally:
            torch.set_default_dtype(old_dtype)
        initialize_dummy_weights(model)
        cache = _hybrid_cache(config, _kv_cache(torch.float16))
        tokens = torch.tensor(
            [3, 5, 7, 9, 0, 0, 0, 0], device="cuda"
        )
        positions = torch.tensor(
            [0, 1, 2, 3, 0, 0, 0, 0], device="cuda"
        )

        with torch.inference_mode():
            output = model(
                tokens,
                positions,
                cache,
                _metadata(4, 0, is_prompt=True, state_cache=cache),
                None,
            )

        self.assertEqual(
            model.layers[1].self_attn.qkv_gate_proj.weight.dtype,
            torch.float8_e4m3fn,
        )
        self.assertTrue(torch.isfinite(output[:4]).all())

    def test_native_mtp_proposes_and_verifies_one_token(self):
        torch.manual_seed(113)
        config = _config()
        model = Qwen3_5ForConditionalGeneration(
            config
        ).cuda().half().eval()
        for parameter in model.parameters():
            if parameter.numel():
                torch.nn.init.normal_(parameter, mean=0.0, std=0.02)
        cache = _hybrid_cache(config, _kv_cache(torch.float16))
        prompt = [3, 5, 7, 9]
        prompt_tokens = torch.tensor(
            prompt + [0, 0, 0, 0], device="cuda"
        )
        prompt_positions = torch.tensor(
            [0, 1, 2, 3, 0, 0, 0, 0], device="cuda"
        )

        with torch.inference_mode():
            first = model(
                prompt_tokens,
                prompt_positions,
                cache,
                _metadata(
                    4,
                    0,
                    is_prompt=True,
                    state_cache=cache,
                    enable_mtp=True,
                ),
                None,
            )[10]

            self.assertIsNotNone(first.draft_token_id)
            current_token = first.output_token
            draft_token = first.draft_token_id
            seq_data = {10: SequenceData(prompt + [current_token])}
            speculative_metadata = InputMetadata(
                seq_groups=[],
                seq_data=seq_data,
                prompt_lens=[2],
                slot_mapping=torch.tensor(
                    [4, 5], dtype=torch.int32, device="cuda"
                ),
                context_lens=torch.empty(
                    0, dtype=torch.int32, device="cuda"
                ),
                max_context_len=0,
                block_tables=torch.empty(
                    (0, 0), dtype=torch.int32, device="cuda"
                ),
                fresh_prompt_lens=[],
                cached_prompt_query_lens=[2],
                cached_prompt_cu_seqlens=torch.tensor(
                    [0, 2], dtype=torch.int32, device="cuda"
                ),
                cached_prompt_context_lens=torch.tensor(
                    [6], dtype=torch.int32, device="cuda"
                ),
                cached_prompt_block_tables=torch.tensor(
                    [[0]], dtype=torch.int32, device="cuda"
                ),
                max_cached_prompt_context_len=6,
                prompt_seq_ids=[10],
                prompt_sample_indices=[],
                speculative_seq_ids=[10],
                speculative_token_ids=[draft_token],
                speculative_hidden_indices=[(0, 1)],
                speculative_sampling_params=[
                    SamplingParams(temperature=0.0)
                ],
                enable_mtp=True,
            )
            speculative_metadata.state_slot_mapping = cache.acquire([10])
            verified = model(
                torch.tensor(
                    [current_token, draft_token, 0, 0, 0, 0, 0, 0],
                    device="cuda",
                ),
                torch.tensor(
                    [4, 5, 0, 0, 0, 0, 0, 0], device="cuda"
                ),
                cache,
                speculative_metadata,
                None,
            )[10]

        self.assertIn(verified.num_computed_tokens, (1, 2))
        self.assertIn(len(verified.output_token_ids), (1, 2))
        self.assertIsNotNone(verified.draft_token_id)

    def test_image_and_video_features_flow_through_hybrid_prefill(self):
        torch.manual_seed(127)
        config = _multimodal_config()
        model = Qwen3_5ForConditionalGeneration(
            config
        ).cuda().half().eval()
        for parameter in model.parameters():
            if parameter.numel():
                torch.nn.init.normal_(parameter, mean=0.0, std=0.02)
        cache = _hybrid_cache(config, _kv_cache(torch.float16))
        prompt = [3, 120, 4, 121, 5, 121, 6]
        multimodal = MultiModalInputs(
            token_type_ids=(0, 1, 0, 2, 0, 2, 0),
            pixel_values=torch.randn(4, 24),
            image_grid_thw=((1, 2, 2),),
            pixel_values_videos=torch.randn(8, 24),
            video_grid_thw=((2, 2, 2),),
        ).with_positions(prompt, spatial_merge_size=2)
        positions = torch.zeros(3, 8, dtype=torch.long, device="cuda")
        positions[:, :7] = torch.tensor(
            multimodal.position_ids, dtype=torch.long, device="cuda"
        )
        metadata = InputMetadata(
            seq_groups=[([10], SamplingParams(temperature=0.0))],
            seq_data={10: SequenceData(prompt)},
            prompt_lens=[7],
            slot_mapping=torch.arange(7, dtype=torch.int32, device="cuda"),
            context_lens=torch.empty(0, dtype=torch.int32, device="cuda"),
            max_context_len=0,
            block_tables=torch.empty((0, 0), dtype=torch.int32, device="cuda"),
            fresh_prompt_lens=[7],
            prompt_seq_ids=[10],
            multimodal_inputs={10: multimodal},
            multimodal_token_maps=[
                (1, 10, 1, 0),
                (3, 10, 2, 0),
                (5, 10, 2, 1),
            ],
        )
        metadata.state_slot_mapping = cache.acquire([10])

        with torch.inference_mode():
            output = model(
                torch.tensor(prompt + [0], device="cuda"),
                positions,
                cache,
                metadata,
                None,
            )

        self.assertIn(10, output)
        self.assertIn(10, model._multimodal_feature_cache)
        cached = model._multimodal_feature_cache[10]
        self.assertEqual(cached[1].shape, (1, 128))
        self.assertEqual(cached[2].shape, (2, 128))

    def test_chunked_visual_feature_offsets_and_decode_positions(self):
        prompt = [3, 120, 120, 120, 120, 4]
        multimodal = MultiModalInputs(
            token_type_ids=(0, 1, 1, 1, 1, 0),
            pixel_values=torch.randn(16, 24),
            image_grid_thw=((1, 4, 4),),
        ).with_positions(prompt, spatial_merge_size=2)
        worker = object.__new__(Worker)
        worker.block_size = 8
        worker.scheduler_config = SimpleNamespace(num_speculative_tokens=0)

        second_chunk = SequenceGroupMetadata(
            request_id="image",
            is_prompt=True,
            seq_data={10: SequenceData(prompt)},
            sampling_params=SamplingParams(temperature=0.0),
            block_tables={10: [0]},
            num_computed_tokens={10: 3},
            num_scheduled_tokens={10: 3},
            multi_modal_inputs=multimodal,
        )
        _, positions, metadata = worker._prepare_inputs([second_chunk])

        self.assertEqual(positions.shape, (3, 8))
        self.assertEqual(
            metadata.multimodal_token_maps,
            [(0, 10, 1, 2), (1, 10, 1, 3)],
        )
        self.assertEqual(
            positions[:, :3].cpu().tolist(),
            [
                [1, 1, 3],
                [2, 2, 3],
                [1, 2, 3],
            ],
        )

        seq_data = SequenceData(prompt)
        seq_data.append_token_id(7, 0.0)
        decode = SequenceGroupMetadata(
            request_id="image",
            is_prompt=False,
            seq_data={10: seq_data},
            sampling_params=SamplingParams(temperature=0.0),
            block_tables={10: [0]},
            num_computed_tokens={10: 6},
            num_scheduled_tokens={10: 1},
            multi_modal_inputs=multimodal,
        )
        _, decode_positions, _ = worker._prepare_inputs([decode])
        expected_decode_position = 6 + multimodal.rope_delta
        self.assertEqual(
            decode_positions[:, 0].cpu().tolist(),
            [expected_decode_position] * 3,
        )


if __name__ == "__main__":
    unittest.main()
