import os
import unittest
from types import SimpleNamespace

import torch

from minivllm.configs.model_architecture import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
)
from minivllm.model_executor.input_metadata import InputMetadata
from minivllm.model_executor.models.qwen3_5 import Qwen3_5Model
from minivllm.model_executor.parallel_utils import parallel_state
from minivllm.sampling_params import SamplingParams
from minivllm.sequence import SequenceData
from minivllm.model_executor.weight_utils import initialize_dummy_weights
from minivllm.worker.hybrid_cache import (
    GatedDeltaNetStateSpec,
    HybridCache,
)


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
    )


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


def _metadata(query_len, start, *, is_prompt, state_cache):
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


if __name__ == "__main__":
    unittest.main()
