"""Independent Transformers oracle, using real mini-vLLM CUDA/model/cache code."""
from contextlib import ExitStack
import os
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from minivllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from minivllm.model_executor.parallel_utils import parallel_state
from minivllm.sampling_params import SamplingParams
from minivllm.sequence import SequenceData, SequenceGroupMetadata
from minivllm.worker.hybrid_cache import HybridCache, GatedDeltaNetStateSpec
from minivllm.worker.worker import Worker


@unittest.skipUnless(
    os.environ.get("MINIVLLM_RUN_QWEN_LOGITS_TESTS") == "1" and torch.cuda.is_available(),
    "set MINIVLLM_RUN_QWEN_LOGITS_TESTS=1 for the Transformers CUDA oracle",
)
class Stage8LogitsTest(unittest.TestCase):
    @torch.inference_mode()
    def test_prefill_decode_and_reused_slots_match_transformers(self):
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

        config = Qwen3_5TextConfig(
            hidden_size=32, intermediate_size=64, vocab_size=128,
            num_hidden_layers=4,
            layer_types=["linear_attention"] * 3 + ["full_attention"],
            num_attention_heads=2, num_key_value_heads=1, head_dim=64,
            linear_num_key_heads=2, linear_num_value_heads=4,
            linear_key_head_dim=16, linear_value_head_dim=16,
            linear_conv_kernel_dim=4, max_position_embeddings=128,
            tie_word_embeddings=True,
            rope_parameters={"rope_type": "default", "rope_theta": 10000000.0,
                             "partial_rotary_factor": 0.5, "mrope_section": [4, 6, 6]},
        )
        config._attn_implementation = "eager"
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), ExitStack() as stack:
                # Only bypass process-group setup at TP=1. Projections, RoPE,
                # attention, GDN and cache operators below are not mocked.
                for attr, value in (("_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE", 1),
                                    ("_MPU_TENSOR_MODEL_PARALLEL_RANK", 0)):
                    stack.enter_context(patch.object(parallel_state, attr, value))
                torch.manual_seed(108)
                # Quantize weights identically, but compute the independent
                # oracle in FP32 (including near-zero Q/K normalization).
                reference = Qwen3_5ForCausalLM(config).eval().to(dtype).float().cuda()
                old_dtype = torch.get_default_dtype()
                try:
                    torch.set_default_dtype(dtype)
                    actual = Qwen3_5ForConditionalGeneration(config).eval().cuda()
                finally:
                    torch.set_default_dtype(old_dtype)
                actual.load_weights_from_iterator(reference.state_dict().items())
                worker = self.make_worker(config, dtype)
                for prompt_len in (1, 5, 67):
                    # 67 crosses the 64-token GDN chunk and several KV blocks.
                    tokens = torch.randint(1, config.vocab_size, (prompt_len + 3,), device="cuda")
                    expected = reference(input_ids=tokens[None], use_cache=False).logits[0].float()
                    token_ids = tokens.tolist()
                    first_run = None
                    for run in range(2):
                        pieces = []
                        for start, size in [(0, prompt_len)] + [(i, 1) for i in range(prompt_len, len(token_ids))]:
                            seq = SequenceData(token_ids[:start + size])
                            metadata = SequenceGroupMetadata(
                                request_id="oracle", is_prompt=start == 0,
                                seq_data={7: seq}, sampling_params=SamplingParams(temperature=0),
                                block_tables={7: list(range(5))},
                                num_computed_tokens={7: start}, num_scheduled_tokens={7: size},
                            )
                            ids, positions, inputs = worker._prepare_inputs([metadata])
                            worker._prepare_hybrid_state([metadata], inputs)
                            hidden = actual.model(ids, positions, worker.hybrid_cache, inputs)
                            pieces.append(F.linear(hidden, actual.lm_head.weight).float())
                        logits = torch.cat(pieces)
                        torch.cuda.synchronize()
                        max_error = (logits - expected).abs().max().item()
                        print(f"oracle dtype={dtype} prompt={prompt_len} run={run} max_abs={max_error:.6g}")
                        rtol, atol = (0.003, 0.001) if dtype == torch.float16 else (0.03, 0.006)
                        torch.testing.assert_close(logits, expected, rtol=rtol, atol=atol)
                        if first_run is not None:
                            torch.testing.assert_close(logits, first_run, rtol=0, atol=0)
                        first_run = logits
                        worker.release_hybrid_state()
                        self.assertEqual(worker.hybrid_cache.num_active_slots, 0)
                del reference, actual, worker
                torch.cuda.empty_cache()

    @staticmethod
    def make_worker(config, dtype):
        block_size, num_blocks = 16, 5
        x = 16 // torch.tensor([], dtype=dtype).element_size()
        full = {3: (
            torch.empty(num_blocks, 1, config.head_dim // x, block_size, x, device="cuda", dtype=dtype),
            torch.empty(num_blocks, 1, config.head_dim, block_size, device="cuda", dtype=dtype),
        )}
        worker = Worker.__new__(Worker)
        worker.block_size = block_size
        worker.hybrid_cache = HybridCache(
            config.layer_types, full, GatedDeltaNetStateSpec.from_text_config(config),
            1, device="cuda",
        )
        worker.hybrid_seq_id = None
        worker.hybrid_num_computed = 0
        return worker


if __name__ == "__main__":
    unittest.main()
