import os
import unittest
from types import SimpleNamespace

import torch

from minivllm.model_executor.models.dspark import DSparkDraftModel
from minivllm.model_executor.parallel_utils import parallel_state
from minivllm.sampling_params import SamplingParams
from minivllm.spec_decode.draft_metadata import DraftAttentionMetadata


RUN_CUDA_DSPARK_TESTS = (
    os.environ.get("MINIVLLM_RUN_CUDA_DSPARK_TESTS") == "1"
    and torch.cuda.is_available()
)


def tiny_config():
    return SimpleNamespace(
        block_size=3,
        draft_vocab_size=17,
        vocab_size=17,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=64,
        hidden_act="silu",
        attention_bias=False,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        max_position_embeddings=32,
        dspark_config={
            "mask_token_id": 16,
            "target_layer_ids": [0],
            "markov_rank": 4,
            "enable_confidence_head": True,
            "confidence_head_with_markov": True,
        },
    )


@unittest.skipUnless(
    RUN_CUDA_DSPARK_TESTS,
    "set MINIVLLM_RUN_CUDA_DSPARK_TESTS=1 after rebuilding extensions",
)
class DSparkModelCudaTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = 1
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK = 0

    @classmethod
    def tearDownClass(cls):
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK = None

    def test_context_projection_then_paged_proposal(self):
        torch.manual_seed(29)
        model = DSparkDraftModel(tiny_config()).cuda().half().eval()
        for parameter in model.parameters():
            torch.nn.init.normal_(parameter, std=0.02)
        block_size = 8
        x = 16 // torch.tensor([], dtype=torch.float16).element_size()
        cache = [(
            torch.zeros(
                1, 1, 64 // x, block_size, x,
                dtype=torch.float16, device="cuda",
            ),
            torch.zeros(
                1, 1, 64, block_size,
                dtype=torch.float16, device="cuda",
            ),
        )]
        model.materialize_context_kv(
            torch.randn(2, 64, dtype=torch.float16, device="cuda"),
            torch.tensor([0, 1], device="cuda"),
            torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
            cache,
        )
        metadata = DraftAttentionMetadata(
            query_lens=[3],
            cu_seqlens_q=torch.tensor(
                [0, 3], dtype=torch.int32, device="cuda"
            ),
            context_lens=torch.tensor([5], dtype=torch.int32, device="cuda"),
            block_tables=torch.tensor(
                [[0]], dtype=torch.int32, device="cuda"
            ),
            slot_mapping=torch.tensor(
                [2, 3, 4], dtype=torch.int32, device="cuda"
            ),
        )

        proposal = model.propose_paged(
            torch.randn(3, 64, dtype=torch.float16, device="cuda"),
            torch.tensor([2, 3, 4], device="cuda"),
            cache,
            metadata,
            torch.randn(17, 64, dtype=torch.float16, device="cuda"),
            torch.tensor([2], device="cuda"),
            [SamplingParams(temperature=0.0)],
            [[2]],
        )
        torch.cuda.synchronize()

        self.assertEqual(tuple(proposal.token_ids.shape), (1, 3))
        self.assertEqual(tuple(proposal.confidence.shape), (1, 3))
        self.assertEqual(tuple(proposal.draft_logits.shape), (1, 0, 17))
        self.assertEqual(proposal.draft_probs, (None,))


if __name__ == "__main__":
    unittest.main()
