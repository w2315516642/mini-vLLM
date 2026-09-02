import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from minivllm.model_executor.models.dspark import (
    DSparkDraftModel,
    _load_draft_qkv_weight,
    dual_source_attention_reference,
)
from minivllm.spec_decode.dspark_context import TargetHiddenStateCollector


def tiny_config():
    return SimpleNamespace(
        block_size=3,
        draft_vocab_size=11,
        vocab_size=11,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        hidden_act="silu",
        attention_bias=False,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=32,
        dspark_config={
            "mask_token_id": 10,
            "target_layer_ids": [1, 3],
            "markov_rank": 2,
            "enable_confidence_head": True,
            "confidence_head_with_markov": True,
        },
    )


class TargetHiddenStateCollectorTest(unittest.TestCase):
    def test_concatenate_follows_configured_layer_order(self):
        collector = TargetHiddenStateCollector([1, 3])
        collector(0, torch.full((2, 2), -1.0))
        collector(3, torch.full((2, 2), 3.0))
        collector(1, torch.full((2, 2), 1.0))

        result = collector.concatenate()

        self.assertEqual(result.tolist(), [[1.0, 1.0, 3.0, 3.0]] * 2)

    def test_missing_layer_is_reported(self):
        collector = TargetHiddenStateCollector([1, 3])
        collector(1, torch.zeros(1, 2))
        with self.assertRaisesRegex(RuntimeError, "missing layers \[3\]"):
            collector.concatenate()


class DualSourceAttentionTest(unittest.TestCase):
    def test_every_block_query_can_see_future_block_values(self):
        query = torch.ones(1, 2, 1, 1)
        context_key = torch.zeros(1, 1, 1, 1)
        context_value = torch.zeros(1, 1, 1, 1)
        block_key = torch.zeros(1, 2, 1, 1)
        block_value = torch.tensor([[[[2.0]], [[4.0]]]])

        output = dual_source_attention_reference(
            query, context_key, context_value, block_key, block_value, 1.0
        )

        torch.testing.assert_close(output, torch.full((1, 2, 1, 1), 2.0))


class DSparkDraftModelTest(unittest.TestCase):
    def test_tiny_reference_model_proposes_full_block(self):
        torch.manual_seed(7)
        tp_patches = (
            patch(
                "minivllm.model_executor.models.dspark."
                "get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch(
                "minivllm.model_executor.parallel_utils.tensor_parallel.layers."
                "get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch(
                "minivllm.model_executor.models.dspark."
                "gather_from_tensor_model_parallel_region",
                side_effect=lambda tensor: tensor,
            ),
        )
        for tp_patch in tp_patches:
            tp_patch.start()
            self.addCleanup(tp_patch.stop)
        model = DSparkDraftModel(tiny_config(), use_cpu_initialization=True)
        for parameter in model.parameters():
            if parameter.ndim > 1:
                torch.nn.init.normal_(parameter, std=0.02)
        target_hidden = torch.randn(1, 2, 16)
        context = model.project_target_hidden(target_hidden)
        proposal = model.propose_reference(
            input_embeddings=torch.randn(1, 3, 8),
            positions=torch.tensor([[2, 3, 4]]),
            context_hidden=context,
            context_positions=torch.tensor([[0, 1]]),
            lm_head_weight=torch.randn(11, 8),
            anchor_token_ids=torch.tensor([2]),
        )

        self.assertEqual(tuple(proposal.token_ids.shape), (1, 3))
        self.assertEqual(tuple(proposal.draft_logits.shape), (1, 3, 11))
        self.assertEqual(tuple(proposal.confidence.shape), (1, 3))
        self.assertEqual(proposal.draft_probs, (None,))

    def test_base_logits_gather_tp_shards_and_remove_padding(self):
        model = DSparkDraftModel.__new__(DSparkDraftModel)
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(hidden_size=2, vocab_size=3)
        hidden = torch.tensor([[1.0, 2.0]])
        local_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        with patch(
            "minivllm.model_executor.models.dspark."
            "gather_from_tensor_model_parallel_region",
            side_effect=lambda local: torch.cat((local, local[:, :1]), dim=-1),
        ):
            logits = model.compute_base_logits(hidden, local_weight)

        torch.testing.assert_close(logits, torch.tensor([[1.0, 2.0, 1.0]]))

    def test_qkv_loader_places_each_local_segment(self):
        parameter = torch.zeros(8, 3)
        q = torch.arange(12.0).reshape(4, 3)
        k = torch.full((2, 3), 20.0)
        v = torch.full((2, 3), 30.0)

        _load_draft_qkv_weight(parameter, q, "q", 0, 4, 2)
        _load_draft_qkv_weight(parameter, k, "k", 0, 4, 2)
        _load_draft_qkv_weight(parameter, v, "v", 0, 4, 2)

        torch.testing.assert_close(parameter, torch.cat((q, k, v)))


if __name__ == "__main__":
    unittest.main()
