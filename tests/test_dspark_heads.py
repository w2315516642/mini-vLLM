import unittest
from types import SimpleNamespace

import torch

from minivllm.spec_decode.dspark_config import DSparkConfig
from minivllm.spec_decode.dspark_heads import (
    DSparkConfidenceHead,
    VanillaMarkov,
)


class DSparkConfigTest(unittest.TestCase):
    def test_published_qwen_fields_are_normalized(self):
        config = SimpleNamespace(
            block_size=7,
            draft_vocab_size=32,
            vocab_size=32,
            hidden_size=16,
            intermediate_size=48,
            num_hidden_layers=5,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=1024,
            dspark_config={
                "mask_token_id": 31,
                "target_layer_ids": [1, 3, 5],
                "markov_rank": 8,
                "enable_confidence_head": True,
                "confidence_head_with_markov": True,
            },
        )

        normalized = DSparkConfig.from_hf_config(config)

        self.assertEqual(normalized.block_size, 7)
        self.assertEqual(normalized.target_layer_ids, (1, 3, 5))
        self.assertEqual(normalized.verification_width, 8)

    def test_invalid_target_layers_are_rejected(self):
        config = SimpleNamespace(
            block_size=2,
            vocab_size=8,
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=2,
            max_position_embeddings=16,
            mask_token_id=7,
            target_layer_ids=[3, 1],
            markov_rank=2,
            enable_confidence_head=True,
            confidence_head_with_markov=True,
        )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            DSparkConfig.from_hf_config(config)


class VanillaMarkovTest(unittest.TestCase):
    def test_sampled_token_changes_next_transition(self):
        head = VanillaMarkov(vocab_size=3, markov_rank=1)
        with torch.no_grad():
            head.markov_w1.weight.copy_(torch.tensor([[0.0], [2.0], [-2.0]]))
            head.markov_w2.weight.copy_(torch.tensor([[0.0], [1.0], [-1.0]]))
        base = torch.zeros(1, 2, 3)

        output = head.sample_block(base, torch.tensor([1]))

        self.assertEqual(output.token_ids.tolist(), [[1, 1]])
        self.assertEqual(tuple(output.logits.shape), (1, 2, 3))
        self.assertEqual(tuple(output.previous_embeddings.shape), (1, 2, 1))

    def test_teacher_forcing_matches_manual_projection(self):
        torch.manual_seed(1)
        head = VanillaMarkov(vocab_size=5, markov_rank=2)
        base = torch.randn(2, 3, 5)
        previous = torch.tensor([[0, 1, 2], [2, 3, 4]])

        actual = head.apply_teacher_forced(base, previous)
        expected = base + head.markov_w2(head.markov_w1(previous))

        torch.testing.assert_close(actual, expected)


class DSparkConfidenceHeadTest(unittest.TestCase):
    def test_markov_features_and_sts_are_applied(self):
        head = DSparkConfidenceHead(2, 1, max_block_size=3)
        with torch.no_grad():
            head.proj.weight.copy_(torch.tensor([[1.0, 0.0, 1.0]]))
            head.proj.bias.zero_()
        head.set_sts_temperatures(torch.tensor([1.0, 2.0]))
        hidden = torch.tensor([[[1.0, 0.0], [2.0, 0.0]]])
        markov = torch.tensor([[[1.0], [2.0]]])

        actual = head.probabilities(hidden, markov)
        expected = torch.sigmoid(torch.tensor([[2.0, 2.0]]))

        torch.testing.assert_close(actual, expected)

    def test_missing_markov_features_are_rejected(self):
        head = DSparkConfidenceHead(4, 2)
        with self.assertRaisesRegex(ValueError, "previous_embeddings"):
            head(torch.zeros(1, 2, 4))


if __name__ == "__main__":
    unittest.main()
