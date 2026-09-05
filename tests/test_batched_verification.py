import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock

import torch

from minivllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration as Model
from minivllm.sampling_params import SamplingParams


class BatchedVerificationTest(unittest.TestCase):
    def test_ragged_reordered_blocks_match_individual_verification(self):
        torch.manual_seed(12)
        hidden = torch.randn(9, 8)
        positions = torch.arange(9)
        for temperature, ignore_eos in [(0.0, True), (0.0, False), (0.8, True)]:
            params = SamplingParams(temperature=temperature, ignore_eos=ignore_eos,
                                    logprobs=2)
            metadata = NS(
                speculative_seq_ids=[11, 22],
                speculative_token_blocks=[[1, 2], [3]],
                speculative_hidden_indices=[[4, 1, 7], [0, 5]],
                speculative_sampling_params=[params, params],
                speculative_draft_probs=[torch.full((2, 8), 1/8), torch.full((1, 8), 1/8)],
                seq_data={11: NS(output_token_ids=[]), 22: NS(output_token_ids=[])},
            )
            compute = Mock(side_effect=lambda h: h * 2)
            model = NS(_compute_logits=compute, _is_eos=lambda t: t == 2,
                       _token_logprobs=Model._token_logprobs,
                       _position_at=Model._position_at, config=NS(vocab_size=8))
            torch.manual_seed(23)
            actual, contexts = Model._verify_speculative_tokens(model, hidden, positions, metadata)
            self.assertEqual(compute.call_count, 1)
            torch.testing.assert_close(compute.call_args.args[0], hidden[[4, 1, 7, 0, 5]])
            torch.manual_seed(23)
            expected = {}
            for i, seq_id in enumerate(metadata.speculative_seq_ids):
                single = NS(**{key: ([value[i]] if key.startswith('speculative_') else value)
                               for key, value in vars(metadata).items()})
                result, _ = Model._verify_speculative_tokens(model, hidden, positions, single)
                expected.update(result)
            for seq_id in expected:
                self.assertEqual(actual[seq_id].output_token_ids, expected[seq_id].output_token_ids)
                self.assertEqual(actual[seq_id].num_computed_tokens, expected[seq_id].num_computed_tokens)
                self.assertEqual(actual[seq_id].output_logprobs, expected[seq_id].output_logprobs)

    def test_empty_verification_does_not_project(self):
        compute = Mock()
        metadata = NS(speculative_draft_probs=None, speculative_seq_ids=[],
                      speculative_hidden_indices=[])
        self.assertEqual(Model._verify_speculative_tokens(NS(_compute_logits=compute),
                         torch.empty(0, 8), torch.empty(0), metadata), ({}, []))
        compute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
