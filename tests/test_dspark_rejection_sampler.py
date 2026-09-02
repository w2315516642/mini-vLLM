import unittest
from types import SimpleNamespace

import torch

from minivllm.spec_decode.rejection_sampler import (
    draft_block_probs,
    logits_to_probs,
    rejection_sample_block,
    target_block_probs,
)
from minivllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
)


class SamplingTransformTest(unittest.TestCase):

    def test_temperature_top_k_and_top_p_are_normalized(self):
        logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])

        probs = logits_to_probs(
            logits, temperature=0.5, top_p=0.95, top_k=2
        )

        self.assertAlmostEqual(float(probs.sum()), 1.0, places=6)
        self.assertEqual(int(torch.count_nonzero(probs)), 2)
        self.assertEqual(int(torch.argmax(probs)), 0)

    def test_penalties_see_preceding_draft_tokens(self):
        logits = torch.zeros(2, 4)
        probs = draft_block_probs(
            logits,
            output_history=[1],
            draft_token_ids=[2, 3],
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            presence_penalty=1.0,
            frequency_penalty=0.0,
        )

        self.assertLess(float(probs[0, 1]), float(probs[0, 0]))
        self.assertLess(float(probs[1, 2]), float(probs[1, 0]))

    def test_target_has_one_extra_bonus_distribution(self):
        probs = target_block_probs(
            torch.zeros(3, 5),
            output_history=[],
            draft_token_ids=[1, 2],
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            presence_penalty=0.0,
            frequency_penalty=0.0,
        )

        self.assertEqual(probs.shape, (3, 5))
        torch.testing.assert_close(probs.sum(-1), torch.ones(3))


class RejectionSamplerTest(unittest.TestCase):

    def test_rejection_samples_positive_residual(self):
        target = torch.tensor([[0.2, 0.8], [0.0, 1.0]])
        draft = torch.tensor([[0.8, 0.2]])

        result = rejection_sample_block(
            target,
            draft,
            [0],
            is_eos=lambda _: False,
            ignore_eos=False,
            uniforms=[0.5],
        )

        self.assertEqual(result.token_ids, (1,))
        self.assertEqual(result.committed_tokens, 1)
        self.assertEqual(result.accepted_draft_tokens, 0)

    def test_all_accepted_drafts_receive_target_bonus(self):
        target = torch.tensor([
            [0.8, 0.2, 0.0],
            [0.1, 0.8, 0.1],
            [0.0, 0.0, 1.0],
        ])
        draft = torch.tensor([
            [0.5, 0.5, 0.0],
            [0.1, 0.7, 0.2],
        ])

        result = rejection_sample_block(
            target,
            draft,
            [0, 1],
            is_eos=lambda _: False,
            ignore_eos=False,
            uniforms=[0.9, 0.9],
        )

        self.assertEqual(result.token_ids, (0, 1, 2))
        self.assertEqual(result.committed_tokens, 3)
        self.assertTrue(result.all_accepted)

    def test_empirical_first_token_matches_target_distribution(self):
        target = torch.tensor([[0.25, 0.75], [1.0, 0.0]])
        draft = torch.tensor([[0.75, 0.25]])
        generator = torch.Generator().manual_seed(1234)
        counts = torch.zeros(2)

        for _ in range(4000):
            draft_token = int(torch.multinomial(
                draft[0], 1, generator=generator
            ))
            result = rejection_sample_block(
                target,
                draft,
                [draft_token],
                is_eos=lambda _: False,
                ignore_eos=False,
                generator=generator,
            )
            counts[result.token_ids[0]] += 1

        empirical = counts / counts.sum()
        torch.testing.assert_close(empirical, target[0], rtol=0.0, atol=0.025)


class TargetVerifierIntegrationTest(unittest.TestCase):

    def test_qwen_dispatches_stochastic_block_to_rejection_sampler(self):
        class FakeTarget:
            config = SimpleNamespace(vocab_size=4)

            @staticmethod
            def _compute_logits(hidden_states):
                return hidden_states.float()

            @staticmethod
            def _is_eos(_):
                return False

            @staticmethod
            def _token_logprobs(logprobs, token_id, _):
                return {token_id: float(logprobs[token_id])}

            @staticmethod
            def _position_at(positions, index):
                return positions[index]

        logits = torch.tensor([
            [-9.0, 9.0, -9.0, -9.0],
            [-9.0, -9.0, 9.0, -9.0],
        ])
        params = SimpleNamespace(
            temperature=1.0,
            top_p=1.0,
            top_k=1,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            ignore_eos=False,
            logprobs=0,
        )
        metadata = SimpleNamespace(
            speculative_seq_ids=[7],
            speculative_token_blocks=[[1]],
            speculative_hidden_indices=[(0, 1)],
            speculative_sampling_params=[params],
            speculative_draft_probs=[torch.tensor([[0.0, 1.0, 0.0, 0.0]])],
            seq_data={7: SimpleNamespace(output_token_ids=[])},
        )

        outputs, contexts = (
            Qwen3_5ForConditionalGeneration._verify_speculative_tokens(
                FakeTarget(), logits, torch.tensor([4, 5]), metadata
            )
        )

        self.assertEqual(outputs[7].output_token_ids, [1, 2])
        self.assertEqual(outputs[7].num_computed_tokens, 2)
        self.assertEqual(len(contexts), 1)


if __name__ == "__main__":
    unittest.main()
