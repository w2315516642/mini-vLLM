import unittest
from types import SimpleNamespace

import torch

from prefix_cache_test_utils import make_group, make_scheduler, make_seq, sequence
from minivllm.spec_decode.greedy_verifier import verify_greedy_block
from minivllm.spec_decode.state_transaction import (
    build_speculative_replay_plan,
)


def _logits(token_ids, vocab_size=128):
    logits = torch.full((len(token_ids), vocab_size), -10.0)
    for row, token_id in enumerate(token_ids):
        logits[row, token_id] = 10.0
    return logits


class GreedyBlockVerifierTest(unittest.TestCase):

    def test_all_drafts_are_followed_by_bonus(self):
        result = verify_greedy_block(
            _logits([11, 12, 13, 99]),
            [11, 12, 13],
            is_eos=lambda _: False,
            ignore_eos=False,
        )

        self.assertEqual(result.token_ids, (11, 12, 13, 99))
        self.assertEqual(result.logit_indices, (0, 1, 2, 3))
        self.assertEqual(result.committed_tokens, 4)
        self.assertEqual(result.last_committed_index, 3)
        self.assertTrue(result.all_accepted)

    def test_first_mismatch_emits_target_correction(self):
        result = verify_greedy_block(
            _logits([11, 77, 13, 99]),
            [11, 12, 13],
            is_eos=lambda _: False,
            ignore_eos=False,
        )

        self.assertEqual(result.token_ids, (11, 77))
        self.assertEqual(result.logit_indices, (0, 1))
        self.assertEqual(result.accepted_draft_tokens, 1)
        self.assertEqual(result.committed_tokens, 2)
        self.assertEqual(result.last_committed_index, 1)
        self.assertFalse(result.all_accepted)

    def test_accepted_eos_does_not_append_bonus(self):
        result = verify_greedy_block(
            _logits([2, 19]),
            [2],
            is_eos=lambda token_id: token_id == 2,
            ignore_eos=False,
        )

        self.assertEqual(result.token_ids, (2,))
        self.assertEqual(result.committed_tokens, 2)
        self.assertTrue(result.all_accepted)


class DSparkSchedulerTest(unittest.TestCase):

    @staticmethod
    def _greedy_params():
        return SimpleNamespace(
            temperature=0.0,
            best_of=1,
            use_beam_search=False,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            stop=[],
            max_tokens=32,
        )

    def _running_sequence(self):
        scheduler = make_scheduler(max_tokens=16, num_speculative_tokens=7)
        seq = make_seq(0, [1, 2, 3, 4])
        seq.append_token_id(90, {90: 0.0})
        seq.set_speculative_tokens([91, 92, 93, 94, 95, 96, 97])
        group = make_group("dspark", [seq])
        group.sampling_params = self._greedy_params()
        scheduler.block_manager.allocate(group)
        seq.status = sequence.SequenceStatus.RUNNING
        seq.num_computed_tokens = 4
        scheduler.running.append(group)
        return scheduler, seq

    def test_scheduler_packs_anchor_and_full_draft_block(self):
        scheduler, seq = self._running_sequence()

        metadata, scheduler_outputs = scheduler.scheduler()

        self.assertEqual(metadata[0].num_scheduled_tokens[seq.seq_id], 8)
        self.assertEqual(
            metadata[0].speculative_token_blocks[seq.seq_id],
            [91, 92, 93, 94, 95, 96, 97],
        )
        self.assertEqual(len(scheduler.block_manager.get_block_table(seq)), 3)
        self.assertIn(seq.seq_id, scheduler_outputs.speculative_seq_ids)

    def test_partial_acceptance_advances_only_committed_inputs(self):
        scheduler, seq = self._running_sequence()
        _, scheduler_outputs = scheduler.scheduler()
        output = sequence.SequenceOutputs(
            seq_id=seq.seq_id,
            parent_seq_id=seq.seq_id,
            output_token=91,
            logprobs={91: 0.0},
            output_token_ids=[91, 92, 111],
            output_logprobs=[{91: 0.0}, {92: 0.0}, {111: 0.0}],
            num_computed_tokens=3,
        )

        scheduler.update({seq.seq_id: output}, scheduler_outputs)

        self.assertEqual(seq.num_computed_tokens, 7)
        self.assertEqual(seq.get_output_token_ids(), [90, 91, 92, 111])
        self.assertEqual(seq.speculative_token_ids, [])

    def test_external_drafter_supports_stochastic_width_one(self):
        scheduler, seq = self._running_sequence()
        scheduler.scheduler_config.num_speculative_tokens = 1
        scheduler.scheduler_config.draft_model = "draft"
        scheduler.running[0].sampling_params = SimpleNamespace(
            temperature=0.7,
            best_of=1,
            use_beam_search=False,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            stop=[],
            max_tokens=32,
        )

        self.assertTrue(
            scheduler._can_speculate(scheduler.running[0], draft_width=1)
        )


class GDNStateTransactionTest(unittest.TestCase):

    def test_replay_plan_contains_anchor_and_accepted_drafts_only(self):
        data = sequence.SequenceData([1, 2, 3, 4])
        data.append_token_id(90, 0.0)
        metadata = sequence.SequenceGroupMetadata(
            request_id="dspark",
            is_prompt=True,
            seq_data={7: data},
            sampling_params=self,
            block_tables={7: [3, 5, 8]},
            num_computed_tokens={7: 4},
            num_scheduled_tokens={7: 8},
            is_speculative=True,
            speculative_token_blocks={7: [91, 92, 93, 94, 95, 96, 97]},
        )

        plan = build_speculative_replay_plan([metadata], {7: 3})

        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0].token_ids, (90, 91, 92))
        self.assertEqual(plan[0].positions, (4, 5, 6))
        self.assertEqual(plan[0].context_len, 7)
        self.assertEqual(plan[0].block_table, (3, 5, 8))


if __name__ == "__main__":
    unittest.main()
