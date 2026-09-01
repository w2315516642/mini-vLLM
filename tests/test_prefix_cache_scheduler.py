import unittest
from types import SimpleNamespace

from prefix_cache_test_utils import (
    make_group,
    make_scheduler,
    make_seq,
    sequence,
)


def make_output(seq_id, token_id=99):
    return sequence.SequenceOutputs(
        seq_id=seq_id,
        parent_seq_id=seq_id,
        output_token=token_id,
        logprobs={token_id: 0.0},
    )


class PrefixCacheSchedulerTest(unittest.TestCase):

    @staticmethod
    def _greedy_params(max_tokens=8):
        return SimpleNamespace(
            temperature=0.0,
            best_of=1,
            use_beam_search=False,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            stop=[],
            max_tokens=max_tokens,
        )

    def _cache_prompt(self, scheduler, seq):
        group = make_group("cached", [seq])
        scheduler.block_manager.allocate(group)
        seq.num_computed_tokens = seq.get_len()
        scheduler.block_manager.cache_blocks(seq)
        scheduler.block_manager.free(seq)

    def test_prompt_budget_counts_only_uncached_suffix(self):
        scheduler = make_scheduler(max_tokens=4)
        self._cache_prompt(scheduler, make_seq(0, list(range(8))))

        seq = make_seq(1, list(range(8)))
        scheduler.add_seq_group(make_group("request", [seq]))
        metadata, outputs = scheduler.scheduler()

        self.assertEqual(len(metadata), 1)
        self.assertTrue(metadata[0].is_prompt)
        self.assertEqual(metadata[0].num_computed_tokens[seq.seq_id], 4)
        self.assertEqual(metadata[0].num_scheduled_tokens[seq.seq_id], 4)
        self.assertEqual(outputs.num_scheduled_tokens[seq.seq_id], 4)

    def test_update_advances_progress_before_appending_sample(self):
        scheduler = make_scheduler(max_tokens=8)
        seq = make_seq(0, list(range(4)))
        scheduler.add_seq_group(make_group("request", [seq]))
        _, scheduler_outputs = scheduler.scheduler()

        scheduler.update(
            {seq.seq_id: make_output(seq.seq_id)}, scheduler_outputs)

        self.assertEqual(seq.num_computed_tokens, 4)
        self.assertEqual(seq.get_len(), 5)
        self.assertEqual(seq.num_cached_blocks, 1)

    def test_recompute_returns_group_and_resets_progress(self):
        scheduler = make_scheduler()
        seq = make_seq(0, list(range(4)))
        group = make_group("request", [seq])
        scheduler.block_manager.allocate(group)
        seq.status = sequence.SequenceStatus.RUNNING
        seq.num_computed_tokens = seq.get_len()

        scheduler._preempt_by_recompute(group)

        self.assertIs(scheduler.waiting[0], group)
        self.assertEqual(seq.status, sequence.SequenceStatus.WAITING)
        self.assertEqual(seq.num_computed_tokens, 0)
        self.assertEqual(seq.num_cached_blocks, 0)

    def test_long_prompt_runs_in_chunks_and_samples_only_at_the_end(self):
        scheduler = make_scheduler(max_tokens=4)
        seq = make_seq(0, list(range(10)))
        scheduler.add_seq_group(make_group("chunked", [seq]))

        for expected_start in (0, 4):
            metadata, outputs = scheduler.scheduler()
            self.assertEqual(
                metadata[0].num_computed_tokens[seq.seq_id], expected_start)
            self.assertEqual(
                metadata[0].num_scheduled_tokens[seq.seq_id], 4)
            self.assertFalse(metadata[0].do_sample)
            self.assertEqual(scheduler.update({}, outputs), [])

        metadata, outputs = scheduler.scheduler()
        self.assertEqual(metadata[0].num_computed_tokens[seq.seq_id], 8)
        self.assertEqual(metadata[0].num_scheduled_tokens[seq.seq_id], 2)
        self.assertTrue(metadata[0].do_sample)
        sampled = scheduler.update(
            {seq.seq_id: make_output(seq.seq_id)}, outputs)

        self.assertEqual(sampled[0].request_id, "chunked")
        self.assertEqual(seq.num_computed_tokens, 10)
        self.assertEqual(seq.get_len(), 11)

    def test_decode_and_prompt_chunk_share_the_token_budget(self):
        scheduler = make_scheduler(max_tokens=4)

        decode_seq = make_seq(0, list(range(4)))
        decode_seq.append_token_id(90, {90: 0.0})
        decode_group = make_group("decode", [decode_seq])
        scheduler.block_manager.allocate(decode_group)
        decode_seq.status = sequence.SequenceStatus.RUNNING
        decode_seq.num_computed_tokens = 4
        scheduler.running.append(decode_group)

        prompt_seq = make_seq(1, list(range(8)))
        scheduler.add_seq_group(make_group("prompt", [prompt_seq]))

        metadata, outputs = scheduler.scheduler()
        by_request = {item.request_id: item for item in metadata}
        self.assertFalse(by_request["decode"].is_prompt)
        self.assertTrue(by_request["decode"].do_sample)
        self.assertEqual(
            by_request["prompt"].num_scheduled_tokens[prompt_seq.seq_id], 3)
        self.assertFalse(by_request["prompt"].do_sample)
        self.assertEqual(sum(outputs.num_scheduled_tokens.values()), 4)

        sampled = scheduler.update(
            {decode_seq.seq_id: make_output(decode_seq.seq_id)}, outputs)
        self.assertEqual([group.request_id for group in sampled], ["decode"])
        self.assertEqual(prompt_seq.num_computed_tokens, 3)

    def test_recompute_queues_recurrent_state_release(self):
        scheduler = make_scheduler()
        seq = make_seq(7, list(range(4)))
        group = make_group("request", [seq])
        scheduler.block_manager.allocate(group)
        seq.status = sequence.SequenceStatus.RUNNING
        seq.num_computed_tokens = seq.get_len()

        scheduler._preempt_by_recompute(group)

        releases, copies = scheduler.pop_pending_state_operations()
        self.assertEqual(releases, [7])
        self.assertEqual(copies, {})

    def test_best_of_prompt_queues_recurrent_state_copy(self):
        scheduler = make_scheduler(max_tokens=8)
        first = make_seq(0, list(range(4)))
        second = make_seq(1, list(range(4)))
        scheduler.add_seq_group(make_group("best-of", [first, second]))

        _, outputs = scheduler.scheduler()
        scheduler.update(
            {
                first.seq_id: make_output(first.seq_id),
                second.seq_id: make_output(second.seq_id),
            },
            outputs,
        )

        releases, copies = scheduler.pop_pending_state_operations()
        self.assertEqual(releases, [])
        self.assertEqual(copies, {second.seq_id: first.seq_id})

    def test_mtp_verification_schedules_two_tokens_and_accepts_bonus(self):
        scheduler = make_scheduler(
            max_tokens=4, num_speculative_tokens=1
        )
        seq = make_seq(0, list(range(4)))
        seq.append_token_id(90, {90: 0.0})
        seq.speculative_token_id = 91
        group = make_group("mtp", [seq])
        group.sampling_params = self._greedy_params()
        scheduler.block_manager.allocate(group)
        seq.status = sequence.SequenceStatus.RUNNING
        seq.num_computed_tokens = 4
        scheduler.running.append(group)

        metadata, scheduler_outputs = scheduler.scheduler()

        self.assertTrue(metadata[0].is_prompt)
        self.assertTrue(metadata[0].is_speculative)
        self.assertFalse(metadata[0].do_sample)
        self.assertEqual(metadata[0].num_scheduled_tokens[seq.seq_id], 2)
        self.assertEqual(metadata[0].speculative_token_ids, {seq.seq_id: 91})

        accepted = sequence.SequenceOutputs(
            seq_id=seq.seq_id,
            parent_seq_id=seq.seq_id,
            output_token=91,
            logprobs={91: -0.1},
            output_token_ids=[91, 92],
            output_logprobs=[{91: -0.1}, {92: -0.2}],
            num_computed_tokens=2,
            draft_token_id=93,
        )
        scheduler.update({seq.seq_id: accepted}, scheduler_outputs)

        self.assertEqual(seq.num_computed_tokens, 6)
        self.assertEqual(seq.get_output_token_ids(), [90, 91, 92])
        self.assertEqual(seq.speculative_token_id, 93)

    def test_mtp_rejection_advances_only_the_confirmed_input(self):
        scheduler = make_scheduler(
            max_tokens=4, num_speculative_tokens=1
        )
        seq = make_seq(0, list(range(4)))
        seq.append_token_id(90, {90: 0.0})
        seq.speculative_token_id = 91
        group = make_group("mtp", [seq])
        group.sampling_params = self._greedy_params()
        scheduler.block_manager.allocate(group)
        seq.status = sequence.SequenceStatus.RUNNING
        seq.num_computed_tokens = 4
        scheduler.running.append(group)
        _, scheduler_outputs = scheduler.scheduler()

        rejected = sequence.SequenceOutputs(
            seq_id=seq.seq_id,
            parent_seq_id=seq.seq_id,
            output_token=95,
            logprobs={95: -0.1},
            num_computed_tokens=1,
            draft_token_id=96,
        )
        scheduler.update({seq.seq_id: rejected}, scheduler_outputs)

        self.assertEqual(seq.num_computed_tokens, 5)
        self.assertEqual(seq.get_output_token_ids(), [90, 95])
        self.assertEqual(seq.speculative_token_id, 96)


if __name__ == "__main__":
    unittest.main()
