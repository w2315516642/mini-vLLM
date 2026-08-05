import unittest

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


if __name__ == "__main__":
    unittest.main()
