import unittest

from tests import test_dspark_verification as verification_tests
from prefix_cache_test_utils import sequence


class SpeculativeStatsTest(unittest.TestCase):
    def check_acceptance(self, tokens, committed, accepted):
        fixture = verification_tests.DSparkSchedulerTest()
        scheduler, seq = fixture._running_sequence()
        _, schedule = scheduler.scheduler()
        scheduler.update({seq.seq_id: sequence.SequenceOutputs(
            seq_id=seq.seq_id, parent_seq_id=seq.seq_id,
            output_token=tokens[0], logprobs={tokens[0]: 0.},
            output_token_ids=tokens, output_logprobs=[{t: 0.} for t in tokens],
            num_computed_tokens=committed)}, schedule)
        self.assertEqual(scheduler.speculative_stats, {
            "verification_rounds": 1, "verified_draft_tokens": 7,
            "accepted_draft_tokens": accepted})

    def test_correction_is_not_an_accepted_draft(self):
        self.check_acceptance([91, 92, 111], 3, 2)

    def test_bonus_is_not_an_accepted_draft(self):
        self.check_acceptance([91, 92, 93, 94, 95, 96, 97, 111], 8, 7)

    def test_accepted_eos_is_counted_without_a_bonus(self):
        self.check_acceptance([91], 2, 1)

    def test_first_draft_rejection_counts_zero(self):
        self.check_acceptance([111], 1, 0)

    def test_actual_verification_width_is_counted(self):
        scheduler, seq = verification_tests.DSparkSchedulerTest()._running_sequence()
        seq.set_speculative_tokens([91, 92])
        _, schedule = scheduler.scheduler()
        scheduler.update({seq.seq_id: sequence.SequenceOutputs(
            seq_id=seq.seq_id, parent_seq_id=seq.seq_id,
            output_token=111, logprobs={111: 0.}, num_computed_tokens=1)}, schedule)
        self.assertEqual(scheduler.speculative_stats["verified_draft_tokens"], 2)


if __name__ == "__main__":
    unittest.main()
