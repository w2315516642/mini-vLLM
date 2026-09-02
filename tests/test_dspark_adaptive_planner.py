import unittest
from types import SimpleNamespace

from prefix_cache_test_utils import make_group, make_scheduler, make_seq, sequence
from minivllm.spec_decode.adaptive_planner import (
    AdaptiveVerificationPlanner,
    VerificationCostProfile,
)


class AdaptiveVerificationPlannerTest(unittest.TestCase):

    @staticmethod
    def _planner():
        return AdaptiveVerificationPlanner(VerificationCostProfile(
            token_counts=(2, 3, 4, 5),
            latency_ms=(1.0, 1.2, 1.4, 3.0),
        ))

    def test_global_ranking_selects_only_contiguous_prefixes(self):
        plan = self._planner().plan(
            {10: [0.9, 0.8], 20: [0.6, 0.5]},
            max_total_tokens=5,
        )

        self.assertEqual(plan.draft_widths, {10: 2, 20: 0})
        self.assertEqual(plan.target_tokens, 4)
        self.assertAlmostEqual(plan.expected_output_tokens, 3.62)

    def test_budget_caps_number_of_verification_candidates(self):
        plan = self._planner().plan(
            {10: [0.99, 0.99, 0.99], 20: [0.98, 0.98]},
            max_total_tokens=3,
        )

        self.assertLessEqual(sum(plan.draft_widths.values()), 1)
        self.assertLessEqual(plan.target_tokens, 3)

    def test_low_survival_tail_is_removed_before_cost_search(self):
        planner = AdaptiveVerificationPlanner(
            VerificationCostProfile.linear(8),
            min_survival_probability=0.5,
        )

        plan = planner.plan({10: [0.8, 0.5, 0.9]}, max_total_tokens=8)

        self.assertLessEqual(plan.draft_widths[10], 1)


class AdaptiveSchedulerIntegrationTest(unittest.TestCase):

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

    def test_scheduler_sends_only_the_planned_draft_prefix(self):
        scheduler = make_scheduler(max_tokens=5, num_speculative_tokens=3)
        scheduler.adaptive_planner = AdaptiveVerificationPlanner(
            VerificationCostProfile(
                token_counts=(1, 2, 3, 4),
                latency_ms=(1.0, 1.1, 1.2, 4.0),
            )
        )
        seq = make_seq(0, [1, 2, 3, 4])
        seq.append_token_id(90, {90: 0.0})
        seq.set_speculative_tokens([91, 92, 93], [0.95, 0.9, 0.1])
        group = make_group("adaptive", [seq])
        group.sampling_params = self._greedy_params()
        scheduler.block_manager.allocate(group)
        seq.status = sequence.SequenceStatus.RUNNING
        seq.num_computed_tokens = 4
        scheduler.running.append(group)

        metadata, outputs = scheduler.scheduler()
        width = outputs.speculative_token_counts[seq.seq_id]

        self.assertGreater(width, 0)
        self.assertLess(width, 3)
        self.assertEqual(
            metadata[0].speculative_token_blocks[seq.seq_id],
            [91, 92, 93][:width],
        )
        self.assertEqual(
            metadata[0].num_scheduled_tokens[seq.seq_id], width + 1
        )


if __name__ == "__main__":
    unittest.main()
