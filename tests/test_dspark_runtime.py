import unittest
from types import SimpleNamespace

import torch

from minivllm.configs.pd_config import PDRole
from minivllm.sequence import SequenceOutputs
from minivllm.worker.worker import Worker, _extend_draft_block_table


def sampling_params(**overrides):
    values = {
        "best_of": 1,
        "use_beam_search": False,
        "stop": [],
        "temperature": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "ignore_eos": False,
        "max_tokens": 32,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class DraftBlockTableTest(unittest.TestCase):

    def test_workspace_extends_only_missing_physical_blocks(self):
        extended = _extend_draft_block_table(
            [4, 5], start_position=15, query_len=7,
            block_size=8, workspace_block_ids=[90],
        )

        self.assertEqual(extended, [4, 5, 90])

    def test_workspace_shortage_is_reported(self):
        with self.assertRaisesRegex(RuntimeError, "workspace"):
            _extend_draft_block_table(
                [4], start_position=8, query_len=17,
                block_size=8, workspace_block_ids=[90],
            )


class DraftWorkerBookkeepingTest(unittest.TestCase):

    def make_worker(self):
        worker = Worker.__new__(Worker)
        worker.scheduler_config = SimpleNamespace(num_speculative_tokens=3)
        worker.draft_config = SimpleNamespace(
            block_size=3,
            max_position_embeddings=64,
        )
        worker.model = SimpleNamespace(
            config=SimpleNamespace(eos_token_id=2)
        )
        worker._draft_probabilities = {}
        worker.draft_model = object()
        worker.pd_config = SimpleNamespace(role=PDRole.UNIFIED)
        return worker

    def test_prefill_role_transfers_context_without_attaching_proposal(self):
        worker = self.make_worker()
        worker.pd_config = SimpleNamespace(role=PDRole.PREFILL)

        self.assertFalse(worker._should_attach_dspark_drafts())
        worker.pd_config = SimpleNamespace(role=PDRole.DECODE)
        self.assertTrue(worker._should_attach_dspark_drafts())

    def test_partial_verification_advances_from_committed_prefix(self):
        worker = self.make_worker()
        output = SequenceOutputs(
            seq_id=7,
            parent_seq_id=7,
            output_token=9,
            logprobs={9: 0.0},
            num_computed_tokens=2,
        )
        metadata = SimpleNamespace(
            sampling_params=sampling_params(),
            seq_data={7: SimpleNamespace(output_token_ids=[4, 5])},
            block_tables={7: [10, 11]},
            num_computed_tokens={7: 8},
            num_scheduled_tokens={7: 4},
        )

        requests = worker._collect_draft_requests({7: output}, [metadata])

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].next_position, 10)
        self.assertEqual(requests[0].output_history, [4, 5, 9])
        self.assertEqual(requests[0].draft_width, 3)

    def test_output_budget_truncates_next_draft(self):
        worker = self.make_worker()
        output = SequenceOutputs(7, 7, 9, {9: 0.0})
        metadata = SimpleNamespace(
            sampling_params=sampling_params(max_tokens=5),
            seq_data={7: SimpleNamespace(output_token_ids=[4, 5])},
            block_tables={7: [10]},
            num_computed_tokens={7: 3},
            num_scheduled_tokens={7: 1},
        )

        requests = worker._collect_draft_requests({7: output}, [metadata])

        self.assertEqual(requests[0].draft_width, 1)

    def test_stochastic_probability_block_is_consumed_once(self):
        worker = self.make_worker()
        saved = torch.rand(3, 5)
        worker._draft_probabilities[7] = saved
        metadata = SimpleNamespace(
            speculative_seq_ids=[7],
            speculative_token_blocks=[[1, 2]],
            speculative_sampling_params=[sampling_params(temperature=1.0)],
            speculative_draft_probs=[],
        )

        worker._attach_saved_draft_probabilities(metadata)

        self.assertIs(metadata.speculative_draft_probs[0]._base, saved)
        self.assertNotIn(7, worker._draft_probabilities)


if __name__ == "__main__":
    unittest.main()
