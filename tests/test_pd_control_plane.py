import unittest

from minivllm.engine.pd_coordinator import (
    DecodeReservation,
    PDRequestState,
    PrefillDecodeCoordinator,
)
from minivllm.engine.pd_handoff import RequestHandoff
from minivllm.multimodal import MultiModalInputs
from minivllm.sampling_params import SamplingParams
from minivllm.sequence import Sequence, SequenceGroup


def make_prefilled_group(multimodal=False):
    sequence = Sequence(
        seq_id=9,
        prompt="hello",
        prompt_token_ids=[10, 11, 12],
        block_size=2,
    )
    sequence.num_computed_tokens = 3
    sequence.append_token_id(13, {13: -0.25, 8: -1.5})
    inputs = None
    if multimodal:
        inputs = MultiModalInputs(
            token_type_ids=(0, 1, 0),
            position_ids=((0, 1, 2), (0, 1, 2), (0, 1, 2)),
            rope_delta=-1,
        )
    return SequenceGroup(
        request_id="request-9",
        seqs=[sequence],
        sampling_params=SamplingParams(
            temperature=0.0, max_tokens=8, stop=["END"]
        ),
        arrival_time=12.5,
        multi_modal_inputs=inputs,
    )


class RequestHandoffTest(unittest.TestCase):
    def test_round_trip_rebuilds_decode_input(self) -> None:
        original = make_prefilled_group(multimodal=True)
        handoff = RequestHandoff.from_sequence_group(
            original,
            block_tables={9: (4, 1)},
            state_slots={9: 3},
        )

        restored = RequestHandoff.from_dict(handoff.to_dict())
        rebuilt = restored.rebuild_sequence_group(block_size=2)

        sequence = rebuilt.seqs[0]
        self.assertEqual(sequence.seq_id, 9)
        self.assertEqual(sequence.get_token_ids(), [10, 11, 12, 13])
        self.assertEqual(sequence.num_computed_tokens, 3)
        self.assertEqual(sequence.output_logprobs, [{13: -0.25, 8: -1.5}])
        self.assertEqual(restored.sequences[0].source_block_ids, (4, 1))
        self.assertEqual(restored.sequences[0].source_state_slot, 3)
        self.assertEqual(rebuilt.sampling_params.temperature, 0.0)
        self.assertEqual(rebuilt.sampling_params.stop, ["END"])
        self.assertEqual(rebuilt.multi_modal_inputs.rope_delta, -1)
        self.assertIsNone(rebuilt.multi_modal_inputs.pixel_values)

    def test_rejects_handoff_before_final_prefill_chunk(self) -> None:
        group = make_prefilled_group()
        group.seqs[0].num_computed_tokens = 2
        with self.assertRaisesRegex(ValueError, "entire prompt"):
            RequestHandoff.from_sequence_group(
                group, block_tables={9: (4, 1)}
            )

    def test_rejects_beam_handoff(self) -> None:
        group = make_prefilled_group()
        group.sampling_params.best_of = 2
        with self.assertRaisesRegex(ValueError, "one non-beam"):
            RequestHandoff.from_sequence_group(
                group, block_tables={9: (4, 1)}
            )


class CoordinatorTest(unittest.TestCase):
    def test_complete_pd_lifecycle(self) -> None:
        handoff = RequestHandoff.from_sequence_group(
            make_prefilled_group(), block_tables={9: (4, 1)}
        )
        coordinator = PrefillDecodeCoordinator()
        coordinator.create(handoff.request_id)
        coordinator.start_prefill(handoff.request_id)
        coordinator.seal(handoff)
        coordinator.reserve_decode(
            handoff.request_id,
            DecodeReservation(block_tables={9: (7, 2)}),
        )
        coordinator.start_transfer(handoff.request_id, "request-9/0")
        coordinator.complete_transfer(handoff.request_id, "request-9/0")
        coordinator.finish(handoff.request_id)

        record = coordinator.get(handoff.request_id)
        self.assertEqual(record.state, PDRequestState.FINISHED)
        self.assertEqual(record.reservation.block_tables[9], (7, 2))

    def test_decode_cannot_start_before_transfer(self) -> None:
        coordinator = PrefillDecodeCoordinator()
        coordinator.create("request")
        coordinator.start_prefill("request")
        with self.assertRaisesRegex(RuntimeError, "cannot transition"):
            coordinator.transition("request", PDRequestState.DECODING)

    def test_failure_is_terminal_and_keeps_reason(self) -> None:
        coordinator = PrefillDecodeCoordinator()
        coordinator.create("request")
        coordinator.start_prefill("request")
        coordinator.fail("request", "prefill worker exited")
        record = coordinator.get("request")
        self.assertEqual(record.state, PDRequestState.FAILED)
        self.assertEqual(record.error, "prefill worker exited")
        with self.assertRaisesRegex(RuntimeError, "already failed"):
            coordinator.cancel("request")


if __name__ == "__main__":
    unittest.main()
