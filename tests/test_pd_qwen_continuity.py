import unittest
from dataclasses import dataclass

import torch

from minivllm.distributed.kv_transfer import (
    InMemoryTransferBackend,
    InMemoryTransferRegistry,
    KVTransferPlanner,
    TransferEndpoint,
    register_cache_layout,
)
from minivllm.engine.pd_handoff import RequestHandoff
from minivllm.model_executor.layers.gated_delta_net import (
    causal_depthwise_conv1d_reference,
    recurrent_gated_delta_rule_reference,
)
from minivllm.multimodal import MultiModalInputs
from minivllm.sampling_params import SamplingParams
from minivllm.sequence import Sequence, SequenceGroup


@dataclass
class StatePool:
    conv_state: torch.Tensor
    recurrent_state: torch.Tensor


class QwenPDContinuityTest(unittest.TestCase):
    def test_transferred_post_prompt_state_matches_direct_continuation(self):
        torch.manual_seed(7)
        batch, prompt_len, channels, kernel = 1, 4, 6, 3
        heads, key_dim, value_dim = 2, 3, 4
        projected = torch.randn(batch, prompt_len + 1, channels)
        conv_weight = torch.randn(channels, kernel)
        query = torch.randn(batch, prompt_len + 1, heads, key_dim)
        key = torch.randn_like(query)
        value = torch.randn(batch, prompt_len + 1, heads, value_dim)
        log_decay = -torch.rand(batch, prompt_len + 1, heads)
        beta = torch.sigmoid(torch.randn(batch, prompt_len + 1, heads))

        _, prompt_conv_state = causal_depthwise_conv1d_reference(
            projected[:, :prompt_len], conv_weight
        )
        _, prompt_recurrent_state = recurrent_gated_delta_rule_reference(
            query[:, :prompt_len],
            key[:, :prompt_len],
            value[:, :prompt_len],
            log_decay[:, :prompt_len],
            beta[:, :prompt_len],
        )
        direct_conv, direct_conv_state = causal_depthwise_conv1d_reference(
            projected[:, prompt_len:], conv_weight, prompt_conv_state
        )
        direct_recurrent, direct_recurrent_state = (
            recurrent_gated_delta_rule_reference(
                query[:, prompt_len:],
                key[:, prompt_len:],
                value[:, prompt_len:],
                log_decay[:, prompt_len:],
                beta[:, prompt_len:],
                prompt_recurrent_state,
            )
        )

        source_pool = StatePool(
            conv_state=torch.zeros(2, channels, kernel),
            recurrent_state=torch.zeros(2, heads, key_dim, value_dim),
        )
        target_pool = StatePool(
            conv_state=torch.zeros(3, channels, kernel),
            recurrent_state=torch.zeros(3, heads, key_dim, value_dim),
        )
        source_pool.conv_state[1].copy_(prompt_conv_state[0])
        source_pool.recurrent_state[1].copy_(prompt_recurrent_state[0])
        source_kv = {0: (torch.randn(2, 3), torch.randn(2, 3))}
        target_kv = {0: (torch.zeros(2, 3), torch.zeros(2, 3))}
        registry = InMemoryTransferRegistry()
        source_backend = InMemoryTransferBackend(
            TransferEndpoint("p/0", "p:1"), registry
        )
        target_backend = InMemoryTransferBackend(
            TransferEndpoint("d/0", "d:2"), registry
        )
        source_layout = register_cache_layout(
            source_backend, 2, source_kv, {1: source_pool}
        )
        target_layout = register_cache_layout(
            target_backend, 2, target_kv, {1: target_pool}
        )
        plan = KVTransferPlanner.build_plan(
            "request/0",
            "request",
            source_layout,
            target_layout,
            source_block_ids=[0, 1],
            target_block_ids=[1, 0],
            num_tokens=prompt_len,
            source_state_slot=1,
            target_state_slot=2,
        )
        source_backend.submit(plan)

        moved_conv, moved_conv_state = causal_depthwise_conv1d_reference(
            projected[:, prompt_len:],
            conv_weight,
            target_pool.conv_state[2:3],
        )
        moved_recurrent, moved_recurrent_state = (
            recurrent_gated_delta_rule_reference(
                query[:, prompt_len:],
                key[:, prompt_len:],
                value[:, prompt_len:],
                log_decay[:, prompt_len:],
                beta[:, prompt_len:],
                target_pool.recurrent_state[2:3],
            )
        )
        torch.testing.assert_close(moved_conv, direct_conv)
        torch.testing.assert_close(moved_conv_state, direct_conv_state)
        torch.testing.assert_close(moved_recurrent, direct_recurrent)
        torch.testing.assert_close(
            moved_recurrent_state, direct_recurrent_state
        )
        source_backend.close()
        target_backend.close()

    def test_handoff_preserves_mtp_proposal_and_mrope_positions(self):
        sequence = Sequence(5, "vision", [10, 11, 12], block_size=2)
        sequence.num_computed_tokens = 3
        sequence.append_token_id(13, {13: -0.1})
        sequence.speculative_token_id = 14
        positions = ((0, 1, 2), (0, 2, 3), (0, 3, 4))
        group = SequenceGroup(
            "vision-request",
            [sequence],
            SamplingParams(temperature=0.0, max_tokens=8),
            arrival_time=1.0,
            multi_modal_inputs=MultiModalInputs(
                token_type_ids=(0, 1, 0),
                position_ids=positions,
                rope_delta=2,
            ),
        )

        restored = RequestHandoff.from_dict(
            RequestHandoff.from_sequence_group(
                group,
                block_tables={5: (3, 1)},
                state_slots={5: 0},
            ).to_dict()
        ).rebuild_sequence_group(block_size=2)

        self.assertEqual(restored.seqs[0].speculative_token_id, 14)
        self.assertEqual(restored.multi_modal_inputs.position_ids, positions)
        self.assertEqual(restored.multi_modal_inputs.rope_delta, 2)
        self.assertIsNone(restored.multi_modal_inputs.pixel_values)
        self.assertIsNone(restored.multi_modal_inputs.pixel_values_videos)


if __name__ == "__main__":
    unittest.main()
