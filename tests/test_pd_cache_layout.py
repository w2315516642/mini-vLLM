import unittest
from dataclasses import dataclass

import torch

from minivllm.distributed.kv_transfer import (
    InMemoryTransferBackend,
    InMemoryTransferRegistry,
    KVTransferPlanner,
    TransferEndpoint,
    TransferStatus,
    register_cache_layout,
)


@dataclass
class StatePool:
    conv_state: torch.Tensor
    recurrent_state: torch.Tensor


def make_cache(num_blocks: int):
    return {
        0: (
            torch.zeros(num_blocks, 2, 4, 3),
            torch.zeros(num_blocks, 2, 4, 3),
        ),
        2: (
            torch.zeros(num_blocks, 2, 4, 3),
            torch.zeros(num_blocks, 2, 4, 3),
        ),
    }


def make_states(num_slots: int):
    return {
        1: StatePool(
            conv_state=torch.zeros(num_slots, 6, 4),
            recurrent_state=torch.zeros(num_slots, 2, 3, 5),
        )
    }


def make_draft_cache(num_blocks: int):
    return {
        0: (
            torch.zeros(num_blocks, 1, 2, 4, 2),
            torch.zeros(num_blocks, 1, 4, 4),
        )
    }


class CacheTransferLayoutTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = InMemoryTransferRegistry()
        self.prefill = InMemoryTransferBackend(
            TransferEndpoint("prefill-0", "127.0.0.1:12000"),
            self.registry,
        )
        self.decode = InMemoryTransferBackend(
            TransferEndpoint("decode-0", "127.0.0.1:12001"),
            self.registry,
        )

    def tearDown(self) -> None:
        self.prefill.close()
        self.decode.close()

    def test_transfers_different_physical_blocks_and_hybrid_slot(self) -> None:
        source_cache = make_cache(5)
        target_cache = make_cache(7)
        source_states = make_states(3)
        target_states = make_states(4)
        for layer_idx, (key, value) in source_cache.items():
            for block_id in range(key.shape[0]):
                key[block_id].fill_(100 * layer_idx + 10 * block_id + 1)
                value[block_id].fill_(100 * layer_idx + 10 * block_id + 2)
        source_states[1].conv_state[2].fill_(31)
        source_states[1].recurrent_state[2].fill_(37)

        source_layout = register_cache_layout(
            self.prefill, 4, source_cache, source_states
        )
        target_layout = register_cache_layout(
            self.decode, 4, target_cache, target_states
        )
        plan = KVTransferPlanner.build_plan(
            transfer_id="request-a/0",
            request_id="request-a",
            source=source_layout,
            target=target_layout,
            source_block_ids=[3, 1],
            target_block_ids=[0, 6],
            num_tokens=7,
            source_state_slot=2,
            target_state_slot=1,
        )

        handle = self.prefill.submit(plan)

        self.assertEqual(handle.status, TransferStatus.COMPLETED)
        for layer_idx in source_cache:
            for source, target in zip(
                source_cache[layer_idx], target_cache[layer_idx]
            ):
                torch.testing.assert_close(target[0], source[3])
                torch.testing.assert_close(target[6], source[1])
        torch.testing.assert_close(
            target_states[1].conv_state[1], source_states[1].conv_state[2]
        )
        torch.testing.assert_close(
            target_states[1].recurrent_state[1],
            source_states[1].recurrent_state[2],
        )
        self.assertEqual(plan.metadata["num_tokens"], 7)
        self.assertEqual(len(plan.slices), 10)

    def test_rejects_wrong_block_count(self) -> None:
        source_layout = register_cache_layout(
            self.prefill, 4, make_cache(4)
        )
        target_layout = register_cache_layout(
            self.decode, 4, make_cache(4)
        )
        with self.assertRaisesRegex(ValueError, "requires 2 blocks"):
            KVTransferPlanner.build_plan(
                transfer_id="bad",
                request_id="request",
                source=source_layout,
                target=target_layout,
                source_block_ids=[0],
                target_block_ids=[1, 2],
                num_tokens=8,
            )

    def test_transfers_dspark_cache_with_target_block_mapping(self) -> None:
        source_cache = make_cache(4)
        target_cache = make_cache(6)
        source_draft = make_draft_cache(7)
        target_draft = make_draft_cache(9)
        source_draft[0][0][2].fill_(17)
        source_draft[0][1][2].fill_(19)

        source_layout = register_cache_layout(
            self.prefill,
            4,
            source_cache,
            draft_attention_caches=source_draft,
        )
        target_layout = register_cache_layout(
            self.decode,
            4,
            target_cache,
            draft_attention_caches=target_draft,
        )
        plan = KVTransferPlanner.build_plan(
            transfer_id="request-dspark/0",
            request_id="request-dspark",
            source=source_layout,
            target=target_layout,
            source_block_ids=[2],
            target_block_ids=[5],
            num_tokens=4,
        )

        self.prefill.submit(plan)

        torch.testing.assert_close(target_draft[0][0][5], source_draft[0][0][2])
        torch.testing.assert_close(target_draft[0][1][5], source_draft[0][1][2])
        self.assertEqual(len(plan.slices), 6)

    def test_rejects_incompatible_region_shape(self) -> None:
        source_layout = register_cache_layout(
            self.prefill,
            4,
            {0: (torch.zeros(2, 3), torch.zeros(2, 3))},
        )
        target_layout = register_cache_layout(
            self.decode,
            4,
            {0: (torch.zeros(2, 4), torch.zeros(2, 4))},
        )
        with self.assertRaisesRegex(ValueError, "incompatible"):
            KVTransferPlanner.build_plan(
                transfer_id="bad-shape",
                request_id="request",
                source=source_layout,
                target=target_layout,
                source_block_ids=[0],
                target_block_ids=[1],
                num_tokens=4,
            )


if __name__ == "__main__":
    unittest.main()
