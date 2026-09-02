import unittest

import torch

from minivllm.distributed.kv_transfer import (
    InMemoryTransferBackend,
    InMemoryTransferRegistry,
    TransferEndpoint,
    register_cache_layout,
)
from minivllm.engine.pd_coordinator import DecodeReservation, PDRequestState
from minivllm.engine.pd_handoff import RequestHandoff, SequenceHandoff
from minivllm.engine.pd_runtime import PDEngineBridge


class FakeEngine:
    def __init__(self, backend, layout, reservation=None):
        self.backend = backend
        self.layout = layout
        self.reservation = reservation
        self.activated = []
        self.released = []
        self.aborted = []
        self.handoffs = []

    def prepare_decode_handoff(self, handoff):
        return self.reservation

    def get_transfer_layouts(self):
        return [self.layout]

    def execute_rank_cache_transfer(self, rank, plan):
        handle = self.backend.submit(plan)
        return {
            "status": handle.status.value,
            "error": handle.error,
            "total_bytes": plan.total_bytes,
        }

    def activate_decode_handoff(self, request_id):
        self.activated.append(request_id)

    def release_prefill_handoff(self, request_id):
        self.released.append(request_id)

    def abort_decode_handoff(self, request_id):
        self.aborted.append(request_id)

    def pop_prefill_handoffs(self):
        values, self.handoffs = self.handoffs, []
        return values


class PDEngineBridgeTest(unittest.TestCase):
    def test_moves_cache_before_decode_activation(self):
        registry = InMemoryTransferRegistry()
        source_backend = InMemoryTransferBackend(
            TransferEndpoint("p/rank-0", "p:1"), registry
        )
        target_backend = InMemoryTransferBackend(
            TransferEndpoint("d/rank-0", "d:2"), registry
        )
        source_cache = {0: (torch.arange(12).reshape(3, 4), torch.arange(20, 32).reshape(3, 4))}
        target_cache = {0: (torch.zeros(4, 4, dtype=torch.int64), torch.zeros(4, 4, dtype=torch.int64))}
        source_layout = register_cache_layout(
            source_backend, 2, source_cache
        )
        target_layout = register_cache_layout(
            target_backend, 2, target_cache
        )
        handoff = RequestHandoff(
            request_id="request",
            sequences=(
                SequenceHandoff(
                    seq_id=3,
                    prompt="x",
                    prompt_token_ids=(1, 2, 3),
                    output_token_ids=(4,),
                    output_logprobs=({4: -0.1},),
                    num_computed_tokens=3,
                    source_block_ids=(2, 0),
                ),
            ),
            sampling_params={
                "n": 1, "best_of": 1, "presence_penalty": 0.0,
                "frequency_penalty": 0.0, "temperature": 0.0,
                "top_p": 1.0, "top_k": -1, "use_beam_search": False,
                "stop": [], "ignore_eos": False, "max_tokens": 4,
                "logprobs": None,
            },
            arrival_time=0.0,
        )
        prefill = FakeEngine(source_backend, source_layout)
        decode = FakeEngine(
            target_backend,
            target_layout,
            DecodeReservation(block_tables={3: (1, 3)}),
        )
        bridge = PDEngineBridge(prefill, decode)

        bridge.transfer(handoff)

        torch.testing.assert_close(target_cache[0][0][1], source_cache[0][0][2])
        torch.testing.assert_close(target_cache[0][0][3], source_cache[0][0][0])
        self.assertEqual(decode.activated, ["request"])
        self.assertEqual(prefill.released, ["request"])
        self.assertEqual(
            bridge.coordinator.get("request").state,
            PDRequestState.DECODING,
        )
        source_backend.close()
        target_backend.close()


if __name__ == "__main__":
    unittest.main()
