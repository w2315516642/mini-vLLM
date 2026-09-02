"""Reference coordinator that joins P/D control and cache data planes."""

from __future__ import annotations

import uuid
from typing import List

from minivllm.distributed.kv_transfer import (
    KVTransferPlanner,
    PDTransferTopology,
    TransferStatus,
)
from minivllm.engine.pd_coordinator import (
    DecodeReservation,
    PrefillDecodeCoordinator,
)
from minivllm.engine.pd_handoff import RequestHandoff


class PDEngineBridge:
    """Move sealed requests between equal-TP P and D engine instances."""

    def __init__(self, prefill_engine, decode_engine) -> None:
        self.prefill_engine = prefill_engine
        self.decode_engine = decode_engine
        self.coordinator = PrefillDecodeCoordinator()

    def transfer(self, handoff: RequestHandoff) -> DecodeReservation:
        """Reserve D, push every rank's cache, then atomically activate D."""
        request_id = handoff.request_id
        self.coordinator.create(request_id)
        self.coordinator.start_prefill(request_id)
        self.coordinator.seal(handoff)
        reservation = None
        transfer_id = f"{request_id}/{uuid.uuid4().hex}"
        try:
            reservation = self.decode_engine.prepare_decode_handoff(handoff)
            self.coordinator.reserve_decode(request_id, reservation)
            source_layouts = self.prefill_engine.get_transfer_layouts()
            target_layouts = self.decode_engine.get_transfer_layouts()
            topology = PDTransferTopology.build(
                source_layouts, target_layouts
            )
            self.coordinator.start_transfer(request_id, transfer_id)
            sequence = handoff.sequences[0]
            num_blocks = (
                sequence.num_computed_tokens
                + topology.pairs[0].source.block_size
                - 1
            ) // topology.pairs[0].source.block_size
            target_blocks = reservation.block_tables[sequence.seq_id]
            plans = []
            for pair in topology.pairs:
                rank = pair.rank
                plan = KVTransferPlanner.build_plan(
                    transfer_id=f"{transfer_id}/rank-{rank}",
                    request_id=request_id,
                    source=pair.source,
                    target=pair.target,
                    source_block_ids=sequence.source_block_ids[:num_blocks],
                    target_block_ids=target_blocks[:num_blocks],
                    num_tokens=sequence.num_computed_tokens,
                    source_state_slot=sequence.source_state_slot,
                    target_state_slot=reservation.state_slots.get(
                        sequence.seq_id
                    ),
                    metadata={"rank": rank},
                )
                plans.append(plan)
            if hasattr(self.prefill_engine, "execute_cache_transfers"):
                results = self.prefill_engine.execute_cache_transfers(plans)
            else:
                results = [
                    self.prefill_engine.execute_rank_cache_transfer(rank, plan)
                    for rank, plan in enumerate(plans)
                ]
            for rank, result in enumerate(results):
                if result["status"] != TransferStatus.COMPLETED.value:
                    raise RuntimeError(
                        f"rank {rank} cache transfer failed: "
                        f"{result.get('error')}"
                    )
            self.coordinator.complete_transfer(request_id, transfer_id)
            self.decode_engine.activate_decode_handoff(request_id)
            self.prefill_engine.release_prefill_handoff(request_id)
            return reservation
        except Exception as exc:
            record = self.coordinator.get(request_id)
            if not record.state.is_terminal:
                self.coordinator.fail(request_id, str(exc))
            if reservation is not None:
                self.decode_engine.abort_decode_handoff(request_id)
            self.prefill_engine.release_prefill_handoff(request_id)
            raise

    def drain_prefill_engine(self) -> List[DecodeReservation]:
        """Transfer every request sealed by the latest P engine step."""
        return [
            self.transfer(handoff)
            for handoff in self.prefill_engine.pop_prefill_handoffs()
        ]
