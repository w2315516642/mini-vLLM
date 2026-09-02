"""Small control-plane state machine for a P-push PD handoff."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Dict, Mapping, Optional, Tuple

from minivllm.engine.pd_handoff import RequestHandoff


class PDRequestState(str, Enum):
    QUEUED = "queued"
    PREFILLING = "prefilling"
    SEALED = "sealed"
    DESTINATION_READY = "destination_ready"
    TRANSFERRING = "transferring"
    DECODING = "decoding"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {
            PDRequestState.FINISHED,
            PDRequestState.FAILED,
            PDRequestState.CANCELLED,
        }


_ALLOWED_TRANSITIONS = {
    PDRequestState.QUEUED: {
        PDRequestState.PREFILLING,
        PDRequestState.CANCELLED,
    },
    PDRequestState.PREFILLING: {
        PDRequestState.SEALED,
        PDRequestState.FAILED,
        PDRequestState.CANCELLED,
    },
    PDRequestState.SEALED: {
        PDRequestState.DESTINATION_READY,
        PDRequestState.FAILED,
        PDRequestState.CANCELLED,
    },
    PDRequestState.DESTINATION_READY: {
        PDRequestState.TRANSFERRING,
        PDRequestState.FAILED,
        PDRequestState.CANCELLED,
    },
    PDRequestState.TRANSFERRING: {
        PDRequestState.DECODING,
        PDRequestState.FAILED,
        PDRequestState.CANCELLED,
    },
    PDRequestState.DECODING: {
        PDRequestState.FINISHED,
        PDRequestState.FAILED,
        PDRequestState.CANCELLED,
    },
}


@dataclass(frozen=True)
class DecodeReservation:
    """D-owned physical resources returned before P starts writing."""

    block_tables: Mapping[int, Tuple[int, ...]]
    state_slots: Mapping[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.block_tables:
            raise ValueError("decode reservation requires block tables")
        if any(not block_ids for block_ids in self.block_tables.values()):
            raise ValueError("decode block tables must not be empty")


@dataclass
class PDRequestRecord:
    request_id: str
    state: PDRequestState = PDRequestState.QUEUED
    handoff: Optional[RequestHandoff] = None
    reservation: Optional[DecodeReservation] = None
    transfer_id: Optional[str] = None
    error: Optional[str] = None
    updated_at: float = field(default_factory=time.monotonic)


class PrefillDecodeCoordinator:
    """Validate handoff ordering without owning either engine's scheduler."""

    def __init__(self) -> None:
        self._records: Dict[str, PDRequestRecord] = {}
        self._lock = RLock()

    def create(self, request_id: str) -> PDRequestRecord:
        if not request_id.strip():
            raise ValueError("request_id must not be empty")
        with self._lock:
            if request_id in self._records:
                raise ValueError(f"duplicate request id: {request_id}")
            record = PDRequestRecord(request_id)
            self._records[request_id] = record
            return record

    def get(self, request_id: str) -> PDRequestRecord:
        with self._lock:
            try:
                return self._records[request_id]
            except KeyError as exc:
                raise ValueError(f"unknown request id: {request_id}") from exc

    def transition(
        self,
        request_id: str,
        state: PDRequestState,
        error: Optional[str] = None,
    ) -> PDRequestRecord:
        with self._lock:
            record = self.get(request_id)
            if record.state.is_terminal:
                raise RuntimeError(
                    f"request {request_id} is already {record.state.value}"
                )
            if state not in _ALLOWED_TRANSITIONS[record.state]:
                raise RuntimeError(
                    f"request {request_id} cannot transition from "
                    f"{record.state.value} to {state.value}"
                )
            if state == PDRequestState.FAILED and not error:
                raise ValueError("failed requests require an error message")
            record.state = state
            record.error = error
            record.updated_at = time.monotonic()
            return record

    def start_prefill(self, request_id: str) -> None:
        self.transition(request_id, PDRequestState.PREFILLING)

    def seal(self, handoff: RequestHandoff) -> None:
        record = self.get(handoff.request_id)
        if record.state != PDRequestState.PREFILLING:
            raise RuntimeError("only a prefilling request can be sealed")
        record.handoff = handoff
        self.transition(handoff.request_id, PDRequestState.SEALED)

    def reserve_decode(
        self,
        request_id: str,
        reservation: DecodeReservation,
    ) -> None:
        record = self.get(request_id)
        if record.handoff is None:
            raise RuntimeError("decode resources cannot be reserved before sealing")
        expected_seq_ids = {item.seq_id for item in record.handoff.sequences}
        if set(reservation.block_tables) != expected_seq_ids:
            raise ValueError("decode reservation sequence IDs do not match handoff")
        record.reservation = reservation
        self.transition(request_id, PDRequestState.DESTINATION_READY)

    def start_transfer(self, request_id: str, transfer_id: str) -> None:
        if not transfer_id.strip():
            raise ValueError("transfer_id must not be empty")
        record = self.get(request_id)
        if record.reservation is None:
            raise RuntimeError("transfer requires a decode reservation")
        record.transfer_id = transfer_id
        self.transition(request_id, PDRequestState.TRANSFERRING)

    def complete_transfer(self, request_id: str, transfer_id: str) -> None:
        record = self.get(request_id)
        if record.transfer_id != transfer_id:
            raise ValueError("transfer completion ID does not match the request")
        self.transition(request_id, PDRequestState.DECODING)

    def finish(self, request_id: str) -> None:
        self.transition(request_id, PDRequestState.FINISHED)

    def fail(self, request_id: str, error: str) -> None:
        self.transition(request_id, PDRequestState.FAILED, error)

    def cancel(self, request_id: str) -> None:
        self.transition(request_id, PDRequestState.CANCELLED)
