"""Backpressure, metrics, and resource lifetime for asynchronous transfers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Callable, Dict, List, Optional

from minivllm.distributed.kv_transfer.backend import TransferBackend
from minivllm.distributed.kv_transfer.types import (
    TransferHandle,
    TransferPlan,
    TransferStatus,
)


class TransferResourceLease:
    """Keep scheduler-owned blocks pinned until backend I/O is terminal."""

    def __init__(self, release: Callable[[], None]) -> None:
        self._release = release
        self.released = False

    def release(self) -> None:
        if self.released:
            return
        self._release()
        self.released = True


@dataclass
class ManagedTransfer:
    plan: TransferPlan
    handle: TransferHandle
    lease: TransferResourceLease
    deadline: float
    cancellation_requested: bool = False
    timeout_requested: bool = False
    finalized: bool = False


@dataclass
class TransferMetrics:
    submitted: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    timed_out: int = 0
    bytes_submitted: int = 0
    bytes_completed: int = 0
    total_latency_s: float = 0.0


class TransferManager:
    """Own handles without taking ownership of Scheduler or Worker threads."""

    def __init__(
        self,
        backend: TransferBackend,
        max_inflight: int,
        timeout_s: float,
    ) -> None:
        if max_inflight <= 0:
            raise ValueError("max_inflight must be positive")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.backend = backend
        self.max_inflight = max_inflight
        self.timeout_s = timeout_s
        self.metrics = TransferMetrics()
        self._tasks: Dict[str, ManagedTransfer] = {}
        self._lock = RLock()

    @property
    def num_inflight(self) -> int:
        return sum(not task.finalized for task in self._tasks.values())

    def submit(
        self,
        plan: TransferPlan,
        lease: Optional[TransferResourceLease] = None,
    ) -> ManagedTransfer:
        with self._lock:
            existing = self._tasks.get(plan.transfer_id)
            if existing is not None:
                if existing.plan != plan:
                    raise ValueError(
                        f"transfer id {plan.transfer_id!r} was reused "
                        "for another plan"
                    )
                return existing
            if self.num_inflight >= self.max_inflight:
                raise RuntimeError(
                    f"transfer backpressure limit reached ({self.max_inflight})"
                )
            lease = lease or TransferResourceLease(lambda: None)
            try:
                handle = self.backend.submit(plan)
            except Exception:
                lease.release()
                raise
            task = ManagedTransfer(
                plan=plan,
                handle=handle,
                lease=lease,
                deadline=time.monotonic() + self.timeout_s,
            )
            self._tasks[plan.transfer_id] = task
            self.metrics.submitted += 1
            self.metrics.bytes_submitted += plan.total_bytes
            self._refresh(task)
            return task

    def _refresh(self, task: ManagedTransfer) -> TransferStatus:
        if task.finalized:
            return task.handle.status
        status = self.backend.poll(task.handle)
        if not status.is_terminal and time.monotonic() >= task.deadline:
            task.timeout_requested = True
            self.backend.abort(task.handle)
            status = self.backend.poll(task.handle)
        if status.is_terminal:
            self._finalize(task)
        return status

    def _finalize(self, task: ManagedTransfer) -> None:
        if task.finalized:
            return
        task.finalized = True
        task.lease.release()
        status = task.handle.status
        if task.timeout_requested:
            self.metrics.timed_out += 1
        elif task.cancellation_requested:
            self.metrics.cancelled += 1
        elif status == TransferStatus.COMPLETED:
            self.metrics.completed += 1
            self.metrics.bytes_completed += task.plan.total_bytes
        else:
            self.metrics.failed += 1
        if task.handle.finished_at is not None:
            self.metrics.total_latency_s += (
                task.handle.finished_at - task.handle.submitted_at
            )

    def poll(self) -> List[ManagedTransfer]:
        """Refresh all active tasks and return newly terminal tasks."""
        completed = []
        with self._lock:
            for task in self._tasks.values():
                was_finalized = task.finalized
                self._refresh(task)
                if not was_finalized and task.finalized:
                    completed.append(task)
        return completed

    def wait(
        self,
        transfer_id: str,
        poll_interval_s: float = 0.001,
    ) -> TransferStatus:
        while True:
            with self._lock:
                try:
                    task = self._tasks[transfer_id]
                except KeyError as exc:
                    raise ValueError(
                        f"unknown transfer id: {transfer_id}"
                    ) from exc
                status = self._refresh(task)
                if status.is_terminal:
                    return status
            time.sleep(poll_interval_s)

    def cancel(self, transfer_id: str) -> TransferStatus:
        with self._lock:
            try:
                task = self._tasks[transfer_id]
            except KeyError as exc:
                raise ValueError(
                    f"unknown transfer id: {transfer_id}"
                ) from exc
            if task.finalized:
                return task.handle.status
            task.cancellation_requested = True
            self.backend.abort(task.handle)
            return self._refresh(task)
