"""Backend interface for cache memory transfer."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from minivllm.distributed.kv_transfer.types import (
    RegisteredBuffer,
    TransferEndpoint,
    TransferHandle,
    TransferPlan,
    TransferStatus,
)


class TransferBackend(ABC):
    """Register local tensors and push byte ranges to a peer."""

    def __init__(self, endpoint: TransferEndpoint) -> None:
        self.endpoint = endpoint

    @abstractmethod
    def register_tensor(self, name: str, tensor: Any) -> RegisteredBuffer:
        """Register one contiguous tensor and return its public descriptor."""

    @abstractmethod
    def unregister_buffer(self, name: str) -> None:
        """Unregister a previously registered local tensor."""

    @abstractmethod
    def submit(self, plan: TransferPlan) -> TransferHandle:
        """Start a P-push transfer from this endpoint."""

    @abstractmethod
    def poll(self, handle: TransferHandle) -> TransferStatus:
        """Refresh and return the handle status."""

    def wait(
        self,
        handle: TransferHandle,
        timeout_s: float,
        poll_interval_s: float = 0.001,
    ) -> TransferStatus:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        deadline = time.monotonic() + timeout_s
        while True:
            status = self.poll(handle)
            if status.is_terminal:
                return status
            if time.monotonic() >= deadline:
                handle.transition(
                    TransferStatus.TIMED_OUT,
                    f"transfer exceeded {timeout_s:.3f}s",
                )
                return handle.status
            time.sleep(poll_interval_s)

    def abort(self, handle: TransferHandle) -> None:
        if not handle.status.is_terminal:
            handle.transition(TransferStatus.CANCELLED)

    @abstractmethod
    def close(self) -> None:
        """Release backend resources."""

