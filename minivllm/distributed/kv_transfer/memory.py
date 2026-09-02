"""Deterministic in-process backend used by tests and local development."""

from __future__ import annotations

from threading import RLock
from typing import Any, Dict, Tuple

import torch

from minivllm.distributed.kv_transfer.backend import TransferBackend
from minivllm.distributed.kv_transfer.types import (
    RegisteredBuffer,
    TransferEndpoint,
    TransferHandle,
    TransferPlan,
    TransferStatus,
)


class InMemoryTransferRegistry:
    """Shared endpoint registry that models remotely registered tensors."""

    def __init__(self) -> None:
        self._buffers: Dict[Tuple[str, str], torch.Tensor] = {}
        self._lock = RLock()

    def register(
        self,
        endpoint: TransferEndpoint,
        name: str,
        tensor: torch.Tensor,
    ) -> None:
        key = (endpoint.endpoint_id, name)
        with self._lock:
            if key in self._buffers:
                raise ValueError(
                    f"buffer {name!r} is already registered at "
                    f"{endpoint.endpoint_id!r}"
                )
            self._buffers[key] = tensor

    def unregister(self, endpoint: TransferEndpoint, name: str) -> None:
        key = (endpoint.endpoint_id, name)
        with self._lock:
            if key not in self._buffers:
                raise ValueError(
                    f"buffer {name!r} is not registered at "
                    f"{endpoint.endpoint_id!r}"
                )
            del self._buffers[key]

    def get(self, descriptor: RegisteredBuffer) -> torch.Tensor:
        key = (descriptor.endpoint.endpoint_id, descriptor.name)
        with self._lock:
            try:
                tensor = self._buffers[key]
            except KeyError as exc:
                raise ValueError(
                    f"buffer {descriptor.name!r} is not registered at "
                    f"{descriptor.endpoint.endpoint_id!r}"
                ) from exc
        if tensor.data_ptr() != descriptor.address:
            raise RuntimeError(
                f"buffer {descriptor.name!r} was replaced after registration"
            )
        return tensor


class InMemoryTransferBackend(TransferBackend):
    """Copy slices with torch operations while preserving the real contract."""

    def __init__(
        self,
        endpoint: TransferEndpoint,
        registry: InMemoryTransferRegistry,
    ) -> None:
        super().__init__(endpoint)
        self._registry = registry
        self._registered: Dict[str, torch.Tensor] = {}
        self._handles: Dict[str, TransferHandle] = {}

    def register_tensor(
        self,
        name: str,
        tensor: Any,
    ) -> RegisteredBuffer:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("the in-memory backend only supports torch tensors")
        if not tensor.is_contiguous():
            raise ValueError("registered tensors must be contiguous")
        if tensor.numel() == 0:
            raise ValueError("registered tensors must not be empty")
        if name in self._registered:
            raise ValueError(f"buffer {name!r} is already registered")
        self._registry.register(self.endpoint, name, tensor)
        self._registered[name] = tensor
        return RegisteredBuffer(
            endpoint=self.endpoint,
            name=name,
            address=tensor.data_ptr(),
            nbytes=tensor.numel() * tensor.element_size(),
            dtype=str(tensor.dtype),
            shape=tuple(tensor.shape),
            device=str(tensor.device),
        )

    def unregister_buffer(self, name: str) -> None:
        if name not in self._registered:
            raise ValueError(f"buffer {name!r} is not registered")
        self._registry.unregister(self.endpoint, name)
        del self._registered[name]

    def submit(self, plan: TransferPlan) -> TransferHandle:
        if plan.source_endpoint != self.endpoint:
            raise ValueError("this backend does not own the plan source")
        if plan.transfer_id in self._handles:
            raise ValueError(f"duplicate transfer id: {plan.transfer_id}")
        handle = TransferHandle(plan.transfer_id)
        self._handles[plan.transfer_id] = handle
        handle.transition(TransferStatus.RUNNING)
        try:
            for item in plan.slices:
                source = self._registry.get(item.source.buffer)
                target = self._registry.get(item.target.buffer)
                source_bytes = source.view(torch.uint8).reshape(-1)
                target_bytes = target.view(torch.uint8).reshape(-1)
                target_bytes.narrow(
                    0, item.target.offset, item.length
                ).copy_(
                    source_bytes.narrow(
                        0, item.source.offset, item.length
                    )
                )
            handle.transition(TransferStatus.COMPLETED)
        except Exception as exc:
            handle.transition(TransferStatus.FAILED, str(exc))
        return handle

    def poll(self, handle: TransferHandle) -> TransferStatus:
        registered = self._handles.get(handle.transfer_id)
        if registered is not handle:
            raise ValueError("transfer handle does not belong to this backend")
        return handle.status

    def close(self) -> None:
        for name in tuple(self._registered):
            self.unregister_buffer(name)

