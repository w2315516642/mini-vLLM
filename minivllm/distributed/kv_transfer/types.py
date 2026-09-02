"""Serializable descriptions for moving cache memory between workers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple


class TransferStatus(str, Enum):
    """Lifecycle of one backend transfer."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"

    @property
    def is_terminal(self) -> bool:
        return self in {
            TransferStatus.COMPLETED,
            TransferStatus.FAILED,
            TransferStatus.CANCELLED,
            TransferStatus.TIMED_OUT,
        }


@dataclass(frozen=True)
class TransferEndpoint:
    """Address advertised by one transfer-engine process."""

    endpoint_id: str
    hostname: str
    rank: int = 0

    def __post_init__(self) -> None:
        if not self.endpoint_id.strip():
            raise ValueError("endpoint_id must not be empty")
        if not self.hostname.strip():
            raise ValueError("hostname must not be empty")
        if self.rank < 0:
            raise ValueError("rank must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "endpoint_id": self.endpoint_id,
            "hostname": self.hostname,
            "rank": self.rank,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TransferEndpoint":
        return cls(
            endpoint_id=str(value["endpoint_id"]),
            hostname=str(value["hostname"]),
            rank=int(value.get("rank", 0)),
        )


@dataclass(frozen=True)
class RegisteredBuffer:
    """One contiguous tensor region registered with a transfer backend."""

    endpoint: TransferEndpoint
    name: str
    address: int
    nbytes: int
    dtype: str
    shape: Tuple[int, ...]
    device: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("buffer name must not be empty")
        if self.address < 0:
            raise ValueError("buffer address must be non-negative")
        if self.nbytes <= 0:
            raise ValueError("buffer size must be positive")
        if not self.dtype.strip():
            raise ValueError("buffer dtype must not be empty")
        if not self.device.strip():
            raise ValueError("buffer device must not be empty")
        if not self.shape or any(dimension <= 0 for dimension in self.shape):
            raise ValueError("buffer shape must contain positive dimensions")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "endpoint": self.endpoint.to_dict(),
            "name": self.name,
            "address": self.address,
            "nbytes": self.nbytes,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RegisteredBuffer":
        return cls(
            endpoint=TransferEndpoint.from_dict(value["endpoint"]),
            name=str(value["name"]),
            address=int(value["address"]),
            nbytes=int(value["nbytes"]),
            dtype=str(value["dtype"]),
            shape=tuple(int(item) for item in value["shape"]),
            device=str(value["device"]),
        )


@dataclass(frozen=True)
class BufferSlice:
    """A byte range within a registered buffer."""

    buffer: RegisteredBuffer
    offset: int
    length: int

    def __post_init__(self) -> None:
        if self.offset < 0:
            raise ValueError("slice offset must be non-negative")
        if self.length <= 0:
            raise ValueError("slice length must be positive")
        if self.offset + self.length > self.buffer.nbytes:
            raise ValueError(
                f"slice exceeds buffer {self.buffer.name}: "
                f"offset={self.offset}, length={self.length}, "
                f"capacity={self.buffer.nbytes}"
            )

    @property
    def address(self) -> int:
        return self.buffer.address + self.offset

    def to_dict(self) -> Dict[str, Any]:
        return {
            "buffer": self.buffer.to_dict(),
            "offset": self.offset,
            "length": self.length,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BufferSlice":
        return cls(
            buffer=RegisteredBuffer.from_dict(value["buffer"]),
            offset=int(value["offset"]),
            length=int(value["length"]),
        )


@dataclass(frozen=True)
class TransferSlice:
    """One source-to-destination byte copy."""

    source: BufferSlice
    target: BufferSlice

    def __post_init__(self) -> None:
        if self.source.length != self.target.length:
            raise ValueError("source and target slices must have equal length")

    @property
    def length(self) -> int:
        return self.source.length

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TransferSlice":
        return cls(
            source=BufferSlice.from_dict(value["source"]),
            target=BufferSlice.from_dict(value["target"]),
        )


@dataclass(frozen=True)
class TransferPlan:
    """Immutable batch of cache ranges transferred as one request."""

    transfer_id: str
    request_id: str
    source_endpoint: TransferEndpoint
    target_endpoint: TransferEndpoint
    slices: Tuple[TransferSlice, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.transfer_id.strip():
            raise ValueError("transfer_id must not be empty")
        if not self.request_id.strip():
            raise ValueError("request_id must not be empty")
        if self.source_endpoint == self.target_endpoint:
            raise ValueError("source and target endpoints must differ")
        if not self.slices:
            raise ValueError("a transfer plan must contain at least one slice")
        for item in self.slices:
            if item.source.buffer.endpoint != self.source_endpoint:
                raise ValueError("source slice belongs to another endpoint")
            if item.target.buffer.endpoint != self.target_endpoint:
                raise ValueError("target slice belongs to another endpoint")

    @property
    def total_bytes(self) -> int:
        return sum(item.length for item in self.slices)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transfer_id": self.transfer_id,
            "request_id": self.request_id,
            "source_endpoint": self.source_endpoint.to_dict(),
            "target_endpoint": self.target_endpoint.to_dict(),
            "slices": [item.to_dict() for item in self.slices],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TransferPlan":
        return cls(
            transfer_id=str(value["transfer_id"]),
            request_id=str(value["request_id"]),
            source_endpoint=TransferEndpoint.from_dict(
                value["source_endpoint"]
            ),
            target_endpoint=TransferEndpoint.from_dict(
                value["target_endpoint"]
            ),
            slices=tuple(
                TransferSlice.from_dict(item) for item in value["slices"]
            ),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass
class TransferHandle:
    """Mutable backend-owned state for one submitted plan."""

    transfer_id: str
    status: TransferStatus = TransferStatus.PENDING
    native_id: Optional[int] = None
    error: Optional[str] = None
    submitted_at: float = field(default_factory=time.monotonic)
    finished_at: Optional[float] = None

    def transition(
        self,
        status: TransferStatus,
        error: Optional[str] = None,
    ) -> None:
        if self.status.is_terminal:
            raise RuntimeError(
                f"transfer {self.transfer_id} is already {self.status.value}"
            )
        if status == TransferStatus.PENDING:
            raise ValueError("a transfer cannot transition back to pending")
        self.status = status
        self.error = error
        if status.is_terminal:
            self.finished_at = time.monotonic()
