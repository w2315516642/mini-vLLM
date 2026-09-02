"""Configuration for prefill/decode disaggregation."""

from dataclasses import dataclass
from enum import Enum

from minivllm.distributed.kv_transfer.types import TransferEndpoint


class PDRole(str, Enum):
    UNIFIED = "unified"
    PREFILL = "prefill"
    DECODE = "decode"


class KVTransferBackend(str, Enum):
    NONE = "none"
    MEMORY = "memory"
    TCP = "tcp"


@dataclass(frozen=True)
class PDConfig:
    """Role and transport settings shared by engine and worker processes."""

    role: PDRole = PDRole.UNIFIED
    backend: KVTransferBackend = KVTransferBackend.NONE
    endpoint_id: str = ""
    hostname: str = ""
    peer_endpoint_id: str = ""
    peer_hostname: str = ""
    metadata_server: str = "P2PHANDSHAKE"
    protocol: str = "tcp"
    device_name: str = ""
    transfer_timeout_s: float = 30.0
    max_inflight_transfers: int = 8

    def __post_init__(self) -> None:
        role = PDRole(self.role)
        backend = KVTransferBackend(self.backend)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "backend", backend)
        if self.transfer_timeout_s <= 0:
            raise ValueError("transfer_timeout_s must be positive")
        if self.max_inflight_transfers <= 0:
            raise ValueError("max_inflight_transfers must be positive")
        if role == PDRole.UNIFIED:
            if backend != KVTransferBackend.NONE:
                raise ValueError("unified role requires the none backend")
            return
        if backend == KVTransferBackend.NONE:
            raise ValueError("prefill/decode roles require a transfer backend")
        for name, value in (
            ("endpoint_id", self.endpoint_id),
            ("hostname", self.hostname),
            ("peer_endpoint_id", self.peer_endpoint_id),
            ("peer_hostname", self.peer_hostname),
        ):
            if not value.strip():
                raise ValueError(f"{name} must not be empty in PD mode")
        if self.endpoint_id == self.peer_endpoint_id:
            raise ValueError("local and peer endpoint IDs must differ")
        if not self.protocol.strip():
            raise ValueError("protocol must not be empty")

    @property
    def enabled(self) -> bool:
        return self.role != PDRole.UNIFIED

    def endpoint_for_rank(
        self,
        rank: int,
        peer: bool = False,
    ) -> TransferEndpoint:
        """Derive one endpoint per TP rank from the configured base port."""
        if not self.enabled:
            raise ValueError("unified mode has no transfer endpoint")
        if rank < 0:
            raise ValueError("rank must be non-negative")
        endpoint_id = self.peer_endpoint_id if peer else self.endpoint_id
        hostname = self.peer_hostname if peer else self.hostname
        host, port_text = hostname.rsplit(":", 1)
        try:
            port = int(port_text) + rank
        except ValueError as exc:
            raise ValueError("PD hostname must use host:port form") from exc
        if not host or not 0 < port < 65536:
            raise ValueError("rank-adjusted PD endpoint is invalid")
        return TransferEndpoint(
            endpoint_id=f"{endpoint_id}/rank-{rank}",
            hostname=f"{host}:{port}",
            rank=rank,
        )
