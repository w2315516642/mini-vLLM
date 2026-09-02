"""Cache-transfer contracts used by prefill/decode disaggregation."""

from minivllm.distributed.kv_transfer.backend import TransferBackend
from minivllm.distributed.kv_transfer.memory import (
    InMemoryTransferBackend,
    InMemoryTransferRegistry,
)
from minivllm.distributed.kv_transfer.layout import (
    CacheLayout,
    CacheRegion,
    CacheRegionKind,
    KVTransferPlanner,
    register_cache_layout,
)
from minivllm.distributed.kv_transfer.p2p import P2PTransferBackend
from minivllm.distributed.kv_transfer.manager import (
    ManagedTransfer,
    TransferManager,
    TransferMetrics,
    TransferResourceLease,
)
from minivllm.distributed.kv_transfer.types import (
    BufferSlice,
    RegisteredBuffer,
    TransferEndpoint,
    TransferHandle,
    TransferPlan,
    TransferSlice,
    TransferStatus,
)

__all__ = [
    "BufferSlice",
    "CacheLayout",
    "CacheRegion",
    "CacheRegionKind",
    "InMemoryTransferBackend",
    "InMemoryTransferRegistry",
    "RegisteredBuffer",
    "KVTransferPlanner",
    "ManagedTransfer",
    "P2PTransferBackend",
    "TransferBackend",
    "TransferEndpoint",
    "TransferHandle",
    "TransferManager",
    "TransferMetrics",
    "TransferPlan",
    "TransferSlice",
    "TransferStatus",
    "TransferResourceLease",
    "register_cache_layout",
]
