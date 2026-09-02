"""Translate paged KV and recurrent-state ownership into byte transfers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from minivllm.distributed.kv_transfer.backend import TransferBackend
from minivllm.distributed.kv_transfer.types import (
    BufferSlice,
    RegisteredBuffer,
    TransferPlan,
    TransferSlice,
)


class CacheRegionKind(str, Enum):
    KEY = "key"
    VALUE = "value"
    DRAFT_KEY = "draft_key"
    DRAFT_VALUE = "draft_value"
    CONV_STATE = "conv_state"
    RECURRENT_STATE = "recurrent_state"


@dataclass(frozen=True)
class CacheRegion:
    """A registered tensor split into fixed-size blocks or request slots."""

    layer_idx: int
    kind: CacheRegionKind
    buffer: RegisteredBuffer
    unit_bytes: int
    num_units: int

    def __post_init__(self) -> None:
        if self.layer_idx < 0:
            raise ValueError("layer_idx must be non-negative")
        if self.unit_bytes <= 0 or self.num_units <= 0:
            raise ValueError("cache regions require positive unit sizes")
        if self.unit_bytes * self.num_units != self.buffer.nbytes:
            raise ValueError(
                f"region {self.layer_idx}/{self.kind.value} does not evenly "
                "cover its registered buffer"
            )

    def slice(self, unit_id: int) -> BufferSlice:
        if unit_id < 0 or unit_id >= self.num_units:
            raise ValueError(
                f"unit {unit_id} is outside region capacity {self.num_units}"
            )
        return BufferSlice(
            self.buffer,
            offset=unit_id * self.unit_bytes,
            length=self.unit_bytes,
        )


@dataclass(frozen=True)
class CacheLayout:
    """Rank-local cache regions advertised to the peer role."""

    block_size: int
    regions: Tuple[CacheRegion, ...]

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if not self.regions:
            raise ValueError("cache layout must contain at least one region")
        endpoints = {region.buffer.endpoint for region in self.regions}
        if len(endpoints) != 1:
            raise ValueError("all cache regions must belong to one endpoint")
        keys = [(region.layer_idx, region.kind) for region in self.regions]
        if len(set(keys)) != len(keys):
            raise ValueError("cache layout contains duplicate regions")

    @property
    def endpoint(self):
        return self.regions[0].buffer.endpoint

    def get(self, layer_idx: int, kind: CacheRegionKind) -> CacheRegion:
        for region in self.regions:
            if region.layer_idx == layer_idx and region.kind == kind:
                return region
        raise KeyError(f"cache region {layer_idx}/{kind.value} is absent")

    def region_keys(self) -> Tuple[Tuple[int, CacheRegionKind], ...]:
        return tuple((region.layer_idx, region.kind) for region in self.regions)


def register_cache_layout(
    backend: TransferBackend,
    block_size: int,
    full_attention_caches: Mapping[int, Tuple[Any, Any]],
    linear_state_pools: Optional[Mapping[int, Any]] = None,
    draft_attention_caches: Optional[Mapping[int, Tuple[Any, Any]]] = None,
) -> CacheLayout:
    """Register layer-aligned tensors without depending on Worker internals."""

    regions = []
    for layer_idx, (key_cache, value_cache) in sorted(
        full_attention_caches.items()
    ):
        for kind, tensor in (
            (CacheRegionKind.KEY, key_cache),
            (CacheRegionKind.VALUE, value_cache),
        ):
            if tensor.ndim < 2:
                raise ValueError("paged cache tensors require a block dimension")
            descriptor = backend.register_tensor(
                f"layer-{layer_idx}.{kind.value}", tensor
            )
            num_units = int(tensor.shape[0])
            regions.append(
                CacheRegion(
                    layer_idx=layer_idx,
                    kind=kind,
                    buffer=descriptor,
                    unit_bytes=descriptor.nbytes // num_units,
                    num_units=num_units,
                )
            )

    # Draft K/V mirrors target physical block IDs. Workspace blocks can remain
    # registered because plans only select scheduler-owned target block IDs.
    for layer_idx, (key_cache, value_cache) in sorted(
        (draft_attention_caches or {}).items()
    ):
        for kind, tensor in (
            (CacheRegionKind.DRAFT_KEY, key_cache),
            (CacheRegionKind.DRAFT_VALUE, value_cache),
        ):
            if tensor.ndim < 2:
                raise ValueError("paged draft cache tensors require a block dimension")
            descriptor = backend.register_tensor(
                f"draft-layer-{layer_idx}.{kind.value}", tensor
            )
            num_units = int(tensor.shape[0])
            regions.append(
                CacheRegion(
                    layer_idx=layer_idx,
                    kind=kind,
                    buffer=descriptor,
                    unit_bytes=descriptor.nbytes // num_units,
                    num_units=num_units,
                )
            )

    for layer_idx, state in sorted((linear_state_pools or {}).items()):
        for kind, tensor in (
            (CacheRegionKind.CONV_STATE, state.conv_state),
            (CacheRegionKind.RECURRENT_STATE, state.recurrent_state),
        ):
            if tensor.ndim < 2:
                raise ValueError("state pools require a request-slot dimension")
            descriptor = backend.register_tensor(
                f"layer-{layer_idx}.{kind.value}", tensor
            )
            num_units = int(tensor.shape[0])
            regions.append(
                CacheRegion(
                    layer_idx=layer_idx,
                    kind=kind,
                    buffer=descriptor,
                    unit_bytes=descriptor.nbytes // num_units,
                    num_units=num_units,
                )
            )

    return CacheLayout(block_size=block_size, regions=tuple(regions))


class KVTransferPlanner:
    """Build P-push plans after D has reserved destination blocks and slots."""

    @staticmethod
    def build_plan(
        transfer_id: str,
        request_id: str,
        source: CacheLayout,
        target: CacheLayout,
        source_block_ids: Sequence[int],
        target_block_ids: Sequence[int],
        num_tokens: int,
        source_state_slot: Optional[int] = None,
        target_state_slot: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> TransferPlan:
        if source.block_size != target.block_size:
            raise ValueError("source and target block sizes differ")
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        num_blocks = (num_tokens + source.block_size - 1) // source.block_size
        if len(source_block_ids) != num_blocks:
            raise ValueError(
                f"source requires {num_blocks} blocks for {num_tokens} tokens"
            )
        if len(target_block_ids) != num_blocks:
            raise ValueError(
                f"target requires {num_blocks} blocks for {num_tokens} tokens"
            )
        if len(set(source_block_ids)) != len(source_block_ids):
            raise ValueError("source block IDs must not contain duplicates")
        if len(set(target_block_ids)) != len(target_block_ids):
            raise ValueError("target block IDs must not contain duplicates")
        if source.region_keys() != target.region_keys():
            raise ValueError("source and target cache layouts differ")

        has_state = any(
            kind in (
                CacheRegionKind.CONV_STATE,
                CacheRegionKind.RECURRENT_STATE,
            )
            for _, kind in source.region_keys()
        )
        if has_state != (
            source_state_slot is not None and target_state_slot is not None
        ):
            raise ValueError(
                "hybrid layouts require both source and target state slots"
            )
        if not has_state and (
            source_state_slot is not None or target_state_slot is not None
        ):
            raise ValueError("state slots were supplied for a KV-only layout")

        slices = []
        for source_region in source.regions:
            target_region = target.get(
                source_region.layer_idx, source_region.kind
            )
            if (
                source_region.unit_bytes != target_region.unit_bytes
                or source_region.buffer.dtype != target_region.buffer.dtype
                or source_region.buffer.shape[1:] != target_region.buffer.shape[1:]
            ):
                raise ValueError(
                    f"region {source_region.layer_idx}/"
                    f"{source_region.kind.value} has incompatible layouts"
                )
            if source_region.kind in (
                CacheRegionKind.KEY,
                CacheRegionKind.VALUE,
                CacheRegionKind.DRAFT_KEY,
                CacheRegionKind.DRAFT_VALUE,
            ):
                unit_pairs = zip(source_block_ids, target_block_ids)
            else:
                unit_pairs = ((source_state_slot, target_state_slot),)
            for source_unit, target_unit in unit_pairs:
                assert source_unit is not None and target_unit is not None
                slices.append(
                    TransferSlice(
                        source_region.slice(source_unit),
                        target_region.slice(target_unit),
                    )
                )

        plan_metadata: Dict[str, Any] = dict(metadata or {})
        plan_metadata.update(
            {
                "num_tokens": num_tokens,
                "block_size": source.block_size,
                "source_block_ids": list(source_block_ids),
                "target_block_ids": list(target_block_ids),
            }
        )
        if has_state:
            plan_metadata.update(
                {
                    "source_state_slot": source_state_slot,
                    "target_state_slot": target_state_slot,
                }
            )
        return TransferPlan(
            transfer_id=transfer_id,
            request_id=request_id,
            source_endpoint=source.endpoint,
            target_endpoint=target.endpoint,
            slices=tuple(slices),
            metadata=plan_metadata,
        )
