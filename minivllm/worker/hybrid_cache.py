"""Layer-aligned cache storage for Qwen hybrid decoders.

Full-attention layers keep using the existing paged KV cache. Linear-attention
layers own fixed-size convolution and recurrent-state pools indexed by a stable
sequence slot. Keeping the two cache kinds behind one layer-aligned object
lets the model follow its original layer order without conflating their
lifecycle rules.
"""

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import torch

from minivllm.configs.model_architecture import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    SUPPORTED_LAYER_TYPES,
)
from minivllm.model_executor.layers.gated_delta_net import GatedDeltaNetState


KVCache = Tuple[torch.Tensor, torch.Tensor]


def _positive_config_int(config, field_name: str) -> int:
    value = getattr(config, field_name, None)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _validate_seq_id(seq_id: int) -> None:
    if not isinstance(seq_id, int) or isinstance(seq_id, bool) or seq_id < 0:
        raise ValueError("seq_id must be a non-negative integer")


@dataclass(frozen=True)
class GatedDeltaNetStateSpec:
    """Per-request state dimensions shared by all Qwen linear layers."""

    num_key_heads: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_kernel_size: int

    @classmethod
    def from_text_config(
        cls,
        text_config,
        tensor_parallel_size: int = 1,
    ) -> "GatedDeltaNetStateSpec":
        if (
            not isinstance(tensor_parallel_size, int)
            or isinstance(tensor_parallel_size, bool)
            or tensor_parallel_size <= 0
        ):
            raise ValueError("tensor_parallel_size must be a positive integer")
        total_num_key_heads = _positive_config_int(
            text_config, "linear_num_key_heads"
        )
        total_num_value_heads = _positive_config_int(
            text_config, "linear_num_value_heads"
        )
        if total_num_key_heads % tensor_parallel_size != 0:
            raise ValueError(
                "linear_num_key_heads must be divisible by "
                "tensor_parallel_size"
            )
        if total_num_value_heads % tensor_parallel_size != 0:
            raise ValueError(
                "linear_num_value_heads must be divisible by "
                "tensor_parallel_size"
            )
        num_key_heads = total_num_key_heads // tensor_parallel_size
        num_value_heads = total_num_value_heads // tensor_parallel_size
        if num_value_heads % num_key_heads != 0:
            raise ValueError(
                "linear_num_value_heads must be divisible by "
                "linear_num_key_heads"
            )
        return cls(
            num_key_heads=num_key_heads,
            num_value_heads=num_value_heads,
            key_head_dim=_positive_config_int(
                text_config, "linear_key_head_dim"
            ),
            value_head_dim=_positive_config_int(
                text_config, "linear_value_head_dim"
            ),
            conv_kernel_size=_positive_config_int(
                text_config, "linear_conv_kernel_dim"
            ),
        )

    @property
    def conv_dim(self) -> int:
        key_dim = self.num_key_heads * self.key_head_dim
        value_dim = self.num_value_heads * self.value_head_dim
        return 2 * key_dim + value_dim

    @property
    def conv_state_shape(self) -> Tuple[int, int]:
        return self.conv_dim, self.conv_kernel_size

    @property
    def recurrent_state_shape(self) -> Tuple[int, int, int]:
        return (
            self.num_value_heads,
            self.key_head_dim,
            self.value_head_dim,
        )


@dataclass
class HybridStateSnapshot:
    """Temporary recurrent-state checkpoint used for MTP rejection rollback."""

    seq_ids: Tuple[int, ...]
    layer_states: Dict[int, GatedDeltaNetState]


class RequestStateSlotAllocator:
    """Map active sequence IDs to a bounded set of stable state slots."""

    def __init__(self, capacity: int) -> None:
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        # Reverse order lets pop() allocate 0, 1, ... while remaining O(1).
        self._free_slots = list(range(capacity - 1, -1, -1))
        self._seq_to_slot: Dict[int, int] = {}

    @property
    def num_active_slots(self) -> int:
        return len(self._seq_to_slot)

    @property
    def num_free_slots(self) -> int:
        return len(self._free_slots)

    def acquire(self, seq_id: int) -> Tuple[int, bool]:
        """Return ``(slot, is_new)`` while preserving existing ownership."""
        _validate_seq_id(seq_id)
        existing = self._seq_to_slot.get(seq_id)
        if existing is not None:
            return existing, False
        if not self._free_slots:
            raise ValueError(
                f"no free state slots remain (capacity={self.capacity})"
            )
        slot = self._free_slots.pop()
        self._seq_to_slot[seq_id] = slot
        return slot, True

    def lookup(self, seq_id: int) -> int:
        """Return the slot owned by an active sequence."""
        _validate_seq_id(seq_id)
        try:
            return self._seq_to_slot[seq_id]
        except KeyError as exc:
            raise ValueError(f"sequence {seq_id} is not active") from exc

    def contains(self, seq_id: int) -> bool:
        _validate_seq_id(seq_id)
        return seq_id in self._seq_to_slot

    def release(self, seq_id: int) -> int:
        """Release one active sequence and return its former slot."""
        _validate_seq_id(seq_id)
        try:
            slot = self._seq_to_slot.pop(seq_id)
        except KeyError as exc:
            raise ValueError(f"sequence {seq_id} is not active") from exc
        self._free_slots.append(slot)
        return slot

    def reset(self) -> None:
        """Release every sequence and restore deterministic slot order."""
        self._seq_to_slot.clear()
        self._free_slots = list(range(self.capacity - 1, -1, -1))


class HybridCache:
    """Coordinate paged KV caches and per-request GDN state pools.

    Global layer indices are retained. ``full_attention_caches`` therefore has
    one entry for every full-attention layer and no entry for a linear layer.
    All linear layers share request-slot ownership, while each layer owns its
    own state tensors.
    """

    def __init__(
        self,
        layer_types: Sequence[str],
        full_attention_caches: Mapping[int, KVCache],
        state_spec: GatedDeltaNetStateSpec,
        max_num_seqs: int,
        device: torch.device | str = "cuda",
    ) -> None:
        self.layer_types = tuple(layer_types)
        if not self.layer_types:
            raise ValueError("layer_types must not be empty")
        for layer_type in self.layer_types:
            if layer_type not in SUPPORTED_LAYER_TYPES:
                raise ValueError(f"unsupported layer type: {layer_type}")

        expected_full_layers = {
            layer_idx
            for layer_idx, layer_type in enumerate(self.layer_types)
            if layer_type == FULL_ATTENTION
        }
        actual_full_layers = set(full_attention_caches)
        if actual_full_layers != expected_full_layers:
            raise ValueError(
                "full_attention_caches keys must match full-attention layer "
                f"indices: expected {sorted(expected_full_layers)}, "
                f"got {sorted(actual_full_layers)}"
            )
        for layer_idx, kv_cache in full_attention_caches.items():
            if (
                not isinstance(kv_cache, tuple)
                or len(kv_cache) != 2
                or not all(isinstance(tensor, torch.Tensor) for tensor in kv_cache)
            ):
                raise TypeError(
                    f"full-attention layer {layer_idx} cache must be a "
                    "(key_tensor, value_tensor) tuple"
                )

        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.state_spec = state_spec
        self._slots = RequestStateSlotAllocator(max_num_seqs)
        self._full_attention_caches = dict(full_attention_caches)
        self._linear_state_pools: Dict[int, GatedDeltaNetState] = {}

        for layer_idx, layer_type in enumerate(self.layer_types):
            if layer_type != LINEAR_ATTENTION:
                continue
            conv_state = torch.zeros(
                max_num_seqs,
                *state_spec.conv_state_shape,
                dtype=torch.float32,
                device=self.device,
            )
            recurrent_state = torch.zeros(
                max_num_seqs,
                *state_spec.recurrent_state_shape,
                dtype=torch.float32,
                device=self.device,
            )
            self._linear_state_pools[layer_idx] = GatedDeltaNetState(
                conv_state=conv_state,
                recurrent_state=recurrent_state,
            )

    @property
    def num_active_slots(self) -> int:
        return self._slots.num_active_slots

    @property
    def num_free_slots(self) -> int:
        return self._slots.num_free_slots

    def _normalize_seq_ids(self, seq_ids: Sequence[int]) -> Tuple[int, ...]:
        normalized = tuple(seq_ids)
        for seq_id in normalized:
            _validate_seq_id(seq_id)
        if len(set(normalized)) != len(normalized):
            raise ValueError("seq_ids must not contain duplicates")
        return normalized

    def _normalize_slot_ids(
        self,
        slot_ids: torch.Tensor | Sequence[int],
    ) -> torch.Tensor:
        if isinstance(slot_ids, torch.Tensor):
            if slot_ids.ndim != 1:
                raise ValueError("slot_ids must be a one-dimensional tensor")
            if slot_ids.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            ):
                raise ValueError("slot_ids must use an integer dtype")
            slots = slot_ids.to(device=self.device, dtype=torch.long)
        else:
            slots = torch.as_tensor(
                tuple(slot_ids), dtype=torch.long, device=self.device
            )

        if slots.numel() == 0:
            return slots
        if torch.any(slots < 0) or torch.any(slots >= self._slots.capacity):
            raise ValueError("slot_ids contain an out-of-range state slot")
        if torch.unique(slots).numel() != slots.numel():
            raise ValueError("slot_ids must not contain duplicates")
        return slots

    def _require_layer_type(self, layer_idx: int, expected: str) -> None:
        if not isinstance(layer_idx, int) or isinstance(layer_idx, bool):
            raise TypeError("layer_idx must be an integer")
        if layer_idx < 0 or layer_idx >= len(self.layer_types):
            raise IndexError(f"layer_idx {layer_idx} is out of range")
        actual = self.layer_types[layer_idx]
        if actual != expected:
            raise ValueError(
                f"layer {layer_idx} is {actual}, expected {expected}"
            )

    def acquire(self, seq_ids: Sequence[int]) -> torch.Tensor:
        """Acquire stable slots in the same order as the execution batch."""
        normalized = self._normalize_seq_ids(seq_ids)
        slots = []
        new_slots = []
        for seq_id in normalized:
            slot, is_new = self._slots.acquire(seq_id)
            slots.append(slot)
            if is_new:
                new_slots.append(slot)

        # A released slot is already cleared. Explicit initialization here also
        # protects callers that acquire after a partial lifecycle failure.
        if new_slots:
            new_slot_ids = torch.tensor(
                new_slots, dtype=torch.long, device=self.device
            )
            for state in self._linear_state_pools.values():
                state.conv_state.index_fill_(0, new_slot_ids, 0.0)
                state.recurrent_state.index_fill_(0, new_slot_ids, 0.0)
        return torch.tensor(slots, dtype=torch.long, device=self.device)

    def get_kv_cache(self, layer_idx: int) -> KVCache:
        """Return the paged KV cache owned by one full-attention layer."""
        self._require_layer_type(layer_idx, FULL_ATTENTION)
        return self._full_attention_caches[layer_idx]

    def get_state_slot(self, seq_id: int) -> int:
        """Return the stable request slot advertised during PD handoff."""
        return self._slots.lookup(seq_id)

    def get_state_pools(self) -> Mapping[int, GatedDeltaNetState]:
        """Expose persistent pools for one-time transfer-engine registration."""
        return self._linear_state_pools.copy()

    def read_state(
        self,
        layer_idx: int,
        slot_ids: torch.Tensor | Sequence[int],
    ) -> GatedDeltaNetState:
        """Gather a contiguous GDN state batch for one linear layer."""
        self._require_layer_type(layer_idx, LINEAR_ATTENTION)
        slots = self._normalize_slot_ids(slot_ids)
        pool = self._linear_state_pools[layer_idx]
        return GatedDeltaNetState(
            conv_state=pool.conv_state.index_select(0, slots),
            recurrent_state=pool.recurrent_state.index_select(0, slots),
        )

    def write_state(
        self,
        layer_idx: int,
        slot_ids: torch.Tensor | Sequence[int],
        state: GatedDeltaNetState,
    ) -> None:
        """Scatter a kernel-updated GDN state batch back to its slots."""
        self._require_layer_type(layer_idx, LINEAR_ATTENTION)
        slots = self._normalize_slot_ids(slot_ids)
        if not isinstance(state, GatedDeltaNetState):
            raise TypeError("state must be a GatedDeltaNetState")

        batch_size = slots.numel()
        expected_conv_shape = (batch_size, *self.state_spec.conv_state_shape)
        expected_recurrent_shape = (
            batch_size,
            *self.state_spec.recurrent_state_shape,
        )
        if tuple(state.conv_state.shape) != expected_conv_shape:
            raise ValueError(
                "conv_state has shape "
                f"{tuple(state.conv_state.shape)}, expected {expected_conv_shape}"
            )
        if tuple(state.recurrent_state.shape) != expected_recurrent_shape:
            raise ValueError(
                "recurrent_state has shape "
                f"{tuple(state.recurrent_state.shape)}, expected "
                f"{expected_recurrent_shape}"
            )
        for name, tensor in (
            ("conv_state", state.conv_state),
            ("recurrent_state", state.recurrent_state),
        ):
            if tensor.dtype != torch.float32:
                raise ValueError(f"{name} must use float32 state storage")
            if tensor.device != self.device:
                raise ValueError(
                    f"{name} is on {tensor.device}, expected {self.device}"
                )

        pool = self._linear_state_pools[layer_idx]
        pool.conv_state.index_copy_(0, slots, state.conv_state)
        pool.recurrent_state.index_copy_(0, slots, state.recurrent_state)

    def fork(self, parent_seq_id: int, child_seq_id: int) -> int:
        """Allocate a child slot and copy every linear-layer state into it."""
        parent_slot = self._slots.lookup(parent_seq_id)
        _validate_seq_id(child_seq_id)
        if child_seq_id == parent_seq_id:
            raise ValueError("child sequence must differ from parent sequence")
        try:
            self._slots.lookup(child_seq_id)
        except ValueError:
            pass
        else:
            raise ValueError(f"child sequence {child_seq_id} is already active")

        child_slot, _ = self._slots.acquire(child_seq_id)
        for state in self._linear_state_pools.values():
            state.conv_state[child_slot].copy_(state.conv_state[parent_slot])
            state.recurrent_state[child_slot].copy_(
                state.recurrent_state[parent_slot]
            )
        return child_slot

    def copy(self, parent_seq_id: int, child_seq_id: int) -> None:
        """Copy a parent's state, allocating the child slot when necessary."""
        parent_slot = self._slots.lookup(parent_seq_id)
        child_slot, _ = self._slots.acquire(child_seq_id)
        if parent_slot == child_slot:
            return
        for state in self._linear_state_pools.values():
            state.conv_state[child_slot].copy_(state.conv_state[parent_slot])
            state.recurrent_state[child_slot].copy_(
                state.recurrent_state[parent_slot]
            )

    def release(self, seq_ids: Sequence[int]) -> None:
        """Release requests and clear their state before slot reuse."""
        normalized = self._normalize_seq_ids(seq_ids)
        slots = [self._slots.release(seq_id) for seq_id in normalized]
        if not slots:
            return
        slot_ids = torch.tensor(slots, dtype=torch.long, device=self.device)
        for state in self._linear_state_pools.values():
            state.conv_state.index_fill_(0, slot_ids, 0.0)
            state.recurrent_state.index_fill_(0, slot_ids, 0.0)

    def release_existing(self, seq_ids: Sequence[int]) -> None:
        """Release active IDs while tolerating requests never run by a worker."""
        active_seq_ids = [
            seq_id for seq_id in self._normalize_seq_ids(seq_ids)
            if self._slots.contains(seq_id)
        ]
        self.release(active_seq_ids)

    def snapshot(self, seq_ids: Sequence[int]) -> HybridStateSnapshot:
        """Clone selected recurrent states before speculative verification."""
        normalized = self._normalize_seq_ids(seq_ids)
        slots = torch.tensor(
            [self._slots.lookup(seq_id) for seq_id in normalized],
            dtype=torch.long,
            device=self.device,
        )
        layer_states = {}
        for layer_idx, pool in self._linear_state_pools.items():
            layer_states[layer_idx] = GatedDeltaNetState(
                conv_state=pool.conv_state.index_select(0, slots).clone(),
                recurrent_state=(
                    pool.recurrent_state.index_select(0, slots).clone()
                ),
            )
        return HybridStateSnapshot(normalized, layer_states)

    def restore(
        self,
        snapshot: HybridStateSnapshot,
        seq_ids: Sequence[int],
    ) -> None:
        """Restore a subset of sequences from an earlier snapshot."""
        normalized = self._normalize_seq_ids(seq_ids)
        snapshot_indices = {
            seq_id: index for index, seq_id in enumerate(snapshot.seq_ids)
        }
        try:
            selected_indices = [snapshot_indices[seq_id] for seq_id in normalized]
        except KeyError as exc:
            raise ValueError("sequence is absent from the state snapshot") from exc
        destination_slots = torch.tensor(
            [self._slots.lookup(seq_id) for seq_id in normalized],
            dtype=torch.long,
            device=self.device,
        )
        source_indices = torch.tensor(
            selected_indices, dtype=torch.long, device=self.device
        )
        for layer_idx, saved in snapshot.layer_states.items():
            pool = self._linear_state_pools[layer_idx]
            pool.conv_state.index_copy_(
                0,
                destination_slots,
                saved.conv_state.index_select(0, source_indices),
            )
            pool.recurrent_state.index_copy_(
                0,
                destination_slots,
                saved.recurrent_state.index_select(0, source_indices),
            )

    def reset(self) -> None:
        """Clear every state pool and reset all request-slot ownership."""
        for state in self._linear_state_pools.values():
            state.conv_state.zero_()
            state.recurrent_state.zero_()
        self._slots.reset()


__all__ = [
    "GatedDeltaNetStateSpec",
    "HybridCache",
    "HybridStateSnapshot",
    "KVCache",
    "RequestStateSlotAllocator",
]
