"""Layer-aligned cache storage for Qwen hybrid decoders.

Full-attention layers keep using the existing paged KV cache. Linear-attention
layers instead own fixed-size convolution and recurrent-state pools indexed by
a stable request slot. Stage 8 will connect this container to the model runner;
this stage focuses on state ownership and lifecycle semantics.
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


@dataclass(frozen=True)
class GatedDeltaNetStateSpec:
    """Per-request state dimensions shared by all Qwen linear layers."""

    num_key_heads: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_kernel_size: int

    @classmethod
    def from_text_config(cls, text_config) -> "GatedDeltaNetStateSpec":
        num_key_heads = _positive_config_int(
            text_config, "linear_num_key_heads"
        )
        num_value_heads = _positive_config_int(
            text_config, "linear_num_value_heads"
        )
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

    def is_active(self, seq_id: int) -> bool:
        """Return whether a sequence already owns a state slot."""
        return seq_id in self._seq_to_slot

    def acquire(self, seq_id: int) -> Tuple[int, bool]:
        """Return ``(slot, is_new)`` while preserving existing ownership."""
        is_new = seq_id not in self._seq_to_slot
        # 重新分配新 slot
        if is_new:
            exhausted = len(self._free_slots) <= 0
            if exhausted:
                raise ValueError("No free state slots available")
            slot = self._free_slots.pop()
            self._seq_to_slot[seq_id] = slot
        else:
            slot = self._seq_to_slot[seq_id]
        return slot, is_new

    def lookup(self, seq_id: int) -> int:
        """Return the slot owned by an active sequence."""
        is_new = seq_id not in self._seq_to_slot
        if is_new:
            raise ValueError(f"Sequence {seq_id} is not active")
        return self._seq_to_slot[seq_id]

    def release(self, seq_id: int) -> int:
        """Release one active sequence and return its former slot."""
        is_new = seq_id not in self._seq_to_slot
        if is_new:
            raise ValueError(f"Sequence {seq_id} is not active")
        slot = self._seq_to_slot[seq_id]
        self._free_slots.append(slot)
        del self._seq_to_slot[seq_id]
        return slot

    def reset(self) -> None:
        """Release every sequence and restore deterministic slot order."""
        self._seq_to_slot: Dict[int, int] = {}
        self._free_slots = list(range(self.capacity - 1, -1, -1))


class HybridCache:
    """Coordinate paged KV caches and per-request GDN state pools.

    The cache keeps global layer indices. ``full_attention_caches`` must have
    exactly one entry for every full-attention layer and no entries for linear
    layers. All linear layers share one request-slot allocator, but own separate
    state tensors.
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
            if (
                not isinstance(seq_id, int)
                or isinstance(seq_id, bool)
                or seq_id < 0
            ):
                raise ValueError("seq_ids must contain non-negative integers")
        if len(set(normalized)) != len(normalized):
            raise ValueError("seq_ids must not contain duplicates")
        return normalized

    def _normalize_slot_ids(
        self,
        slot_ids: torch.Tensor | Sequence[int],
    ) -> torch.Tensor:
        """Normalize index metadata without inspecting CUDA tensor values."""
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

        # Slots from the allocator are unique and in range. Rechecking their
        # CUDA values here would synchronize the host on every layer read/write.
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
        seq_ids = self._normalize_seq_ids(seq_ids)
        num_new_seq_ids = sum(
            not self._slots.is_active(seq_id) for seq_id in seq_ids
        )
        if num_new_seq_ids > self._slots.num_free_slots:
            raise ValueError(
                "Not enough free state slots: "
                f"need {num_new_seq_ids}, available {self._slots.num_free_slots}"
            )

        slot_ids = []
        for seq_id in seq_ids:
            slot, is_new = self._slots.acquire(seq_id)
            if is_new:
                self._clear_rows([slot])
            slot_ids.append(slot)
        return self._normalize_slot_ids(slot_ids)

    def get_kv_cache(self, layer_idx: int) -> KVCache:
        """Return the paged KV cache owned by one full-attention layer."""
        self._require_layer_type(layer_idx, FULL_ATTENTION)
        return self._full_attention_caches[layer_idx]

    def read_state(
        self,
        layer_idx: int,
        slot_ids: torch.Tensor | Sequence[int],
    ) -> GatedDeltaNetState:
        """Gather a contiguous GDN state batch using caller-owned valid slots."""
        self._require_layer_type(layer_idx, LINEAR_ATTENTION)

        slot_ids = self._normalize_slot_ids(slot_ids)
        state = self._linear_state_pools[layer_idx]
        conv_state = state.conv_state.index_select(0, slot_ids)
        recurrent_state = state.recurrent_state.index_select(0, slot_ids)
        return GatedDeltaNetState(conv_state, recurrent_state)

    def write_state(
        self,
        layer_idx: int,
        slot_ids: torch.Tensor | Sequence[int],
        state: GatedDeltaNetState,
    ) -> None:
        """Scatter a kernel-updated GDN state batch back to its slots.

        The caller supplies unique, valid slots and a matching FP32 state batch
        on the pool device. No state validation or rollback runs on this path.
        """
        self._require_layer_type(layer_idx, LINEAR_ATTENTION)

        old_state = self._linear_state_pools[layer_idx]
        conv_state = state.conv_state
        recurrent_state = state.recurrent_state

        slot_ids = self._normalize_slot_ids(slot_ids)
        old_state.conv_state.index_copy_(0, slot_ids, conv_state)
        old_state.recurrent_state.index_copy_(0, slot_ids, recurrent_state)

    def fork(self, parent_seq_id: int, child_seq_id: int) -> int:
        """Allocate a child slot and copy every linear-layer state into it."""
        self._normalize_seq_ids((parent_seq_id, child_seq_id))
        parent_slot_id = self._slots.lookup(parent_seq_id)
        if self._slots.is_active(child_seq_id):
            raise ValueError(f"Child sequence {child_seq_id} is already active")
        child_slot_id, _ = self._slots.acquire(child_seq_id)

        # Keep slot IDs as Python integers. Each copy overwrites a whole child
        # row, so no gather buffer, device-to-host index read, or pre-clear is needed.
        for state in self._linear_state_pools.values():
            state.conv_state[child_slot_id].copy_(state.conv_state[parent_slot_id])
            state.recurrent_state[child_slot_id].copy_(
                state.recurrent_state[parent_slot_id]
            )
        return child_slot_id

    def release(self, seq_ids: Sequence[int]) -> None:
        """Release requests and clear their state before slot reuse."""
        seq_ids = self._normalize_seq_ids(seq_ids)
        # Resolve the entire batch before changing any state or ownership.
        slot_ids = [self._slots.lookup(seq_id) for seq_id in seq_ids]
        self._clear_rows(slot_ids)
        for seq_id in seq_ids:
            self._slots.release(seq_id)

    def reset(self) -> None:
        """Reset request ownership while retaining the allocated cache layout."""
        for state in self._linear_state_pools.values():
            state.conv_state.zero_()
            state.recurrent_state.zero_()
        self._slots.reset()

    def _clear_rows(self, slot_ids: torch.Tensor | Sequence[int]) -> None:
        slot_ids = self._normalize_slot_ids(slot_ids)
        if slot_ids.numel() == 0:
            return
        # Advanced indexing followed by zero_() would clear a temporary copy.
        for state in self._linear_state_pools.values():
            state.conv_state.index_fill_(0, slot_ids, 0)
            state.recurrent_state.index_fill_(0, slot_ids, 0)


__all__ = [
    "GatedDeltaNetStateSpec",
    "HybridCache",
    "KVCache",
    "RequestStateSlotAllocator",
]
