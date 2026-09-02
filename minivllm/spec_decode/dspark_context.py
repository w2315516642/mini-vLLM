"""Target hidden-state collection for DFlash/DSpark context injection."""

from typing import Dict, Iterable, Tuple

import torch


class TargetHiddenStateCollector:
    """Collect selected decoder outputs without changing the target API.

    A collector is created for one target forward. Captured tensors retain the
    packed-token order used by the target model, so concatenating features on
    the last dimension preserves every request's token order.
    """

    def __init__(self, layer_ids: Iterable[int]) -> None:
        normalized = tuple(int(layer_id) for layer_id in layer_ids)
        if not normalized:
            raise ValueError("At least one target layer must be collected")
        if tuple(sorted(set(normalized))) != normalized:
            raise ValueError("Target layer ids must be increasing and unique")
        self.layer_ids: Tuple[int, ...] = normalized
        self._layer_id_set = frozenset(normalized)
        self._hidden_states: Dict[int, torch.Tensor] = {}

    def __call__(self, layer_id: int, hidden_states: torch.Tensor) -> None:
        if layer_id not in self._layer_id_set:
            return
        if layer_id in self._hidden_states:
            raise RuntimeError(f"Target layer {layer_id} was captured twice")
        self._hidden_states[layer_id] = hidden_states

    def concatenate(self) -> torch.Tensor:
        missing = [
            layer_id
            for layer_id in self.layer_ids
            if layer_id not in self._hidden_states
        ]
        if missing:
            raise RuntimeError(f"Target hidden states are missing layers {missing}")
        tensors = [self._hidden_states[layer_id] for layer_id in self.layer_ids]
        token_shape = tensors[0].shape[:-1]
        if any(tensor.shape[:-1] != token_shape for tensor in tensors[1:]):
            raise RuntimeError("Collected target hidden states use different token shapes")
        return torch.cat(tensors, dim=-1)

    def clear(self) -> None:
        self._hidden_states.clear()
