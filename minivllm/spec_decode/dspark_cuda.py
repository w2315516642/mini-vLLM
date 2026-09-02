"""Small CUDA primitives used by the DSpark sequential head."""

from importlib import import_module

import torch


def markov_argmax(
    base_logits: torch.Tensor,
    previous_embeddings: torch.Tensor,
    projection_weight: torch.Tensor,
) -> torch.Tensor:
    """Apply ``base + previous @ weight.T`` and return each row's argmax."""
    return import_module("minivllm.dspark_ops").markov_argmax(
        base_logits,
        previous_embeddings,
        projection_weight,
    )


__all__ = ["markov_argmax"]
