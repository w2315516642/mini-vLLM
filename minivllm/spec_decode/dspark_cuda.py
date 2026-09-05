"""Small CUDA primitives used by the DSpark sequential head."""

from importlib import import_module

import torch


def markov_argmax(
    base_logits: torch.Tensor,
    previous_embeddings: torch.Tensor,
    projection_weight: torch.Tensor,
) -> torch.Tensor:
    """Apply ``base + previous @ weight.T`` and return each row's argmax."""
    if (base_logits.is_cuda and base_logits.dtype == torch.float32
        and previous_embeddings.dtype in (torch.float16, torch.bfloat16)
        and projection_weight.dtype == previous_embeddings.dtype
        and base_logits.ndim == previous_embeddings.ndim == projection_weight.ndim == 2
        and previous_embeddings.device == projection_weight.device == base_logits.device
        and previous_embeddings.shape[0] == base_logits.shape[0]
        and projection_weight.shape == (base_logits.shape[1], previous_embeddings.shape[1])
        and min(*base_logits.shape, previous_embeddings.shape[1]) > 0
        and all(t.stride(1) == 1 for t in (base_logits, previous_embeddings, projection_weight))):
        from minivllm.spec_decode.markov_triton import tiled_markov_argmax
        return tiled_markov_argmax(base_logits, previous_embeddings, projection_weight)
    return import_module("minivllm.dspark_ops").markov_argmax(
        base_logits,
        previous_embeddings,
        projection_weight,
    )


__all__ = ["markov_argmax"]
