"""CUDA wrappers for the stateful Gated DeltaNet operators.

The extension is imported lazily so the readable stage-5 reference and the
rest of mini-vLLM remain importable before the stage-6 CUDA code is rebuilt.
Both CUDA state tensors are updated in place; cache ownership stays outside
these numerical operators and will be introduced by the Hybrid Cache stage.
"""

from importlib import import_module
from typing import Tuple

import torch
from minivllm.profiling import nvtx_function


def _load_ops():
    return import_module("minivllm.gated_delta_net_ops")


@nvtx_function("gdn_prepare_qk")
def prepare_gated_delta_qk(
    query: torch.Tensor,
    key: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """L2-normalize Q/K in FP32 and apply the query scale.

    The function accepts decode ``[B,H,Dk]`` or prefill ``[B,T,H,Dk]``
    layouts and returns contiguous tensors in the original input dtype.
    """
    if query.ndim not in (3, 4) or key.ndim != query.ndim:
        raise ValueError(
            "query and key must use [B,H,Dk] or [B,T,H,Dk] layouts"
        )
    if query.shape != key.shape:
        raise ValueError(
            "query and key must have the same shape: "
            f"got {tuple(query.shape)} and {tuple(key.shape)}"
        )
    if query.dtype != key.dtype or query.device != key.device:
        raise ValueError("query and key must have the same dtype and device")
    if query.shape[-1] <= 0:
        raise ValueError("key head dimension must be positive")
    if eps <= 0:
        raise ValueError("eps must be positive")

    query_fp32 = query.float()
    key_fp32 = key.float()
    query_fp32 = query_fp32 / query_fp32.norm(
        dim=-1,
        keepdim=True,
    ).clamp_min(eps)
    key_fp32 = key_fp32 / key_fp32.norm(
        dim=-1,
        keepdim=True,
    ).clamp_min(eps)
    query_fp32 = query_fp32 * (query.shape[-1] ** -0.5)
    return (
        query_fp32.to(query.dtype).contiguous(),
        key_fp32.to(key.dtype).contiguous(),
    )


@nvtx_function("gdn_conv")
def causal_conv1d_update(
    projected_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Update an FP32 convolution state and return one activated token."""
    output = torch.empty_like(projected_qkv)
    _load_ops().causal_conv1d_update(
        output,
        projected_qkv,
        conv_state,
        weight,
    )
    return output


@nvtx_function("gdn_recurrence")
def gated_delta_rule_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
) -> torch.Tensor:
    """Apply one Gated Delta Rule step and update state in place.

    ``query`` and ``key`` must already be prepared by
    :func:`prepare_gated_delta_qk`.
    """
    output = torch.empty_like(value)
    _load_ops().gated_delta_rule_decode(
        output,
        query,
        key,
        value,
        log_decay,
        beta,
        recurrent_state,
    )
    return output


@nvtx_function("gdn_recurrence")
def gated_delta_rule_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Run chunked prefill and leave state after the final token in place.

    ``query`` and ``key`` must already be prepared by
    :func:`prepare_gated_delta_qk`.
    """
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if query.is_cuda and query.ndim == 4 and query.shape[1] >= 16:
        from .gdn_prefill import gdn_prefill
        return gdn_prefill(query, key, value, log_decay, beta, recurrent_state)
    output = torch.empty_like(value)
    _load_ops().gated_delta_rule_prefill(
        output,
        query,
        key,
        value,
        log_decay,
        beta,
        recurrent_state,
        chunk_size,
    )
    return output


@nvtx_function("gdn_conv")
def causal_conv1d_varlen(projected_qkv, conv_state, weight, cu_seqlens, lengths=None):
    """Packed [tokens, channels] convolution with one state per sequence.

    cu_seqlens is immutable int32 metadata built from CPU sequence lengths.
    Optional lengths limits each sequence to its accepted prefix during replay;
    output outside those prefixes is unspecified and must not be consumed.
    """
    output = torch.empty_like(projected_qkv)
    _load_ops().causal_conv1d_varlen(
        output, projected_qkv, conv_state, weight, cu_seqlens, lengths,
    )
    return output


@nvtx_function("gdn_recurrence")
def gated_delta_rule_varlen(
    query, key, value, log_decay, beta, recurrent_state,
    cu_seqlens, max_seqlen, lengths=None,
):
    """Packed [tokens, heads, dim] recurrence without advancing padded tokens.

    max_seqlen and optional accepted lengths are validated on the CPU by the
    metadata/replay owner. All layers reuse the same CUDA metadata tensors.
    """
    if query.is_cuda and max_seqlen >= 16:
        from .gdn_prefill import gdn_prefill
        return gdn_prefill(query, key, value, log_decay, beta, recurrent_state,
                           cu_seqlens, lengths)
    output = torch.empty_like(value)
    _load_ops().gated_delta_rule_varlen(
        output, query, key, value, log_decay, beta, recurrent_state,
        cu_seqlens, max_seqlen, lengths,
    )
    return output
