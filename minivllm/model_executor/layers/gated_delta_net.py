"""Readable Qwen Gated DeltaNet reference implementation.

This module is the correctness oracle for the optimized kernels introduced in
stage 6. It intentionally uses batch-major tensors and explicit PyTorch state
instead of the runtime's packed token layout. Hybrid cache ownership and model
integration are deferred to stages 7 and 8.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class GatedDeltaNetState:
    """State needed to continue one Gated DeltaNet layer.

    Attributes:
        conv_state: Last convolution window, shaped
            ``[batch, conv_dim, conv_kernel_size]``.
        recurrent_state: Delta-rule memory, shaped
            ``[batch, num_value_heads, key_head_dim, value_head_dim]``.

    Both tensors use FP32 in the reference path. The dataclass is immutable,
    but callers should still treat the contained tensors as owned state and
    avoid modifying them in place.
    """

    conv_state: torch.Tensor
    recurrent_state: torch.Tensor


def causal_depthwise_conv1d_reference(
    projected_qkv: torch.Tensor,
    weight: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Qwen's causal depthwise convolution with an explicit state.

    Args:
        projected_qkv: Tensor shaped ``[batch, sequence, channels]``.
        weight: Per-channel kernels shaped ``[channels, kernel_size]``.
        initial_state: Optional previous window shaped
            ``[batch, channels, kernel_size]``.

    Returns:
        A pair ``(activated_output, final_state)``. The output has the same
        shape and dtype as ``projected_qkv``; the final state is FP32.
    """
    if projected_qkv.ndim != 3:
        raise ValueError(
            "projected_qkv must have shape [batch, sequence, channels]"
        )
    if weight.ndim != 2:
        raise ValueError("weight must have shape [channels, kernel_size]")

    input_dtype = projected_qkv.dtype
    device = projected_qkv.device
    batch, seq, c = projected_qkv.shape
    weight_channels, kernel_size = weight.shape
    if weight_channels != c:
        raise ValueError(
            "weight channels must match projected_qkv channels: "
            f"expected {c}, got {weight_channels}"
        )
    if kernel_size <= 0:
        raise ValueError("weight kernel_size must be positive")

    expected_state_shape = (batch, c, kernel_size)
    if (
        initial_state is not None
        and tuple(initial_state.shape) != expected_state_shape
    ):
        raise ValueError(
            "initial_state must have shape "
            f"{expected_state_shape}, got {tuple(initial_state.shape)}"
        )

    if initial_state is None:
        final_state = torch.zeros(
            expected_state_shape,
            dtype=torch.float32,
            device=device,
        )
    else:
        final_state = initial_state.to(
            dtype=torch.float32,
            device=device,
        ).clone()

    weight = weight.to(torch.float32)
    activated_output = torch.empty_like(
        projected_qkv,
        device=device,
        dtype=torch.float32,
    )
    for s in range(seq):
        cur_state = (
            projected_qkv[:, s, :]
            .reshape(batch, c)
            .unsqueeze(-1)
            .to(torch.float32)
        )
        final_state = torch.cat((final_state[..., 1:], cur_state), dim=-1)

        cur_out = final_state * weight  # [batch, channels, kernel_size]
        activated_output[:, s, :] = torch.sum(
            cur_out,
            dim=-1,
            keepdim=False,
        )
    activated_output = torch.nn.functional.silu(activated_output)
    return activated_output.to(dtype=input_dtype), final_state


def recurrent_gated_delta_rule_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the Gated Delta Rule one token at a time.

    Args:
        query: ``[batch, sequence, heads, key_head_dim]``.
        key: ``[batch, sequence, heads, key_head_dim]``.
        value: ``[batch, sequence, heads, value_head_dim]``.
        log_decay: Non-positive ``g`` values shaped
            ``[batch, sequence, heads]``. The actual decay is ``exp(g)``.
        beta: Delta update strengths shaped ``[batch, sequence, heads]``.
        initial_state: Optional FP32 memory shaped
            ``[batch, heads, key_head_dim, value_head_dim]``.
        eps: Epsilon used by Q/K L2 normalization.

    Returns:
        A pair ``(output, final_state)``. Output follows the input value dtype;
        final state remains FP32 for use by a later decode call.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError(
            "query, key, and value must have shape "
            "[batch, sequence, heads, head_dim]"
        )
    if query.shape != key.shape:
        raise ValueError(
            "query and key must have the same shape: "
            f"got {tuple(query.shape)} and {tuple(key.shape)}"
        )
    batch, sequence, heads, key_head_dim = query.shape
    if value.shape[:3] != (batch, sequence, heads):
        raise ValueError(
            "value must match query's batch, sequence, and head dimensions: "
            f"expected {(batch, sequence, heads)}, got {tuple(value.shape[:3])}"
        )
    parameter_shape = (batch, sequence, heads)
    if tuple(log_decay.shape) != parameter_shape:
        raise ValueError(
            f"log_decay must have shape {parameter_shape}, "
            f"got {tuple(log_decay.shape)}"
        )
    if tuple(beta.shape) != parameter_shape:
        raise ValueError(
            f"beta must have shape {parameter_shape}, got {tuple(beta.shape)}"
        )
    if eps <= 0:
        raise ValueError("eps must be positive")

    value_head_dim = value.shape[-1]
    expected_state_shape = (batch, heads, key_head_dim, value_head_dim)
    if (
        initial_state is not None
        and tuple(initial_state.shape) != expected_state_shape
    ):
        raise ValueError(
            "initial_state must have shape "
            f"{expected_state_shape}, got {tuple(initial_state.shape)}"
        )

    out_dtype = value.dtype

    # The reference keeps normalization and recurrent accumulation in FP32.
    query = query.float()
    key = key.float()
    value = value.float()

    q_norm = torch.norm(query, dim=-1, keepdim=True).clamp_min(eps)
    k_norm = torch.norm(key, dim=-1, keepdim=True).clamp_min(eps)

    query = query / q_norm
    key = key / k_norm

    query = query * (query.shape[-1] ** (-0.5))

    decay = log_decay.float().exp()

    if initial_state is None:
        final_state = torch.zeros(
            size=expected_state_shape,
            dtype=torch.float32,
            device=query.device,
        )
    else:
        final_state = initial_state.to(
            device=query.device,
            dtype=torch.float32,
        ).clone()

    output = torch.empty(
        size=(batch, sequence, heads, value_head_dim),
        dtype=torch.float32,
        device=query.device,
    )

    beta = beta.float().unsqueeze(-1).unsqueeze(-1)
    for s in range(sequence):
        decay_fac = decay[:, s, :].unsqueeze(-1).unsqueeze(-1)
        s_decay = decay_fac * final_state

        old_value = key[:, s, ...].unsqueeze(-2) @ s_decay

        delta = value[:, s, ...].unsqueeze(-2) - old_value
        delta *= beta[:, s, ...]

        new_value = key[:, s, ...].unsqueeze(-1) @ delta
        final_state = s_decay + new_value

        q = query[:, s, ...].unsqueeze(-2)
        out = q @ final_state
        output[:, s, ...] = out.squeeze(-2)

    return output.to(out_dtype), final_state


class RMSNormGated(nn.Module):
    """Per-value-head RMSNorm followed by Qwen's SiLU output gate."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self,
        hidden_states: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.shape != gate.shape:
            raise ValueError(
                "hidden_states and gate must have the same shape: "
                f"got {tuple(hidden_states.shape)} and {tuple(gate.shape)}"
            )
        if hidden_states.ndim == 0 or hidden_states.shape[-1] != self.weight.numel():
            raise ValueError(
                "hidden_states last dimension must match norm weight: "
                f"expected {self.weight.numel()}"
            )

        in_dtype = hidden_states.dtype

        hidden_states = hidden_states.float()
        variance = hidden_states.square().mean(dim=-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon
        )
        weighted = normalized * self.weight.float()
        gated = weighted * nn.functional.silu(gate.float())

        return gated.to(in_dtype)


class Qwen3_5GatedDeltaNetReference(nn.Module):
    """Unsharded, readable Qwen Gated DeltaNet layer.

    This class mirrors the checkpoint-level projections, but it is not yet the
    packed runtime layer. Its explicit state interface is intentionally stable
    so later kernels and Hybrid Cache code can be tested against it.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_norm_epsilon = config.rms_norm_eps

        dimensions = {
            "hidden_size": self.hidden_size,
            "linear_num_key_heads": self.num_k_heads,
            "linear_num_value_heads": self.num_v_heads,
            "linear_key_head_dim": self.head_k_dim,
            "linear_value_head_dim": self.head_v_dim,
            "linear_conv_kernel_dim": self.conv_kernel_size,
        }
        for name, value in dimensions.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                "linear_num_value_heads must be divisible by "
                "linear_num_key_heads"
            )
        if getattr(config, "hidden_act", "silu") not in ("silu", "swish"):
            raise ValueError("Qwen Gated DeltaNet reference requires SiLU")

        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim

        self.in_proj_qkv = nn.Linear(
            self.hidden_size,
            self.conv_dim,
            bias=False,
        )
        self.in_proj_z = nn.Linear(
            self.hidden_size,
            self.value_dim,
            bias=False,
        )
        self.in_proj_b = nn.Linear(
            self.hidden_size,
            self.num_v_heads,
            bias=False,
        )
        self.in_proj_a = nn.Linear(
            self.hidden_size,
            self.num_v_heads,
            bias=False,
        )
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            bias=False,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        initial_a = torch.empty(self.num_v_heads).uniform_(0.01, 16.0)
        self.A_log = nn.Parameter(initial_a.log())
        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
        )
        self.out_proj = nn.Linear(
            self.value_dim,
            self.hidden_size,
            bias=False,
        )

    def empty_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> GatedDeltaNetState:
        """Allocate an FP32 zero state matching this layer."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        conv_state = torch.zeros(
            batch_size,
            self.conv_dim,
            self.conv_kernel_size,
            dtype=torch.float32,
            device=device,
        )
        recurrent_state = torch.zeros(
            batch_size,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            dtype=torch.float32,
            device=device,
        )
        return GatedDeltaNetState(conv_state, recurrent_state)

    def forward(
        self,
        hidden_states: torch.Tensor,
        state: Optional[GatedDeltaNetState] = None,
    ) -> Tuple[torch.Tensor, GatedDeltaNetState]:
        """Run a batch-major prefill or continuation through the reference.

        ``hidden_states`` must be ``[batch, sequence, hidden_size]``. Passing
        the returned state into a later call must be equivalent to evaluating
        the concatenated sequence in one call.
        """
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                "hidden_states must have shape [batch, sequence, hidden_size] "
                f"with hidden_size={self.hidden_size}, "
                f"got {tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[0] <= 0:
            raise ValueError("hidden_states batch dimension must be positive")
        if state is not None and not isinstance(state, GatedDeltaNetState):
            raise TypeError("state must be a GatedDeltaNetState or None")

        qkv = self.in_proj_qkv(hidden_states)
        gate = self.in_proj_z(hidden_states)
        beta = torch.sigmoid(self.in_proj_b(hidden_states))
        decay_input = self.in_proj_a(hidden_states)

        batch, sequence = qkv.shape[0], qkv.shape[1]
        if state is None:
            state = self.empty_state(batch, hidden_states.device)

        # short conv
        conv_weight = self.conv1d.weight.squeeze(1)
        conv_qkv, conv_state = causal_depthwise_conv1d_reference(
            qkv, conv_weight, state.conv_state)

        # get q k v
        query, key, value = conv_qkv.split(
            [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape([batch, sequence, self.num_k_heads, self.head_k_dim])
        key = key.reshape([batch, sequence, self.num_k_heads, self.head_k_dim])
        value = value.reshape([batch, sequence, self.num_v_heads, self.head_v_dim])
        if self.num_k_heads < self.num_v_heads:
            repeat_cnt = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeat_cnt, dim=2)
            key = key.repeat_interleave(repeat_cnt, dim=2)

        # g is the log-domain decay consumed by the recurrent reference.
        decay_rate = self.A_log.float().exp()
        bias = self.dt_bias.float().unsqueeze(0).unsqueeze(0)
        forget = nn.functional.softplus(decay_input.float() + bias)
        log_decay = -decay_rate * forget

        new_value, recurrent_state = recurrent_gated_delta_rule_reference(
            query,
            key,
            value,
            log_decay,
            beta,
            state.recurrent_state,
        )

        gate = gate.reshape(
            batch,
            sequence,
            self.num_v_heads,
            self.head_v_dim,
        )
        norm_value = self.norm(new_value, gate)
        intermediate = norm_value.reshape(batch, sequence, self.value_dim)
        output = self.out_proj(intermediate)
        new_state = GatedDeltaNetState(conv_state, recurrent_state)
        return output, new_state


__all__ = [
    "GatedDeltaNetState",
    "Qwen3_5GatedDeltaNetReference",
    "RMSNormGated",
    "causal_depthwise_conv1d_reference",
    "recurrent_gated_delta_rule_reference",
]
