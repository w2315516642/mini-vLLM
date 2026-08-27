"""Qwen3.5/3.8 full-attention building blocks.

Qwen3.8 keeps the Qwen3.5 checkpoint schema. This stage implements only the
full-attention token mixer; the hybrid decoder and its linear-attention layers
are introduced in later stages.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from minivllm.model_executor.input_metadata import InputMetadata
from minivllm.model_executor.layers.activation import SigmoidAndMul
from minivllm.model_executor.layers.attention import PagedAttentionWithRoPE
from minivllm.model_executor.layers.layer_norm import Qwen3_5RMSNorm
from minivllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size,
)
from minivllm.model_executor.parallel_utils.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
)

KVCache = Tuple[torch.Tensor, torch.Tensor]


def _split_q_gate_kv(
    packed: torch.Tensor,   # [num_tokens, 2 * q_size + kv_size + kv_size]
    num_query_heads: int,
    head_dim: int,
    kv_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split rank-local ``[Q+gate | K | V]`` projection output.

    The Q checkpoint segment is arranged per head as ``[q_head | gate_head]``.
    Returned tensors are flattened to the interfaces used by PagedAttention.
    """
    q_size = num_query_heads * head_dim
    q_gate, k, v = torch.split(packed, [2 * q_size, kv_size, kv_size], dim=-1)

    q_gate = q_gate.view(-1, num_query_heads, 2, head_dim)
    q, gate = q_gate[..., 0, :], q_gate[..., 1, :]
    q = q.contiguous().view(-1, q_size)
    gate = gate.contiguous().view(-1, q_size)

    return q, gate, k, v


def _load_qkv_gate_weight(
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    shard_id: str,
    tensor_model_parallel_rank: int,
    q_size: int,
    kv_size: int,
) -> None:
    """Load one global Q/K/V checkpoint tensor into a local packed weight."""
    if shard_id not in ("q", "k", "v"):
        raise ValueError(
            f"Unknown QKV checkpoint shard {shard_id!r}; "
            "expected one of 'q', 'k', or 'v'"
        )
    qkv_map = {"q": [2 * q_size, 0], "k": [kv_size, 2 * q_size], "v": [kv_size, 2 * q_size + kv_size]}
    shard_size, offset = qkv_map[shard_id]

    cur_tp_weight = loaded_weight[
        shard_size * tensor_model_parallel_rank:
        shard_size * (tensor_model_parallel_rank + 1)]
    param_slice = param.data[offset : offset + shard_size]
    assert param_slice.shape == cur_tp_weight.shape, (
        f"Parameter slice shape {param_slice.shape} does not match "
        f"cur_tp_weight shape {cur_tp_weight.shape}"
    )
    param_slice.copy_(cur_tp_weight)


class Qwen3_5Attention(nn.Module):
    """Gated full-attention layer shared by Qwen3.5 and Qwen3.8."""

    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        tp_world_size = get_tensor_model_parallel_world_size()
        if self.total_num_heads % tp_world_size != 0:
            raise ValueError("Query heads must be divisible by TP size")
        if self.total_num_kv_heads % tp_world_size != 0:
            raise ValueError("KV heads must be divisible by TP size")
        if self.total_num_heads % self.total_num_kv_heads != 0:
            raise ValueError("Query heads must be grouped evenly by KV heads")

        self.num_heads = self.total_num_heads // tp_world_size
        self.num_kv_heads = self.total_num_kv_heads // tp_world_size
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        self.rotary_dim = int(self.head_dim * partial_rotary_factor)
        if self.rotary_dim <= 0 or self.rotary_dim > self.head_dim:
            raise ValueError("rotary_dim must be in (0, head_dim]")
        if self.rotary_dim % 2 != 0:
            raise ValueError("rotary_dim must be even")

        self.qkv_gate_proj = ColumnParallelLinear(
            self.hidden_size,
            (2 * self.total_num_heads + 2 * self.total_num_kv_heads)
            * self.head_dim,
            bias=False,
            gather_output=False,
            perform_initialization=False,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            perform_initialization=False,
        )
        self.q_norm = Qwen3_5RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
        )
        self.k_norm = Qwen3_5RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
        )
        self.attn = PagedAttentionWithRoPE(
            self.num_heads,
            self.head_dim,
            self.scaling,
            rotary_dim=self.rotary_dim,
            max_position=config.max_position_embeddings,
            base=float(config.rope_theta),
            num_kv_heads=self.num_kv_heads,
        )
        self.gate_fn = SigmoidAndMul()

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project and prepare flat Q/K/V/gate tensors for the hot path."""
        qgkv, _ = self.qkv_gate_proj(hidden_states)
        q, gate, k, v = _split_q_gate_kv(qgkv, self.num_heads, self.head_dim, self.kv_size)

        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q.view(-1, self.q_size)
        k = k.view(-1, self.kv_size)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        gate = gate.contiguous()
        return q, k, v, gate

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        q, k, v, gate = self._project_qkv(hidden_states)

        k_cache, v_cache = kv_cache
        attn_out = self.attn(
            positions, q, k, v, k_cache, v_cache, input_metadata, cache_event
        )
        gate_out = self.gate_fn(attn_out, gate)
        output, _ = self.o_proj(gate_out)
        return output


__all__ = [
    "Qwen3_5Attention",
    "_load_qkv_gate_weight",
    "_split_q_gate_kv",
]
