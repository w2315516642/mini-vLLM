"""Qwen3.5/3.8 hybrid text decoder used by the mini-vLLM runtime.

The checkpoint alternates stateful Gated DeltaNet token mixers with gated
full-attention layers. Packed prompts and decode tokens share one residual
stream, while :class:`HybridCache` provides the different cache type required
by each layer.
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from minivllm.configs.model_architecture import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
)
from minivllm.model_executor.input_metadata import InputMetadata
from minivllm.model_executor.layers.activation import (
    SigmoidAndMul,
    SiluAndMul,
)
from minivllm.model_executor.layers.attention import (
    PagedAttention,
    PagedAttentionWithRoPE,
)
from minivllm.model_executor.layers.gated_delta_net import (
    GatedDeltaNetState,
    RMSNormGated,
)
from minivllm.model_executor.layers.gated_delta_net_cuda import (
    causal_conv1d_update,
    gated_delta_rule_decode,
    gated_delta_rule_prefill,
    prepare_gated_delta_qk,
)
from minivllm.model_executor.layers.layer_norm import Qwen3_5RMSNorm
from minivllm.model_executor.layers.sampler import Sampler
from minivllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from minivllm.model_executor.parallel_utils.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    gather_from_tensor_model_parallel_region,
)
from minivllm.model_executor.weight_utils import (
    hf_model_weights_iterator,
    load_tensor_parallel_weights,
)
from minivllm.model_executor.models.qwen3_5_vision import Qwen3_5VisionModel
from minivllm.sequence import SequenceOutputs
from minivllm.spec_decode.greedy_verifier import verify_greedy_block
from minivllm.worker.hybrid_cache import HybridCache

KVCache = Tuple[torch.Tensor, torch.Tensor]


def _apply_interleaved_mrope(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    head_dim: int,
    cos_sin_cache: torch.Tensor,
    mrope_section: Sequence[int],
) -> None:
    """Apply Qwen temporal/height/width RoPE to flat Q/K in place."""
    if positions.ndim != 2 or positions.shape[0] != 3:
        raise ValueError("Qwen M-RoPE positions must have shape [3, tokens]")
    rotary_dim = cos_sin_cache.shape[1]
    frequency_dim = rotary_dim // 2
    if sum(mrope_section) != frequency_dim:
        raise ValueError(
            "mrope_section must sum to half the rotary dimension"
        )
    selected = cos_sin_cache[positions]
    cos = selected[0, :, :frequency_dim].clone()
    sin = selected[0, :, frequency_dim:].clone()
    for dimension, offset in enumerate((1, 2), start=1):
        indices = slice(offset, mrope_section[dimension] * 3, 3)
        cos[:, indices] = selected[dimension, :, :frequency_dim][:, indices]
        sin[:, indices] = selected[dimension, :, frequency_dim:][:, indices]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    def rotate(tensor: torch.Tensor) -> None:
        heads = tensor.view(tensor.shape[0], -1, head_dim)
        first = heads[..., :frequency_dim].clone()
        second = heads[..., frequency_dim:rotary_dim].clone()
        heads[..., :frequency_dim] = first * cos - second * sin
        heads[..., frequency_dim:rotary_dim] = second * cos + first * sin

    rotate(query)
    rotate(key)


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


def _load_qkv_gate_scale(
    param: torch.Tensor,
    loaded_scale: torch.Tensor,
    shard_id: str,
    tensor_model_parallel_rank: int,
    q_size: int,
    kv_size: int,
    block_n: int,
) -> None:
    """Load block-row scales for one Q/K/V source into packed local scales."""
    if shard_id not in ("q", "k", "v"):
        raise ValueError(f"Unknown QKV checkpoint shard {shard_id!r}")
    local_rows = {"q": 2 * q_size, "k": kv_size, "v": kv_size}
    local_scale_rows = {
        name: (rows + block_n - 1) // block_n
        for name, rows in local_rows.items()
    }
    if sum(local_scale_rows.values()) != param.shape[0]:
        raise ValueError(
            "Packed QKV segments must align to FP8 output block boundaries"
        )
    shard_order = ("q", "k", "v")
    shard_rows = local_scale_rows[shard_id]
    offset = sum(
        local_scale_rows[name]
        for name in shard_order[:shard_order.index(shard_id)]
    )
    current = loaded_scale[
        shard_rows * tensor_model_parallel_rank:
        shard_rows * (tensor_model_parallel_rank + 1)
    ]
    target = param.data[offset : offset + shard_rows]
    if target.shape != current.shape:
        raise ValueError(
            f"Scale slice shape {target.shape} does not match {current.shape}"
        )
    target.copy_(current)


def _load_merged_column_shard(
    param: torch.Tensor,
    loaded_value: torch.Tensor,
    shard_idx: int,
    tensor_model_parallel_rank: int,
) -> None:
    """Load one of two equal global projections into a local packed tensor."""
    if shard_idx not in (0, 1) or param.shape[0] % 2 != 0:
        raise ValueError("merged column parameter must contain two equal shards")
    shard_size = param.shape[0] // 2
    current = loaded_value[
        shard_size * tensor_model_parallel_rank:
        shard_size * (tensor_model_parallel_rank + 1)
    ]
    target = param.data[shard_idx * shard_size:(shard_idx + 1) * shard_size]
    if target.shape != current.shape:
        raise ValueError(
            f"Merged parameter slice shape {target.shape} does not match "
            f"checkpoint shard shape {current.shape}"
        )
    target.copy_(current)


def _load_gdn_qkv_shard(
    param: torch.Tensor,
    loaded_value: torch.Tensor,
    tensor_model_parallel_rank: int,
    tensor_model_parallel_world_size: int,
    global_key_dim: int,
    global_value_dim: int,
    block_n: Optional[int] = None,
) -> None:
    """Shard a checkpoint-level ``[Q | K | V]`` GDN tensor by segment."""
    global_rows = (global_key_dim, global_key_dim, global_value_dim)
    if any(rows % tensor_model_parallel_world_size != 0 for rows in global_rows):
        raise ValueError("GDN Q/K/V dimensions must be divisible by TP size")

    if block_n is None:
        global_segment_sizes = global_rows
        local_segment_sizes = tuple(
            rows // tensor_model_parallel_world_size for rows in global_rows
        )
    else:
        if any(rows % block_n != 0 for rows in global_rows):
            raise ValueError("GDN segments must align to FP8 output blocks")
        global_segment_sizes = tuple(rows // block_n for rows in global_rows)
        if any(
            rows % tensor_model_parallel_world_size != 0
            for rows in global_segment_sizes
        ):
            raise ValueError(
                "GDN FP8 scale blocks must be divisible by TP size"
            )
        local_segment_sizes = tuple(
            rows // tensor_model_parallel_world_size
            for rows in global_segment_sizes
        )

    if sum(local_segment_sizes) != param.shape[0]:
        raise ValueError(
            f"Local GDN packed rows {param.shape[0]} do not match expected "
            f"{sum(local_segment_sizes)}"
        )
    global_offset = 0
    local_offset = 0
    for global_size, local_size in zip(
        global_segment_sizes, local_segment_sizes
    ):
        source_start = (
            global_offset
            + tensor_model_parallel_rank * local_size
        )
        source = loaded_value[source_start:source_start + local_size]
        target = param.data[local_offset:local_offset + local_size]
        if target.shape != source.shape:
            raise ValueError(
                f"GDN target shape {target.shape} does not match "
                f"checkpoint shard shape {source.shape}"
            )
        target.copy_(source)
        global_offset += global_size
        local_offset += local_size


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
        quant_config = getattr(config, "quantization_config", None)

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
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            perform_initialization=False,
            quant_config=quant_config,
        )
        self.q_norm = Qwen3_5RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
        )
        self.k_norm = Qwen3_5RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
        )
        rope_theta = getattr(config, "rope_theta", None)
        rope_parameters = getattr(config, "rope_parameters", {}) or {}
        if rope_theta is None:
            rope_theta = rope_parameters.get("rope_theta", 10_000.0)
        self.mrope_section = tuple(
            rope_parameters.get(
                "mrope_section", (self.rotary_dim // 2, 0, 0)
            )
        )
        self.attn = PagedAttentionWithRoPE(
            self.num_heads,
            self.head_dim,
            self.scaling,
            rotary_dim=self.rotary_dim,
            max_position=config.max_position_embeddings,
            base=float(rope_theta),
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
        kv_cache: Optional[KVCache],
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        q, k, v, gate = self._project_qkv(hidden_states)

        # The native MTP layer attends over independent one-token prompts and
        # deliberately has no persistent target-model KV cache.
        if kv_cache is None:
            k_cache = v_cache = None
        else:
            k_cache, v_cache = kv_cache
        if positions.ndim == 2:
            _apply_interleaved_mrope(
                q,
                k,
                positions,
                self.head_dim,
                self.attn.cos_sin_cache,
                self.mrope_section,
            )
            # RoPE is already applied above; call the cache/attention base
            # implementation directly to avoid applying one-dimensional RoPE.
            attn_out = PagedAttention.forward(
                self.attn,
                q,
                k,
                v,
                k_cache,
                v_cache,
                input_metadata,
                cache_event,
            )
        else:
            attn_out = self.attn(
                positions,
                q,
                k,
                v,
                k_cache,
                v_cache,
                input_metadata,
                cache_event,
            )
        gate_out = self.gate_fn(attn_out, gate)
        output, _ = self.o_proj(gate_out)
        return output


def _causal_conv1d_prefill(
    projected_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Run a continuation-aware causal depthwise convolution.

    ``projected_qkv`` uses ``[batch, sequence, channels]`` while the persistent
    FP32 state uses ``[batch, channels, kernel]``. The final window is copied
    back in place so a later chunk or decode token sees the same history as a
    single concatenated evaluation.
    """
    if projected_qkv.ndim != 3:
        raise ValueError(
            "projected_qkv must have shape [batch, sequence, channels]"
        )
    batch_size, sequence_length, channels = projected_qkv.shape
    if conv_state.ndim != 3 or conv_state.shape[:2] != (
        batch_size,
        channels,
    ):
        raise ValueError(
            "conv_state must match projected_qkv batch and channel dimensions"
        )
    if weight.shape != (channels, conv_state.shape[-1]):
        raise ValueError("weight must have shape [channels, kernel_size]")
    if sequence_length == 0:
        return projected_qkv

    # Keeping the concatenated window in FP32 matches the persistent state and
    # avoids accumulating recurrence error across chunk boundaries.
    projected_channels = projected_qkv.transpose(1, 2).float()
    history = torch.cat((conv_state, projected_channels), dim=-1)
    conv_state.copy_(history[:, :, -conv_state.shape[-1] :])
    output = F.conv1d(
        history,
        weight.float().unsqueeze(1),
        groups=channels,
    )
    output = output[:, :, -sequence_length:]
    return F.silu(output).transpose(1, 2).to(projected_qkv.dtype)


class Qwen3_5MLP(nn.Module):
    """Tensor-parallel SwiGLU feed-forward block."""

    def __init__(self, config) -> None:
        super().__init__()
        if config.hidden_act not in ("silu", "swish"):
            raise ValueError("Qwen MLP requires the SiLU activation")
        quant_config = getattr(config, "quantization_config", None)
        self.gate_up_proj = ColumnParallelLinear(
            config.hidden_size,
            2 * config.intermediate_size,
            bias=False,
            gather_output=False,
            perform_initialization=False,
            quant_config=quant_config,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            perform_initialization=False,
            quant_config=quant_config,
        )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        activated = self.act_fn(gate_up)
        output, _ = self.down_proj(activated)
        return output


class Qwen3_5GatedDeltaNet(nn.Module):
    """Packed-token Gated DeltaNet layer backed by the custom CUDA kernels."""

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        tp_world_size = get_tensor_model_parallel_world_size()
        self.total_num_k_heads = config.linear_num_key_heads
        self.total_num_v_heads = config.linear_num_value_heads
        if self.total_num_k_heads % tp_world_size != 0:
            raise ValueError("Linear key heads must be divisible by TP size")
        if self.total_num_v_heads % tp_world_size != 0:
            raise ValueError("Linear value heads must be divisible by TP size")
        self.num_k_heads = self.total_num_k_heads // tp_world_size
        self.num_v_heads = self.total_num_v_heads // tp_world_size
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                "linear_num_value_heads must be divisible by "
                "linear_num_key_heads"
            )
        if config.hidden_act not in ("silu", "swish"):
            raise ValueError("Qwen Gated DeltaNet requires SiLU")

        self.total_key_dim = self.total_num_k_heads * self.head_k_dim
        self.total_value_dim = self.total_num_v_heads * self.head_v_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.total_conv_dim = 2 * self.total_key_dim + self.total_value_dim
        quant_config = getattr(config, "quantization_config", None)

        self.in_proj_qkv = ColumnParallelLinear(
            self.hidden_size,
            self.total_conv_dim,
            bias=False,
            gather_output=False,
            perform_initialization=False,
            quant_config=quant_config,
        )
        self.in_proj_z = ColumnParallelLinear(
            self.hidden_size,
            self.total_value_dim,
            bias=False,
            gather_output=False,
            perform_initialization=False,
            quant_config=quant_config,
        )
        self.in_proj_b = ColumnParallelLinear(
            self.hidden_size,
            self.total_num_v_heads,
            bias=False,
            gather_output=False,
            perform_initialization=False,
        )
        self.in_proj_a = ColumnParallelLinear(
            self.hidden_size,
            self.total_num_v_heads,
            bias=False,
            gather_output=False,
            perform_initialization=False,
        )
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            bias=False,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        initial_a = torch.empty(self.num_v_heads).uniform_(0.01, 16.0)
        self.A_log = nn.Parameter(initial_a.log())
        self.norm = RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = RowParallelLinear(
            self.total_value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            perform_initialization=False,
            quant_config=quant_config,
        )

    def _empty_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> GatedDeltaNetState:
        return GatedDeltaNetState(
            conv_state=torch.zeros(
                batch_size,
                self.conv_dim,
                self.conv_kernel_size,
                dtype=torch.float32,
                device=device,
            ),
            recurrent_state=torch.zeros(
                batch_size,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
                dtype=torch.float32,
                device=device,
            ),
        )

    def _project(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv, _ = self.in_proj_qkv(hidden_states)
        gate, _ = self.in_proj_z(hidden_states)
        beta, _ = self.in_proj_b(hidden_states)
        decay_input, _ = self.in_proj_a(hidden_states)
        return qkv, gate, beta, decay_input

    def _run_core(
        self,
        qkv: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        decay_input: torch.Tensor,
        state: GatedDeltaNetState,
        *,
        is_decode: bool,
    ) -> torch.Tensor:
        if is_decode:
            mixed_qkv = causal_conv1d_update(
                qkv.contiguous(),
                state.conv_state,
                self.conv1d.weight.squeeze(1).contiguous(),
            )
        else:
            mixed_qkv = _causal_conv1d_prefill(
                qkv,
                state.conv_state,
                self.conv1d.weight.squeeze(1),
            )

        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        prefix_shape = mixed_qkv.shape[:-1]
        query = query.reshape(*prefix_shape, self.num_k_heads, self.head_k_dim)
        key = key.reshape(*prefix_shape, self.num_k_heads, self.head_k_dim)
        value = value.reshape(*prefix_shape, self.num_v_heads, self.head_v_dim)

        repeat_count = self.num_v_heads // self.num_k_heads
        if repeat_count > 1:
            query = query.repeat_interleave(repeat_count, dim=-2)
            key = key.repeat_interleave(repeat_count, dim=-2)
        query, key = prepare_gated_delta_qk(query, key)

        beta = torch.sigmoid(beta).to(query.dtype).contiguous()
        log_decay = -self.A_log.float().exp() * F.softplus(
            decay_input.float() + self.dt_bias.float()
        )
        log_decay = log_decay.contiguous()
        value = value.contiguous()

        if is_decode:
            core_output = gated_delta_rule_decode(
                query,
                key,
                value,
                log_decay,
                beta,
                state.recurrent_state,
            )
        else:
            core_output = gated_delta_rule_prefill(
                query,
                key,
                value,
                log_decay,
                beta,
                state.recurrent_state,
            )

        gate = gate.reshape(*prefix_shape, self.num_v_heads, self.head_v_dim)
        normalized = self.norm(core_output, gate)
        return normalized.reshape(*prefix_shape, self.value_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_metadata: InputMetadata,
        state_cache: Optional[HybridCache],
    ) -> torch.Tensor:
        """Run every packed prompt independently and all decode tokens together."""
        num_valid_tokens = input_metadata.num_valid_tokens
        valid_hidden_states = hidden_states[:num_valid_tokens]
        projected = self._project(valid_hidden_states)
        mixed_output = torch.zeros(
            (num_valid_tokens, self.value_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        state_slots = getattr(input_metadata, "state_slot_mapping", None)
        expected_state_count = (
            input_metadata.num_prompts
            + input_metadata.num_generation_tokens
        )
        if state_cache is not None:
            if state_slots is None or state_slots.numel() != expected_state_count:
                raise ValueError(
                    "state_slot_mapping must contain one slot per packed sequence"
                )

        prompt_offset = 0
        for prompt_idx, prompt_len in enumerate(input_metadata.prompt_lens):
            end = prompt_offset + prompt_len
            if state_cache is None:
                state = self._empty_state(1, hidden_states.device)
                slot = None
            else:
                slot = state_slots[prompt_idx : prompt_idx + 1]
                state = state_cache.read_state(self.layer_idx, slot)

            prompt_projected = tuple(
                tensor[prompt_offset:end].unsqueeze(0) for tensor in projected
            )
            prompt_output = self._run_core(
                *prompt_projected,
                state,
                is_decode=False,
            )
            mixed_output[prompt_offset:end] = prompt_output.squeeze(0)
            if state_cache is not None:
                state_cache.write_state(self.layer_idx, slot, state)
            prompt_offset = end

        num_decode_tokens = input_metadata.num_generation_tokens
        if num_decode_tokens:
            decode_end = prompt_offset + num_decode_tokens
            if state_cache is None:
                state = self._empty_state(num_decode_tokens, hidden_states.device)
                slots = None
            else:
                slots = state_slots[input_metadata.num_prompts :]
                state = state_cache.read_state(self.layer_idx, slots)
            decode_projected = tuple(
                tensor[prompt_offset:decode_end] for tensor in projected
            )
            mixed_output[prompt_offset:decode_end] = self._run_core(
                *decode_projected,
                state,
                is_decode=True,
            )
            if state_cache is not None:
                state_cache.write_state(self.layer_idx, slots, state)

        output_valid, _ = self.out_proj(mixed_output)
        output = torch.zeros_like(hidden_states)
        output[:num_valid_tokens] = output_valid
        return output


class Qwen3_5DecoderLayer(nn.Module):
    """One residual decoder block with a config-selected token mixer."""

    def __init__(
        self,
        config,
        layer_idx: int,
        layer_type: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = (
            config.layer_types[layer_idx] if layer_type is None else layer_type
        )
        if self.layer_type == FULL_ATTENTION:
            self.self_attn = Qwen3_5Attention(config)
            self.linear_attn = None
        elif self.layer_type == LINEAR_ATTENTION:
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
            self.self_attn = None
        else:
            raise ValueError(f"unsupported Qwen layer type: {self.layer_type}")
        self.mlp = Qwen3_5MLP(config)
        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]],
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
        state_cache: Optional[HybridCache],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == FULL_ATTENTION:
            hidden_states = self.self_attn(
                positions,
                hidden_states,
                cache,
                input_metadata,
                cache_event,
            )
        else:
            hidden_states = self.linear_attn(
                hidden_states,
                input_metadata,
                state_cache,
            )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen3_5MultiTokenPredictor(nn.Module):
    """Native Qwen MTP-1 head sharing the target embedding and lm_head."""

    def __init__(self, config) -> None:
        super().__init__()
        num_layers = getattr(config, "mtp_num_hidden_layers", 0)
        if num_layers != 1:
            raise ValueError("The mini runtime supports exactly one Qwen MTP layer")
        if getattr(config, "mtp_use_dedicated_embeddings", False):
            raise ValueError("Dedicated Qwen MTP embeddings are not supported")
        self.config = config
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # Official FP8 checkpoints intentionally keep mtp.fc in BF16.
        self.fc = ColumnParallelLinear(
            2 * config.hidden_size,
            config.hidden_size,
            bias=False,
            gather_output=True,
            perform_initialization=False,
            quant_config=None,
        )
        self.layers = nn.ModuleList([
            Qwen3_5DecoderLayer(
                config, layer_idx=0, layer_type=FULL_ATTENTION
            )
        ])
        self.norm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if inputs_embeds.ndim != 2 or target_hidden_states.ndim != 2:
            raise ValueError("MTP inputs must be [batch, hidden] tensors")
        if inputs_embeds.shape != target_hidden_states.shape:
            raise ValueError("MTP embedding and target hidden shapes must match")
        inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
        target_hidden_states = self.pre_fc_norm_hidden(target_hidden_states)
        hidden_states, _ = self.fc(torch.cat(
            (inputs_embeds, target_hidden_states), dim=-1
        ))

        batch_size = inputs_embeds.shape[0]
        metadata = InputMetadata(
            seq_groups=[],
            seq_data={},
            prompt_lens=[1] * batch_size,
            slot_mapping=torch.zeros(
                batch_size, dtype=torch.int32, device=inputs_embeds.device
            ),
            context_lens=torch.empty(
                0, dtype=torch.int32, device=inputs_embeds.device
            ),
            max_context_len=0,
            block_tables=torch.empty(
                (0, 0), dtype=torch.int32, device=inputs_embeds.device
            ),
            fresh_prompt_lens=[1] * batch_size,
            prompt_sample_indices=[],
        )
        hidden_states = self.layers[0](
            positions,
            hidden_states,
            cache=None,
            input_metadata=metadata,
            cache_event=None,
            state_cache=None,
        )
        return self.norm(hidden_states)


class Qwen3_5Model(nn.Module):
    """Text backbone preserving the checkpoint's global layer order."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            perform_initialization=False,
        )
        self.layers = nn.ModuleList(
            Qwen3_5DecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Sequence[Optional[Tuple[torch.Tensor, torch.Tensor]]]
        | HybridCache,
        input_metadata: InputMetadata,
        cache_events: Optional[Sequence[Optional[torch.cuda.Event]]],
        inputs_embeds: Optional[torch.Tensor] = None,
        hidden_state_collector: Optional[
            Callable[[int, torch.Tensor], None]
        ] = None,
    ) -> torch.Tensor:
        hidden_states = (
            self.embed_tokens(input_ids)
            if inputs_embeds is None
            else inputs_embeds
        )
        state_cache = kv_caches if isinstance(kv_caches, HybridCache) else None
        for layer_idx, layer in enumerate(self.layers):
            cache_event = None if cache_events is None else cache_events[layer_idx]
            if state_cache is not None:
                cache = (
                    state_cache.get_kv_cache(layer_idx)
                    if layer.layer_type == FULL_ATTENTION
                    else None
                )
            else:
                cache = kv_caches[layer_idx]
            hidden_states = layer(
                positions,
                hidden_states,
                cache,
                input_metadata,
                cache_event,
                state_cache,
            )
            if hidden_state_collector is not None:
                hidden_state_collector(layer_idx, hidden_states)
        return self.norm(hidden_states)


class Qwen3_5ForConditionalGeneration(nn.Module):
    """Qwen text generation with optional shared image/video vision tower."""

    def __init__(self, config) -> None:
        super().__init__()
        self.root_config = config
        text_config = getattr(config, "text_config", config)
        vision_config = getattr(config, "vision_config", None)
        self.config = text_config
        self.model = Qwen3_5Model(text_config)
        self.visual = (
            Qwen3_5VisionModel(vision_config)
            if vision_config is not None
            else None
        )
        if (
            vision_config is not None
            and vision_config.out_hidden_size != text_config.hidden_size
        ):
            raise ValueError(
                "Vision out_hidden_size must equal text hidden_size"
            )
        self.image_token_id = getattr(config, "image_token_id", None)
        self.video_token_id = getattr(config, "video_token_id", None)
        self._multimodal_feature_cache: Dict[
            int, Dict[int, torch.Tensor]
        ] = {}
        self.lm_head = ColumnParallelLinear(
            text_config.hidden_size,
            text_config.vocab_size,
            bias=False,
            gather_output=False,
            perform_initialization=False,
        )
        self.sampler = Sampler(text_config.vocab_size)
        self.mtp = (
            Qwen3_5MultiTokenPredictor(text_config)
            if getattr(text_config, "mtp_num_hidden_layers", 0)
            else None
        )

    def _encode_multimodal_inputs(
        self,
        seq_id: int,
        input_metadata: InputMetadata,
    ) -> Dict[int, torch.Tensor]:
        cached = self._multimodal_feature_cache.get(seq_id)
        if cached is not None:
            return cached
        if self.visual is None:
            raise ValueError("Multimodal inputs require a loaded vision tower")
        multimodal = input_metadata.multimodal_inputs[seq_id]
        device = self.model.embed_tokens.weight.device
        features: Dict[int, torch.Tensor] = {}
        if multimodal.pixel_values is not None:
            image_grid = torch.tensor(
                multimodal.image_grid_thw,
                dtype=torch.long,
                device=device,
            )
            features[1] = self.visual(
                multimodal.pixel_values.to(device=device), image_grid
            )
        if multimodal.pixel_values_videos is not None:
            video_grid = torch.tensor(
                multimodal.video_grid_thw,
                dtype=torch.long,
                device=device,
            )
            features[2] = self.visual(
                multimodal.pixel_values_videos.to(device=device), video_grid
            )
        expected_counts = {
            modality: multimodal.token_type_ids.count(modality)
            for modality in (1, 2)
        }
        for modality, expected in expected_counts.items():
            actual = features[modality].shape[0] if modality in features else 0
            if actual != expected:
                name = "Image" if modality == 1 else "Video"
                raise ValueError(
                    f"{name} features and placeholder tokens do not match: "
                    f"features={actual}, tokens={expected}"
                )
        self._multimodal_feature_cache[seq_id] = features
        return features

    def _embed_inputs(
        self,
        input_ids: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        inputs_embeds = self.model.embed_tokens(input_ids)
        # The first prompt chunk carries the full processor output. Encode it
        # eagerly so later chunks and decode steps only need position metadata.
        for seq_id, multimodal in input_metadata.multimodal_inputs.items():
            if (
                multimodal.pixel_values is not None
                or multimodal.pixel_values_videos is not None
            ):
                self._encode_multimodal_inputs(seq_id, input_metadata)
        for packed_index, seq_id, modality, feature_index in (
            input_metadata.multimodal_token_maps
        ):
            expected_token_id = (
                self.image_token_id if modality == 1 else self.video_token_id
            )
            if (
                expected_token_id is not None
                and int(input_ids[packed_index]) != expected_token_id
            ):
                raise ValueError(
                    "Multimodal token type does not match the configured "
                    "image/video placeholder token"
                )
            features = self._encode_multimodal_inputs(
                seq_id, input_metadata
            )
            inputs_embeds[packed_index] = features[modality][
                feature_index
            ].to(inputs_embeds.dtype)
        return inputs_embeds

    def copy_multimodal_cache(self, parent_seq_id: int, child_seq_id: int) -> None:
        cached = self._multimodal_feature_cache.get(parent_seq_id)
        if cached is not None:
            self._multimodal_feature_cache[child_seq_id] = cached

    def release_multimodal_cache(self, seq_ids: Sequence[int]) -> None:
        for seq_id in seq_ids:
            self._multimodal_feature_cache.pop(seq_id, None)

    def _compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = F.linear(hidden_states, self.lm_head.weight)
        logits = gather_from_tensor_model_parallel_region(logits)
        return logits[:, :self.config.vocab_size].float()

    @staticmethod
    def _token_logprobs(
        logprobs: torch.Tensor,
        token_id: int,
        num_logprobs: Optional[int],
    ) -> Dict[int, float]:
        result: Dict[int, float] = {}
        if num_logprobs:
            values, indices = torch.topk(logprobs, num_logprobs)
            result.update({
                int(index): float(value)
                for index, value in zip(indices.tolist(), values.tolist())
            })
        result[token_id] = float(logprobs[token_id])
        return result

    def _is_eos(self, token_id: int) -> bool:
        eos_token_id = getattr(self.config, "eos_token_id", None)
        if isinstance(eos_token_id, (list, tuple, set)):
            return token_id in eos_token_id
        return eos_token_id is not None and token_id == eos_token_id

    @staticmethod
    def _mtp_params_are_supported(sampling_params) -> bool:
        return (
            sampling_params.temperature == 0.0
            and sampling_params.best_of == 1
            and not sampling_params.use_beam_search
            and sampling_params.presence_penalty == 0.0
            and sampling_params.frequency_penalty == 0.0
            and not sampling_params.stop
        )

    @staticmethod
    def _position_at(
        positions: torch.Tensor,
        token_index: int,
    ) -> torch.Tensor:
        return (
            positions[token_index]
            if positions.ndim == 1
            else positions[:, token_index]
        )

    def _verify_speculative_tokens(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> Tuple[
        Dict[int, SequenceOutputs],
        List[Tuple[SequenceOutputs, torch.Tensor, torch.Tensor, object]],
    ]:
        outputs: Dict[int, SequenceOutputs] = {}
        proposal_contexts = []
        for seq_id, draft_token_ids, indices, sampling_params in zip(
            input_metadata.speculative_seq_ids,
            input_metadata.speculative_token_blocks,
            input_metadata.speculative_hidden_indices,
            input_metadata.speculative_sampling_params,
        ):
            block_logits = self._compute_logits(
                hidden_states[list(indices)]
            )
            block_logprobs = F.log_softmax(block_logits, dim=-1)
            verification = verify_greedy_block(
                block_logits,
                draft_token_ids,
                is_eos=self._is_eos,
                ignore_eos=sampling_params.ignore_eos,
            )
            token_ids = list(verification.token_ids)
            token_logprobs = [
                self._token_logprobs(
                    block_logprobs[logit_index],
                    token_id,
                    sampling_params.logprobs,
                )
                for token_id, logit_index in zip(
                    verification.token_ids, verification.logit_indices
                )
            ]
            selected_index = indices[verification.last_committed_index]
            selected_hidden = hidden_states[selected_index]
            selected_position = self._position_at(positions, selected_index)

            output = SequenceOutputs(
                seq_id=seq_id,
                parent_seq_id=seq_id,
                output_token=token_ids[0],
                logprobs=token_logprobs[0],
                output_token_ids=token_ids,
                output_logprobs=token_logprobs,
                num_computed_tokens=verification.committed_tokens,
            )
            outputs[seq_id] = output
            final_token_id = token_ids[-1]
            if (
                sampling_params.ignore_eos
                or not self._is_eos(final_token_id)
            ):
                proposal_contexts.append((
                    output,
                    selected_hidden,
                    selected_position,
                    sampling_params,
                ))
        return outputs, proposal_contexts

    def _standard_proposal_contexts(
        self,
        outputs: Dict[int, SequenceOutputs],
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> List[Tuple[SequenceOutputs, torch.Tensor, torch.Tensor, object]]:
        contexts = []
        decode_offset = input_metadata.num_prompt_tokens
        for group_idx, (seq_ids, sampling_params) in enumerate(
            input_metadata.seq_groups
        ):
            if group_idx < input_metadata.num_prompt_samples:
                hidden_idx = input_metadata.prompt_sample_indices[group_idx]
                hidden_indices = [hidden_idx] * len(seq_ids)
            else:
                hidden_indices = list(range(
                    decode_offset, decode_offset + len(seq_ids)
                ))
                decode_offset += len(seq_ids)
            if not self._mtp_params_are_supported(sampling_params):
                continue
            for seq_id, hidden_idx in zip(seq_ids, hidden_indices):
                output = outputs[seq_id]
                if (
                    not sampling_params.ignore_eos
                    and self._is_eos(output.output_token)
                ):
                    continue
                contexts.append((
                    output,
                    hidden_states[hidden_idx],
                    self._position_at(positions, hidden_idx),
                    sampling_params,
                ))
        return contexts

    def _attach_mtp_drafts(
        self,
        contexts: List[Tuple[SequenceOutputs, torch.Tensor, torch.Tensor, object]],
    ) -> None:
        if self.mtp is None or not contexts:
            return
        token_ids = torch.tensor(
            [context[0].output_token_ids[-1] for context in contexts],
            dtype=torch.long,
            device=contexts[0][1].device,
        )
        target_hidden = torch.stack([context[1] for context in contexts])
        next_positions = torch.stack([context[2] for context in contexts]) + 1
        if next_positions.ndim == 2:
            next_positions = next_positions.transpose(0, 1).contiguous()
        mtp_hidden = self.mtp(
            self.model.embed_tokens(token_ids),
            next_positions,
            target_hidden,
        )
        draft_logits = self._compute_logits(mtp_hidden)
        draft_token_ids = torch.argmax(draft_logits, dim=-1).tolist()
        for context, draft_token_id in zip(contexts, draft_token_ids):
            context[0].set_draft_tokens([int(draft_token_id)])

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Sequence[Optional[Tuple[torch.Tensor, torch.Tensor]]]
        | HybridCache,
        input_metadata: InputMetadata,
        cache_events: Optional[Sequence[Optional[torch.cuda.Event]]],
    ) -> Dict[int, SequenceOutputs]:
        inputs_embeds = self._embed_inputs(input_ids, input_metadata)
        hidden_states = self.model(
            input_ids,
            positions,
            kv_caches,
            input_metadata,
            cache_events,
            inputs_embeds=inputs_embeds,
        )
        outputs: Dict[int, SequenceOutputs] = {}
        proposal_contexts = []
        if input_metadata.seq_groups:
            outputs.update(self.sampler(
                self.lm_head.weight,
                hidden_states,
                input_metadata,
            ))
            proposal_contexts.extend(self._standard_proposal_contexts(
                outputs, hidden_states, positions, input_metadata
            ))
        if input_metadata.speculative_seq_ids:
            speculative_outputs, speculative_contexts = (
                self._verify_speculative_tokens(
                    hidden_states, positions, input_metadata
                )
            )
            outputs.update(speculative_outputs)
            proposal_contexts.extend(speculative_contexts)
        if input_metadata.enable_mtp:
            self._attach_mtp_drafts(proposal_contexts)
        return outputs

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        use_np_cache: bool = False,
    ) -> None:
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        tensor_model_parallel_world_size = (
            get_tensor_model_parallel_world_size()
        )
        q_size = (
            self.config.num_attention_heads
            // tensor_model_parallel_world_size
            * self.config.head_dim
        )
        kv_size = (
            self.config.num_key_value_heads
            // tensor_model_parallel_world_size
            * self.config.head_dim
        )
        quant_config = getattr(self.config, "quantization_config", None)
        if isinstance(quant_config, dict):
            weight_block_size = quant_config.get("weight_block_size")
        else:
            weight_block_size = getattr(
                quant_config, "weight_block_size", None
            )
        block_n = weight_block_size[0] if weight_block_size else None

        state_dict = self.state_dict()
        column_parallel_weights = [
            "embed_tokens.weight",
            "lm_head.weight",
            "qkv_gate_proj.weight",
            "gate_up_proj.weight",
            "in_proj_qkv.weight",
            "in_proj_z.weight",
            "in_proj_a.weight",
            "in_proj_b.weight",
            "linear_attn.dt_bias",
            "linear_attn.A_log",
            "mtp.fc.weight",
        ]
        row_parallel_weights = [
            "self_attn.o_proj.weight",
            "linear_attn.out_proj.weight",
            "mlp.down_proj.weight",
        ]

        for checkpoint_name, loaded_value in hf_model_weights_iterator(
            model_name_or_path,
            cache_dir,
            use_np_cache,
        ):
            if checkpoint_name.startswith("model.visual."):
                name = "visual." + checkpoint_name.removeprefix(
                    "model.visual."
                )
            elif checkpoint_name.startswith("model.language_model."):
                name = "model." + checkpoint_name.removeprefix(
                    "model.language_model."
                )
            elif checkpoint_name.startswith("language_model."):
                name = "model." + checkpoint_name.removeprefix(
                    "language_model."
                )
            else:
                name = checkpoint_name

            if "rotary_emb.inv_freq" in name:
                continue

            is_gdn_qkv = ".linear_attn.in_proj_qkv." in name
            is_gdn_conv = name.endswith(".linear_attn.conv1d.weight")
            if is_gdn_qkv or is_gdn_conv:
                try:
                    param = state_dict[name]
                except KeyError as exc:
                    raise KeyError(
                        f"No target parameter for checkpoint weight {name}"
                    ) from exc
                is_scale = name.endswith("weight_scale_inv")
                _load_gdn_qkv_shard(
                    param,
                    loaded_value,
                    tensor_model_parallel_rank,
                    tensor_model_parallel_world_size,
                    self.config.linear_num_key_heads
                    * self.config.linear_key_head_dim,
                    self.config.linear_num_value_heads
                    * self.config.linear_value_head_dim,
                    block_n=block_n if is_scale else None,
                )
                continue

            attention_shards = (
                (".q_proj.", "q"),
                (".k_proj.", "k"),
                (".v_proj.", "v"),
            )
            attention_match = next(
                (
                    (source, shard_id)
                    for source, shard_id in attention_shards
                    if source in name
                ),
                None,
            )
            if attention_match is not None:
                source, shard_id = attention_match
                target_name = name.replace(source, ".qkv_gate_proj.")
                try:
                    param = state_dict[target_name]
                except KeyError as exc:
                    raise KeyError(
                        f"No target parameter for checkpoint weight {name}"
                    ) from exc
                if name.endswith("weight_scale_inv"):
                    if block_n is None:
                        raise ValueError(
                            "FP8 QKV scales require weight_block_size"
                        )
                    _load_qkv_gate_scale(
                        param,
                        loaded_value,
                        shard_id,
                        tensor_model_parallel_rank,
                        q_size,
                        kv_size,
                        block_n,
                    )
                else:
                    _load_qkv_gate_weight(
                        param,
                        loaded_value,
                        shard_id,
                        tensor_model_parallel_rank,
                        q_size,
                        kv_size,
                    )
                continue

            mlp_shards = ((".gate_proj.", 0), (".up_proj.", 1))
            mlp_match = next(
                (
                    (source, shard_idx)
                    for source, shard_idx in mlp_shards
                    if source in name
                ),
                None,
            )
            if mlp_match is not None:
                source, shard_idx = mlp_match
                target_name = name.replace(source, ".gate_up_proj.")
                try:
                    param = state_dict[target_name]
                except KeyError as exc:
                    raise KeyError(
                        f"No target parameter for checkpoint weight {name}"
                    ) from exc
                _load_merged_column_shard(
                    param,
                    loaded_value,
                    shard_idx,
                    tensor_model_parallel_rank,
                )
                continue

            try:
                param = state_dict[name]
            except KeyError as exc:
                raise KeyError(
                    f"Unexpected Qwen checkpoint weight {checkpoint_name}"
                ) from exc
            load_tensor_parallel_weights(
                param=param,
                loaded_weight=loaded_value,
                param_name=name,
                column_parallel_weight_name=column_parallel_weights,
                row_parallel_weight_name=row_parallel_weights,
                tensor_model_parallel_rank=tensor_model_parallel_rank,
            )


__all__ = [
    "Qwen3_5Attention",
    "Qwen3_5DecoderLayer",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5GatedDeltaNet",
    "Qwen3_5MLP",
    "Qwen3_5Model",
    "_causal_conv1d_prefill",
    "_load_qkv_gate_weight",
    "_load_qkv_gate_scale",
    "_load_merged_column_shard",
    "_load_gdn_qkv_shard",
    "_split_q_gate_kv",
]
