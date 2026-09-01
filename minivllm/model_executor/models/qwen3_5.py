"""Qwen3.5/3.8 hybrid text decoder used by the mini-vLLM runtime.

The checkpoint alternates stateful Gated DeltaNet token mixers with gated
full-attention layers. Packed prompts and decode tokens share one residual
stream, while :class:`HybridCache` provides the different cache type required
by each layer.
"""

from typing import Dict, Optional, Sequence, Tuple

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
from minivllm.model_executor.layers.attention import PagedAttentionWithRoPE
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
    get_tensor_model_parallel_world_size,
)
from minivllm.model_executor.parallel_utils.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from minivllm.sequence import SequenceOutputs
from minivllm.worker.hybrid_cache import HybridCache

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
        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is None:
            rope_parameters = getattr(config, "rope_parameters", {}) or {}
            rope_theta = rope_parameters.get("rope_theta", 10_000.0)
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
        self.gate_up_proj = ColumnParallelLinear(
            config.hidden_size,
            2 * config.intermediate_size,
            bias=False,
            gather_output=False,
            perform_initialization=False,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            perform_initialization=False,
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
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
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

        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim

        self.in_proj_qkv = ColumnParallelLinear(
            self.hidden_size,
            self.conv_dim,
            bias=False,
            gather_output=False,
            perform_initialization=False,
        )
        self.in_proj_z = ColumnParallelLinear(
            self.hidden_size,
            self.value_dim,
            bias=False,
            gather_output=False,
            perform_initialization=False,
        )
        self.in_proj_b = ColumnParallelLinear(
            self.hidden_size,
            self.num_v_heads,
            bias=False,
            gather_output=False,
            perform_initialization=False,
        )
        self.in_proj_a = ColumnParallelLinear(
            self.hidden_size,
            self.num_v_heads,
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
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            perform_initialization=False,
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

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
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
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
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
        return self.norm(hidden_states)


class Qwen3_5ForConditionalGeneration(nn.Module):
    """Text generation entry point selected by the multimodal architecture."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3_5Model(config)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            gather_output=False,
            perform_initialization=False,
        )
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Sequence[Optional[Tuple[torch.Tensor, torch.Tensor]]]
        | HybridCache,
        input_metadata: InputMetadata,
        cache_events: Optional[Sequence[Optional[torch.cuda.Event]]],
    ) -> Dict[int, SequenceOutputs]:
        hidden_states = self.model(
            input_ids,
            positions,
            kv_caches,
            input_metadata,
            cache_events,
        )
        return self.sampler(
            self.lm_head.weight,
            hidden_states,
            input_metadata,
        )

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        use_np_cache: bool = False,
    ) -> None:
        raise NotImplementedError(
            "Qwen checkpoint loading is implemented in the safetensors/FP8 stage"
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
    "_split_q_gate_kv",
]
