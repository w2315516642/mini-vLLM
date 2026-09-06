"""Qwen3.5/3.8 text runtime and stage 8 hybrid-model assignments.

The completed attention and state primitives are assembled here. Stage 8
learner TODOs own decoder execution, CUDA GDN integration and weight mapping.
"""

from typing import Iterable, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn

from minivllm.model_executor.input_metadata import InputMetadata
from minivllm.model_executor.layers.activation import SigmoidAndMul
from minivllm.model_executor.layers.attention import PagedAttentionWithRoPE
from minivllm.model_executor.layers.layer_norm import Qwen3_5RMSNorm
from minivllm.configs.model_architecture import FULL_ATTENTION, LINEAR_ATTENTION
from minivllm.model_executor.layers.gated_delta_net import (
    GatedDeltaNetState, Qwen3_5GatedDeltaNetReference,
    causal_depthwise_conv1d_reference,
)
from minivllm.model_executor.layers.gated_delta_net_cuda import (
    causal_conv1d_update, gated_delta_rule_decode, gated_delta_rule_prefill,
    prepare_gated_delta_qk,
)
from minivllm.model_executor.layers.sampler import Sampler
from minivllm.model_executor.models.llama import LlamaMLP
from minivllm.model_executor.weight_utils import hf_model_weights_iterator
from minivllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size,
)
from minivllm.model_executor.parallel_utils.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
)

KVCache = Tuple[torch.Tensor, torch.Tensor]

if TYPE_CHECKING:
    from minivllm.worker.hybrid_cache import HybridCache


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

        rope = getattr(config, "rope_parameters", None) or {}
        partial_rotary_factor = rope.get(
            "partial_rotary_factor", getattr(config, "partial_rotary_factor", 1.0)
        )
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
            base=float(rope.get("rope_theta", getattr(config, "rope_theta", 10000.0))),
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
        if input_metadata.is_profile_run:
            # Profile activations and GDN state before paged KV is allocated.
            k_cache, v_cache = None, None
        attn_out = self.attn(
            positions, q, k, v, k_cache, v_cache, input_metadata, cache_event
        )
        gate_out = self.gate_fn(attn_out, gate)
        output, _ = self.o_proj(gate_out)
        return output


class Qwen3_5GatedDeltaNet(Qwen3_5GatedDeltaNetReference):
    """Stage 8 CUDA execution with the reference layer's checkpoint parameters."""

    def forward(
        self, hidden_states: torch.Tensor, state: GatedDeltaNetState,
    ) -> Tuple[torch.Tensor, GatedDeltaNetState]:
        """Consume [1,T,H] valid tokens; return output and FP32 updated state.

        T=1 uses the conv/decode CUDA wrappers. T>1 uses reference causal conv
        and the stage 6 prefill kernel. Prepare Q/K once before either kernel.
        """
        self._check_input(hidden_states, state)
        token = hidden_states.size(1)

        qkv = self.in_proj_qkv(hidden_states)
        gate = self.in_proj_z(hidden_states)
        beta = torch.sigmoid(self.in_proj_b(hidden_states))
        decay_input = self.in_proj_a(hidden_states)

        batch, sequence = qkv.shape[0], qkv.shape[1]
        if state is None:
            state = self.empty_state(batch, hidden_states.device)

        # short conv
        conv_weight = self.conv1d.weight.squeeze(1)

        if token == 1:
            qkv = qkv.squeeze(1)
            conv_qkv = causal_conv1d_update(
                qkv, state.conv_state, conv_weight)
            conv_state = state.conv_state
        else:
            conv_qkv, conv_state = causal_depthwise_conv1d_reference(
                qkv, conv_weight, state.conv_state)

        # get q k v
        query, key, value = conv_qkv.split(
            [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape([batch, sequence, self.num_k_heads, self.head_k_dim])
        key = key.reshape([batch, sequence, self.num_k_heads, self.head_k_dim])
        value = value.reshape([batch, sequence, self.num_v_heads, self.head_v_dim]).contiguous()
        if self.num_k_heads < self.num_v_heads:
            repeat_cnt = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeat_cnt, dim=2)
            key = key.repeat_interleave(repeat_cnt, dim=2)

        # g is the log-domain decay consumed by the recurrent reference.
        decay_rate = self.A_log.float().exp()
        bias = self.dt_bias.float().unsqueeze(0).unsqueeze(0)
        forget = nn.functional.softplus(decay_input.float() + bias)
        log_decay = -decay_rate * forget

        query, key = prepare_gated_delta_qk(query, key)
        if token == 1:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)
            log_decay = log_decay.squeeze(1)
            beta = beta.squeeze(1)
            new_value = gated_delta_rule_decode(
                query,
                key,
                value,
                log_decay,
                beta,
                state.recurrent_state,
            )
            new_value = new_value.unsqueeze(1)
        else:
            new_value = gated_delta_rule_prefill(
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
        new_state = GatedDeltaNetState(conv_state, state.recurrent_state)
        return output, new_state


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        # Expected names: self_attn OR linear_attn, mlp, input_layernorm,
        # post_attention_layernorm. These names also define checkpoint targets.
        if self.layer_type == FULL_ATTENTION:
            self.self_attn = Qwen3_5Attention(config)
        elif self.layer_type == LINEAR_ATTENTION:
            self.linear_attn = Qwen3_5GatedDeltaNet(config)
        else:
            raise ValueError(f"Layer type error: {self.layer_type}")

        self.mlp = LlamaMLP(config.hidden_size, config.intermediate_size, config.hidden_act)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor,
        hybrid_cache: "HybridCache", input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event] = None,
    ) -> torch.Tensor:
        """Apply pre-norm mixer/residual, then pre-norm MLP/residual.

        The linear branch gathers/writes input_metadata.state_slot_ids at this
        layer_idx. The full branch obtains its paged KV from hybrid_cache.
        """
        x = hidden_states
        x_norm = self.input_layernorm(hidden_states)

        if self.layer_type == FULL_ATTENTION:
            kv_cache = hybrid_cache.get_kv_cache(self.layer_idx)
            residual = self.self_attn(positions, x_norm, kv_cache, input_metadata, cache_event)
        else:
            slot_ids = input_metadata.state_slot_ids
            state = hybrid_cache.read_state(self.layer_idx, slot_ids)

            extend = False
            if x_norm.dim() == 2:
                extend = True
                x_norm = x_norm.unsqueeze(0)
            residual, new_state = self.linear_attn(x_norm, state)
            hybrid_cache.write_state(self.layer_idx, slot_ids, new_state)
            if extend:
                residual = residual.squeeze(0)

        x = x + residual

        hidden_states = x
        x_norm = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(x_norm)

        return x + mlp_out


class Qwen3_5Model(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3_5DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor,
        hybrid_cache: "HybridCache", input_metadata: InputMetadata,
        cache_events=None,
    ) -> torch.Tensor:
        """Embed valid tokens, run global layer order, and apply final norm.

        Alignment padding must never advance Conv/Recurrent State. Return
        [num_valid_tokens, hidden_size]; Sampler already understands that layout.
        """
        num_valid_tokens = input_metadata.num_valid_tokens
        input_ids = input_ids[:num_valid_tokens]
        positions = positions[:num_valid_tokens]

        hidden_states = self.embed_tokens(input_ids)
        for layer_idx, layer in enumerate(self.layers):
            if cache_events is None:
                cache_event = None
            else:
                cache_event = cache_events[layer_idx]
            hidden_states = layer(
                positions,
                hidden_states,
                hybrid_cache,
                input_metadata,
                cache_event
            )
        out = self.norm(hidden_states)
        return out


class Qwen3_5ForConditionalGeneration(nn.Module):
    """Text-only stage 8 entry point for the checkpoint's registered name."""

    def __init__(self, config) -> None:
        super().__init__()
        if get_tensor_model_parallel_world_size() != 1:
            raise ValueError("Stage 8 Qwen runtime requires tensor_parallel_size=1")
        self.config = config
        self.model = Qwen3_5Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self, input_ids, positions, kv_caches, input_metadata, cache_events=None,
        hybrid_cache=None,
    ):
        hidden_states = self.model(
            input_ids, positions, hybrid_cache, input_metadata, cache_events
        )
        return self.sampler(self.lm_head.weight, hidden_states, input_metadata)

    def load_weights(self, model_name_or_path, cache_dir=None, use_np_cache=False):
        self.load_weights_from_iterator(
            hf_model_weights_iterator(model_name_or_path, cache_dir, use_np_cache)
        )

    @torch.no_grad()
    def load_weights_from_iterator(
        self, weights: Iterable[Tuple[str, torch.Tensor]],
    ) -> None:
        """Map text checkpoint keys, load packed shards, and check coverage.

        Accept model.language_model.* or local model.* text names. Skip only
        model.visual.*, mtp.*, and rotary_emb.inv_freq. Track q/k/v and gate/up
        individually; tied lm_head does not need a separate checkpoint entry.
        Unknown text keys, duplicate shards, missing shards and bad shapes fail
        at startup. Reuse _load_qkv_gate_weight for the packed attention matrix.
        """
        # Keep the lm_head alias visible; tied embeddings still require their
        # own checkpoint entry, but do not require a separate lm_head entry.
        params = dict(self.named_parameters(remove_duplicate=False))
        routes = {}
        for name, param in params.items():
            if name.endswith(".self_attn.qkv_gate_proj.weight"):
                prefix = name.removesuffix(".qkv_gate_proj.weight")
                attention = self.get_submodule(prefix)
                q_rows, kv_rows = 2 * attention.q_size, attention.kv_size
                for shard, offset, size in (
                    ("q", 0, q_rows),
                    ("k", q_rows, kv_rows),
                    ("v", q_rows + kv_rows, kv_rows),
                ):
                    routes[f"{prefix}.{shard}_proj.weight"] = (
                        name, shard, offset, size)
            elif name.endswith(".mlp.gate_up_proj.weight"):
                prefix = name.removesuffix(".gate_up_proj.weight")
                size = param.shape[0] // 2
                for shard, offset in (("gate", 0), ("up", size)):
                    routes[f"{prefix}.{shard}_proj.weight"] = (
                        name, shard, offset, size)
            else:
                routes[name] = (name, None, None, None)

        # Coverage is tracked by source shard, not packed destination parameter.
        # Seeing Q alone must not make the whole QKV matrix appear loaded.
        required = set(routes)
        tied = self.config.tie_word_embeddings
        if tied:
            required.remove("lm_head.weight")
        loaded = set()
        aliases = {"lm_head.weight": "model.embed_tokens.weight",
                   "model.embed_tokens.weight": "lm_head.weight"}

        for source_name, loaded_weight in weights:
            if source_name.startswith(("model.visual.", "mtp.")):
                continue
            name = source_name
            if name.startswith("model.language_model."):
                name = "model." + name.removeprefix("model.language_model.")
            if name.endswith(".rotary_emb.inv_freq"):
                continue
            if name not in routes:
                raise ValueError(f"Unknown text checkpoint weight: {source_name!r}")
            if name in loaded:
                raise ValueError(f"Duplicate checkpoint weight or shard: {name!r}")

            target_name, shard, offset, size = routes[name]
            param = params[target_name]
            expected_shape = (param.shape if shard is None else
                              (size, *param.shape[1:]))
            # Check exact source size before the TP helper slices it. Otherwise
            # an oversized source could be silently truncated even at TP=1.
            if tuple(loaded_weight.shape) != tuple(expected_shape):
                raise ValueError(
                    f"Checkpoint weight {source_name!r} for {target_name!r} "
                    f"has shape {tuple(loaded_weight.shape)}; "
                    f"expected {tuple(expected_shape)}")

            if tied and name in aliases and aliases[name] in loaded:
                # Some checkpoints include both aliases. Check agreement in
                # destination precision instead of overwriting shared storage.
                if not torch.equal(param, loaded_weight.to(param)):
                    raise ValueError(
                        "Tied embedding and lm_head checkpoint weights disagree")
            elif shard in ("q", "k", "v"):
                attention = self.get_submodule(
                    target_name.removesuffix(".qkv_gate_proj.weight"))
                _load_qkv_gate_weight(
                    param, loaded_weight, shard, tensor_model_parallel_rank=0,
                    q_size=attention.q_size, kv_size=attention.kv_size)
            elif shard in ("gate", "up"):
                param.narrow(0, offset, size).copy_(loaded_weight)
            else:
                param.copy_(loaded_weight)
            loaded.add(name)

        missing = required - loaded
        if missing:
            raise ValueError(
                "Missing checkpoint weights or shards: " + ", ".join(sorted(missing)))


__all__ = [
    "Qwen3_5Attention",
    "Qwen3_5GatedDeltaNet",
    "Qwen3_5DecoderLayer",
    "Qwen3_5Model",
    "Qwen3_5ForConditionalGeneration",
    "_load_qkv_gate_weight",
    "_split_q_gate_kv",
]
