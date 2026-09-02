"""DFlash backbone and DSpark heads for Qwen3.8 speculative decoding."""

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from minivllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from minivllm.model_executor.layers.dspark_attention import DraftPagedAttention
from minivllm.spec_decode.draft_metadata import DraftAttentionMetadata
from minivllm.model_executor.parallel_utils.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from minivllm.model_executor.weight_utils import (
    hf_model_weights_iterator,
    load_tensor_parallel_weights,
)
from minivllm.spec_decode.dspark_config import DSparkConfig
from minivllm.spec_decode.dspark_heads import (
    DSparkConfidenceHead,
    StepSampler,
    VanillaMarkov,
)


class Qwen3RMSNorm(nn.Module):
    """Standard Qwen3 RMSNorm whose checkpoint weight is not zero-centered."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_fp32 = hidden_states.float()
        variance = hidden_fp32.square().mean(dim=-1, keepdim=True)
        normalized = hidden_fp32 * torch.rsqrt(variance + self.variance_epsilon)
        return (normalized * self.weight.float()).to(input_dtype)


def _apply_neox_rope(
    tensor: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> torch.Tensor:
    """Apply GPT-NeoX rotary embedding to [batch, tokens, heads, dim]."""
    if tensor.ndim != 4 or positions.shape != tensor.shape[:2]:
        raise ValueError("RoPE expects tensor [batch, tokens, heads, dim]")
    head_dim = tensor.shape[-1]
    if head_dim % 2:
        raise ValueError("DSpark head_dim must be even")
    selected = cos_sin_cache[positions.long()]
    half = head_dim // 2
    cos = selected[..., :half].unsqueeze(2).to(tensor.dtype)
    sin = selected[..., half:].unsqueeze(2).to(tensor.dtype)
    first, second = tensor[..., :half], tensor[..., half:]
    return torch.cat((first * cos - second * sin, second * cos + first * sin), -1)


def _repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return hidden_states
    return hidden_states.repeat_interleave(repeats, dim=2)


def dual_source_attention_reference(
    query: torch.Tensor,
    context_key: torch.Tensor,
    context_value: torch.Tensor,
    block_key: torch.Tensor,
    block_value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Dense correctness reference for DFlash dual-source attention.

    Historical target features form the prefix. Every query in the current
    draft block can see that prefix and the entire current block, including
    positions to its right.
    """
    if query.ndim != 4:
        raise ValueError("query must have shape [batch, block, heads, dim]")
    tensors = (context_key, context_value, block_key, block_value)
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError("DFlash key/value tensors must be four-dimensional")
    num_heads = query.shape[2]
    num_kv_heads = block_key.shape[2]
    if num_heads % num_kv_heads:
        raise ValueError("Query heads must be grouped evenly by KV heads")
    if context_key.shape[2:] != block_key.shape[2:]:
        raise ValueError("Context and block keys must use the same head layout")
    if context_value.shape != context_key.shape or block_value.shape != block_key.shape:
        raise ValueError("DSpark key and value shapes must match")

    repeats = num_heads // num_kv_heads
    key = _repeat_kv(torch.cat((context_key, block_key), dim=1), repeats)
    value = _repeat_kv(torch.cat((context_value, block_value), dim=1), repeats)
    output = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
        scale=float(scale),
    )
    return output.transpose(1, 2)


class DFlashAttention(nn.Module):
    """Standard Qwen3 GQA projection with DFlash dual-source attention."""

    def __init__(self, config, *, use_cpu_initialization: bool = False) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = int(config.num_attention_heads)
        self.total_num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim)
        if self.total_num_heads % tp_size or self.total_num_kv_heads % tp_size:
            raise ValueError("DSpark attention heads must be divisible by TP size")
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        if self.num_heads % self.num_kv_heads:
            raise ValueError("DSpark query heads must be grouped by KV heads")
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        output_size = (
            self.total_num_heads + 2 * self.total_num_kv_heads
        ) * self.head_dim
        self.qkv_proj = ColumnParallelLinear(
            config.hidden_size,
            output_size,
            bias=bool(getattr(config, "attention_bias", False)),
            gather_output=False,
            perform_initialization=False,
            use_cpu_initialization=use_cpu_initialization,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=bool(getattr(config, "attention_bias", False)),
            input_is_parallel=True,
            perform_initialization=False,
            use_cpu_initialization=use_cpu_initialization,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, config.rms_norm_eps)
        self.paged_attention = DraftPagedAttention(
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.scaling,
        )

    def _project_qkv(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        packed, _ = self.qkv_proj(hidden_states)
        query, key, value = packed.split(
            (self.q_size, self.kv_size, self.kv_size), dim=-1
        )
        query = self.q_norm(query.view(*query.shape[:-1], self.num_heads, self.head_dim))
        key = self.k_norm(key.view(*key.shape[:-1], self.num_kv_heads, self.head_dim))
        value = value.view(*value.shape[:-1], self.num_kv_heads, self.head_dim)
        return query, key, value

    def _project_context_kv(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        packed, _ = self.qkv_proj(hidden_states)
        _, key, value = packed.split(
            (self.q_size, self.kv_size, self.kv_size), dim=-1
        )
        key = self.k_norm(key.view(*key.shape[:-1], self.num_kv_heads, self.head_dim))
        value = value.view(*value.shape[:-1], self.num_kv_heads, self.head_dim)
        return key, value

    def forward_reference(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        context_hidden: torch.Tensor,
        context_positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
    ) -> torch.Tensor:
        query, block_key, block_value = self._project_qkv(hidden_states)
        context_key, context_value = self._project_context_kv(context_hidden)
        query = _apply_neox_rope(query, positions, cos_sin_cache)
        block_key = _apply_neox_rope(block_key, positions, cos_sin_cache)
        context_key = _apply_neox_rope(
            context_key, context_positions, cos_sin_cache
        )
        output = dual_source_attention_reference(
            query,
            context_key,
            context_value,
            block_key,
            block_value,
            self.scaling,
        )
        output, _ = self.o_proj(output.flatten(-2))
        return output

    def project_context_kv(
        self,
        context_hidden: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key, value = self._project_context_kv(context_hidden)
        key = _apply_neox_rope(
            key.unsqueeze(0), positions.unsqueeze(0), cos_sin_cache
        ).squeeze(0)
        return key, value

    def forward_paged(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        cache: Tuple[torch.Tensor, torch.Tensor],
        metadata: DraftAttentionMetadata,
        cos_sin_cache: torch.Tensor,
    ) -> torch.Tensor:
        query, key, value = self._project_qkv(hidden_states)
        query = _apply_neox_rope(
            query.unsqueeze(0), positions.unsqueeze(0), cos_sin_cache
        ).squeeze(0)
        key = _apply_neox_rope(
            key.unsqueeze(0), positions.unsqueeze(0), cos_sin_cache
        ).squeeze(0)
        output = self.paged_attention(query, key, value, cache, metadata)
        output, _ = self.o_proj(output.flatten(-2))
        return output


class DFlashMLP(nn.Module):
    def __init__(self, config, *, use_cpu_initialization: bool = False) -> None:
        super().__init__()
        self.intermediate_size = int(config.intermediate_size)
        self.gate_up_proj = ColumnParallelLinear(
            config.hidden_size,
            2 * self.intermediate_size,
            bias=False,
            gather_output=False,
            perform_initialization=False,
            use_cpu_initialization=use_cpu_initialization,
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            perform_initialization=False,
            use_cpu_initialization=use_cpu_initialization,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        output, _ = self.down_proj(F.silu(gate) * up)
        return output


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config, *, use_cpu_initialization: bool = False) -> None:
        super().__init__()
        self.input_layernorm = Qwen3RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.self_attn = DFlashAttention(
            config, use_cpu_initialization=use_cpu_initialization
        )
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.mlp = DFlashMLP(
            config, use_cpu_initialization=use_cpu_initialization
        )

    def forward_reference(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        context_hidden: torch.Tensor,
        context_positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn.forward_reference(
            hidden_states,
            positions,
            context_hidden,
            context_positions,
            cos_sin_cache,
        )
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)

    def forward_paged(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        cache: Tuple[torch.Tensor, torch.Tensor],
        metadata: DraftAttentionMetadata,
        cos_sin_cache: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn.forward_paged(
            hidden_states, positions, cache, metadata, cos_sin_cache
        )
        residual = hidden_states
        return residual + self.mlp(self.post_attention_layernorm(hidden_states))


@dataclass
class DSparkProposal:
    token_ids: torch.Tensor
    draft_logits: torch.Tensor
    confidence: Optional[torch.Tensor]
    hidden_states: torch.Tensor


class DSparkDraftModel(nn.Module):
    """Published Qwen3.8 DSpark drafter without duplicated embedding/LM head."""

    def __init__(self, config, *, use_cpu_initialization: bool = False) -> None:
        super().__init__()
        self.hf_config = config
        self.config = DSparkConfig.from_hf_config(config)
        self.layers = nn.ModuleList([
            DFlashDecoderLayer(
                config, use_cpu_initialization=use_cpu_initialization
            )
            for _ in range(self.config.num_hidden_layers)
        ])
        self.norm = Qwen3RMSNorm(self.config.hidden_size, self.config.rms_norm_eps)
        self.fc = nn.Linear(
            len(self.config.target_layer_ids) * self.config.hidden_size,
            self.config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Qwen3RMSNorm(
            self.config.hidden_size, self.config.rms_norm_eps
        )
        self.markov_head = VanillaMarkov(
            self.config.vocab_size, self.config.markov_rank
        )
        self.confidence_head = (
            DSparkConfidenceHead(
                self.config.hidden_size,
                self.config.markov_rank,
                with_markov=self.config.confidence_head_with_markov,
                max_block_size=self.config.block_size,
            )
            if self.config.enable_confidence_head
            else None
        )
        inv_freq = 1.0 / (
            self.config.rope_theta
            ** (torch.arange(0, self.config.head_dim, 2).float() / self.config.head_dim)
        )
        positions = torch.arange(self.config.max_position_embeddings).float()
        frequencies = torch.outer(positions, inv_freq)
        self.register_buffer(
            "cos_sin_cache",
            torch.cat((frequencies.cos(), frequencies.sin()), dim=-1),
            persistent=False,
        )

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        expected = len(self.config.target_layer_ids) * self.config.hidden_size
        if target_hidden.ndim not in (2, 3) or target_hidden.shape[-1] != expected:
            raise ValueError(
                "DSpark target hidden states must end with "
                f"{expected} features, got {tuple(target_hidden.shape)}"
            )
        return self.hidden_norm(self.fc(target_hidden.to(self.fc.weight.dtype)))

    def forward_reference(
        self,
        input_embeddings: torch.Tensor,
        positions: torch.Tensor,
        context_hidden: torch.Tensor,
        context_positions: torch.Tensor,
    ) -> torch.Tensor:
        if input_embeddings.ndim != 3:
            raise ValueError("DSpark input embeddings must be [batch, block, hidden]")
        if input_embeddings.shape[1] != self.config.block_size:
            raise ValueError(
                f"DSpark block must contain {self.config.block_size} positions"
            )
        hidden_states = input_embeddings
        for layer in self.layers:
            hidden_states = layer.forward_reference(
                hidden_states,
                positions,
                context_hidden,
                context_positions,
                self.cos_sin_cache,
            )
        return self.norm(hidden_states)

    def materialize_context_kv(
        self,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        caches: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Project committed target features into every draft-layer cache."""
        context_hidden = self.project_target_hidden(target_hidden)
        cache_list = list(caches)
        if len(cache_list) != len(self.layers):
            raise ValueError("Draft cache layer count does not match the model")
        for layer, cache in zip(self.layers, cache_list):
            key, value = layer.self_attn.project_context_kv(
                context_hidden, positions, self.cos_sin_cache
            )
            layer.self_attn.paged_attention.write_context(
                key, value, cache, slot_mapping
            )
        return context_hidden

    def forward_paged(
        self,
        input_embeddings: torch.Tensor,
        positions: torch.Tensor,
        caches: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        metadata: DraftAttentionMetadata,
    ) -> torch.Tensor:
        if input_embeddings.shape != (
            metadata.num_tokens,
            self.config.hidden_size,
        ):
            raise ValueError("Packed DSpark embeddings do not match metadata")
        cache_list = list(caches)
        if len(cache_list) != len(self.layers):
            raise ValueError("Draft cache layer count does not match the model")
        hidden_states = input_embeddings
        for layer, cache in zip(self.layers, cache_list):
            hidden_states = layer.forward_paged(
                hidden_states,
                positions,
                cache,
                metadata,
                self.cos_sin_cache,
            )
        return self.norm(hidden_states)

    def propose_reference(
        self,
        input_embeddings: torch.Tensor,
        positions: torch.Tensor,
        context_hidden: torch.Tensor,
        context_positions: torch.Tensor,
        lm_head_weight: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        sampler: Optional[StepSampler] = None,
    ) -> DSparkProposal:
        hidden_states = self.forward_reference(
            input_embeddings, positions, context_hidden, context_positions
        )
        if lm_head_weight.shape != (
            self.config.vocab_size,
            self.config.hidden_size,
        ):
            raise ValueError("Target LM head is incompatible with DSpark checkpoint")
        base_logits = F.linear(hidden_states.to(lm_head_weight.dtype), lm_head_weight)
        markov = self.markov_head.sample_block(
            base_logits, anchor_token_ids, sampler
        )
        confidence = None
        if self.confidence_head is not None:
            confidence = self.confidence_head.probabilities(
                hidden_states, markov.previous_embeddings
            )
        return DSparkProposal(
            token_ids=markov.token_ids,
            draft_logits=markov.logits,
            confidence=confidence,
            hidden_states=hidden_states,
        )

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        use_np_cache: bool = False,
    ) -> None:
        self.load_weight_iterator(hf_model_weights_iterator(
            model_name_or_path, cache_dir, use_np_cache
        ))

    def load_weight_iterator(
        self, weights: Iterable[Tuple[str, torch.Tensor]]
    ) -> None:
        state_dict = self.state_dict()
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        q_size = self.config.num_attention_heads // tp_size * self.config.head_dim
        kv_size = self.config.num_key_value_heads // tp_size * self.config.head_dim
        column_weights = ["gate_up_proj.weight"]
        row_weights = ["o_proj.weight", "down_proj.weight"]
        aliases = {
            "encoder.fc.weight": "fc.weight",
            "encoder.output_norm_enc.weight": "hidden_norm.weight",
        }

        for checkpoint_name, loaded_weight in weights:
            name = checkpoint_name.removeprefix("model.")
            name = aliases.get(name, name)
            if "rotary_emb" in name or name.startswith(("embed_tokens.", "lm_head.")):
                continue
            qkv_match = next(
                ((source, shard) for source, shard in (
                    (".q_proj.", "q"),
                    (".k_proj.", "k"),
                    (".v_proj.", "v"),
                ) if source in name),
                None,
            )
            if qkv_match is not None:
                source, shard = qkv_match
                target_name = name.replace(source, ".qkv_proj.")
                if not target_name.endswith(".weight"):
                    continue
                _load_draft_qkv_weight(
                    state_dict[target_name], loaded_weight, shard,
                    tp_rank, q_size, kv_size,
                )
                continue
            mlp_match = next(
                ((source, index) for source, index in (
                    (".gate_proj.", 0), (".up_proj.", 1)
                ) if source in name),
                None,
            )
            if mlp_match is not None:
                source, index = mlp_match
                target_name = name.replace(source, ".gate_up_proj.")
                _load_merged_draft_weight(
                    state_dict[target_name], loaded_weight, index, tp_rank
                )
                continue
            if name not in state_dict:
                raise KeyError(
                    f"Unexpected DSpark checkpoint weight {checkpoint_name!r}"
                )
            load_tensor_parallel_weights(
                state_dict[name], loaded_weight, name,
                column_weights, row_weights, tp_rank,
            )


def _load_draft_qkv_weight(
    parameter: torch.Tensor,
    loaded_weight: torch.Tensor,
    shard: str,
    tp_rank: int,
    q_size: int,
    kv_size: int,
) -> None:
    offsets = {"q": (0, q_size), "k": (q_size, kv_size), "v": (q_size + kv_size, kv_size)}
    if shard not in offsets:
        raise ValueError(f"Unknown QKV shard {shard!r}")
    target_start, local_size = offsets[shard]
    source_start = tp_rank * local_size
    source = loaded_weight[source_start:source_start + local_size]
    target = parameter.data[target_start:target_start + local_size]
    if target.shape != source.shape:
        raise ValueError(
            f"DSpark {shard} shard shape {source.shape} does not match {target.shape}"
        )
    target.copy_(source)


def _load_merged_draft_weight(
    parameter: torch.Tensor,
    loaded_weight: torch.Tensor,
    shard_index: int,
    tp_rank: int,
) -> None:
    local_size = parameter.shape[0] // 2
    source_start = tp_rank * local_size
    source = loaded_weight[source_start:source_start + local_size]
    target = parameter.data[
        shard_index * local_size:(shard_index + 1) * local_size
    ]
    if target.shape != source.shape:
        raise ValueError("DSpark merged MLP shard shape mismatch")
    target.copy_(source)


__all__ = [
    "DFlashAttention",
    "DFlashDecoderLayer",
    "DSparkDraftModel",
    "DSparkProposal",
    "Qwen3RMSNorm",
    "dual_source_attention_reference",
]
