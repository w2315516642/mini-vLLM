from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
from transformers import LlamaConfig

from minivllm.sequence import SequenceOutputs
from minivllm.model_executor.input_metadata import InputMetadata
from minivllm.model_executor.layers.activation import SiluAndMul
from minivllm.model_executor.layers.layer_norm import RMSNorm
from minivllm.model_executor.layers.attention import PagedAttentionWithRoPE
from minivllm.model_executor.layers.sampler import Sampler
from minivllm.model_executor.weight_utils import (
    hf_model_weights_iterator, load_tensor_parallel_weights)
from minivllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from minivllm.model_executor.parallel_utils.tensor_parallel import (
    VocabParallelEmbedding, ColumnParallelLinear, RowParallelLinear)

KVCache = Tuple[torch.Tensor, torch.Tensor]

class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: int,
    ):
        super().__init__()

        self.gate_up_proj = ColumnParallelLinear(
            hidden_size, 2 * intermediate_size, bias=False,
            gather_output=False, perform_initialization=False
        )

        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=False,
            gather_output=False, perform_initialization=False
        )

        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class LlamaAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = self.total_num_heads // tensor_model_parallel_world_size
        self.head_dim = hidden_size // self.total_num_heads
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = ColumnParallelLinear(
            hidden_size,
            3 * self.num_heads * self.head_dim,
            bias=False,
            gather_output=False,
            perform_initialization=False,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            perform_initialization=False,
        )
        self.attn = PagedAttentionWithRoPE(
            self.num_heads, self.head_dim, self.scaling, rotary_dim=self.head_dim
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k ,v = qkv.chunk(chunks=3, dim=-1)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(
            positions, q, k, v, k_cache, v_cache, input_metadata, cache_event
        )
        output, _ = self.o_proj(attn_output)
        return output


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attn = LlamaAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event]
    ) -> torch.Tensor:
        # Attn
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            cache_event=cache_event
        )
        hidden_states = residual + hidden_states

        # FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, perform_initialization=False)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: KVCache,
        input_metadata: InputMetadata,
        cache_events: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for i in range(len(self.layers)):
            if cache_events is None:
                cache_event = None
            else:
                cache_event = cache_events[i]
            layer = self.layers[i]
            hidden_states = layer(
                positions,
                hidden_states,
                kv_caches[i],
                input_metadata,
                cache_event,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig):
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = ColumnParallelLinear(
            config.head_size,
            config.vocab_size,
            bias=False,
            gather_output=False,
            perform_initialization=False
        )
        self.sampler = Sampler(config.vocab_size)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: KVCache,
        input_metadata: InputMetadata,
        cache_events: Optional[torch.cuda.Event],
    ) -> Dict[int, SequenceOutputs]:
        hidden_states = self.model(
            input_ids, positions, kv_caches, input_metadata, cache_events)
        next_tokens = self.sampler(
            self.lm_head.weight, hidden_states, input_metadata)
        return next_tokens
    
    _column_parallel_weights = ["embed_tokens.weight", "lm_head.weight",
                                "qkv_proj.weight", "gate_proj.weight",
                                "up_proj.weight"]
    _row_parallel_weights = ["o_proj.weight", "down_proj.weight"]

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        use_np_cache: bool = False
    ) -> None:
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, use_np_cache):
            if "rotary_emb.inf_freq" in name:
                continue
            is_attn_weight = False
            for stride_id, att_weight_name in enumerate(["q_proj", "k_proj", "v_proj"]):
                if att_weight_name not in name:
                    continue
                param = state_dict[name.replace(att_weight_name, "qkv_proj")]
                shard_size = param.shape[0] // 3
                loaded_weight = loaded_weight[
                    shard_size * tensor_model_parallel_rank:
                    shard_size * (tensor_model_parallel_rank + 1)]
                param_slice = param.data[shard_size * stride_id:
                                         shard_size * (stride_id + 1)]
                assert param_slice.shape == loaded_weight.shape
                param_slice.copy_(loaded_weight)
                is_attn_weight = True
                break
            if is_attn_weight:
                continue
            
            is_gate_up_weight = False
            for stride_id, weight_name in enumerate(["gate_proj", "up_proj"]):
                if weight_name not in name:
                    continue
                param = state_dict[name.replace(weight_name, "gate_up_proj")]
                shard_size = param.shape[0] // 2
                loaded_weight = loaded_weight[
                    shard_size * tensor_model_parallel_rank:
                    shard_size * (tensor_model_parallel_rank + 1)]
                param_slice = param.data[shard_size * stride_id:
                                         shard_size * (stride_id + 1)]
                assert param_slice.shape == loaded_weight.shape
                param_slice.copy_(loaded_weight)
                is_gate_up_weight = True
                break
            if is_gate_up_weight:
                continue

            param = state_dict[name]
            load_tensor_parallel_weights(
                param=param,
                loaded_weight=loaded_weight,
                param_name=name,
                column_parallel_weight_name=self._column_parallel_weights,
                row_parallel_weight_name=self._row_parallel_weights,
                tensor_model_parallel_rank=tensor_model_parallel_rank
            )