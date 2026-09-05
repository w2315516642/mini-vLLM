from typing import Optional, Tuple

import torch
import torch.nn as nn
from xformers import ops as xops

from minivllm import attention_ops
from minivllm import cache_ops
from minivllm import pos_encoding_ops
from minivllm.model_executor.input_metadata import InputMetadata
from minivllm.profiling import nvtx_function

_SUPPORTED_HEAD_SIZES = {64, 80, 96, 128, 256}

class PagedAttention(nn.Module):
    """(by original author) GPT-style multi-head PagedAttention.

    This class takes flattened 1D query, key, and value tensors as input. The
    input 1D tensors can be split into four parts: fresh prompt tokens, cached
    prompt suffixes, generation tokens, and paddings.

    |<------------------------------------- num_valid_tokens ------------------------------------->|
    |<--------------- num_prompt_tokens -------------->|<------- generation tokens ------->|
    |<--fresh prompts-->|<--cached prompt suffixes-->|<--generation_0-->|...|<--generation_M-1-->|<--padding-->|

    The prompts might have different lengths, while the generation tokens always
    have length 1. The paddings are appended to make the input length a multiple
    of 8, which is desirable for Tensor Cores.

    The class does the following:
    1. Perform multi_query_kv_attention for fresh prompts. This operation does
        not read the KV cache.
    2. Wait for the cache operations (e.g., swap, copy) to finish. The cache
        operations are issued by the cache engine before executing the forward
        pass of the model, and they are executed asynchronously.
    3. Reshape and store the input key and value tensors in the KV cache.
    4. Perform varlen_query_cached_kv_attention for cached prompt suffixes.
        This operation reads their cached prefixes and the suffix K/V written
        in step 3.
    5. Perform single_query_cached_kv_attention for the generation tokens.
        This operation reads the previous key and value tensors from the KV
        cache.
    6. Output a flattened 1D tensor.
    """

    def __init__(
        self, 
        num_heads: int, 
        head_size: int, 
        scale: float,
        num_kv_heads: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = (
            num_heads if num_kv_heads is None else num_kv_heads
        )
        if self.num_heads <= 0 or self.num_kv_heads <= 0:
            raise ValueError("Query and KV head counts must be positive")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("Query heads must be grouped evenly by KV heads")
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.head_size = head_size
        self.scale = scale
        # Let xFormers select a compatible CUDA backend. The legacy CUTLASS
        # operator rejects GPUs newer than SM90, including SM120 (Blackwell).
        self.attn_op = None

        if self.head_size not in _SUPPORTED_HEAD_SIZES:
            raise ValueError(f"head size ({self.head_size}) is not supported. "
                             f"Supported head sizes: {_SUPPORTED_HEAD_SIZES}")

    def _reshape_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Restore explicit query and compact KV head dimensions."""
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)
        return query, key, value

    def _grouped_prefill_inputs(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build xFormers [B, T, KV groups, Q per group, D] views."""
        kv_groups = self.num_kv_heads
        q_per_group = self.num_heads // kv_groups

        query = query.view(-1, kv_groups, q_per_group, self.head_size).unsqueeze(0)
        key = key.unsqueeze(-2).expand(-1, -1, q_per_group, -1).unsqueeze(0)
        value = value.unsqueeze(-2).expand(-1, -1, q_per_group, -1).unsqueeze(0)
        return query, key, value

    def multi_query_kv_attention(
        self,
        output: torch.Tensor,       # [num_prompt_tokens, num_query_heads, head_size]
        query: torch.Tensor,        # [num_prompt_tokens, num_query_heads, head_size]
        key: torch.Tensor,          # [num_prompt_tokens, num_kv_heads, head_size]
        value: torch.Tensor,        # [num_prompt_tokens, num_kv_heads, head_size]
        attn_bias: xops.AttentionBias,
    ) -> torch.Tensor:
        # TODO(woosuk): The unsqueeze op may incur some CPU overhead. Optimize.
        if self.num_heads == self.num_kv_heads:
            xformers_query = query.unsqueeze(0)
            xformers_key = key.unsqueeze(0)
            xformers_value = value.unsqueeze(0)
        else:
            xformers_query, xformers_key, xformers_value = (
                self._grouped_prefill_inputs(query, key, value)
            )
        out = xops.memory_efficient_attention_forward(
            xformers_query,
            xformers_key,
            xformers_value,
            attn_bias=attn_bias,
            p=0.0,
            scale=self.scale,
            op=self.attn_op
        )
        output.copy_(out.reshape_as(output))
        return output

    def single_query_cached_kv_attention(
        self,
        output: torch.Tensor,       # [num_generation_tokens, num_query_heads, head_size]
        query: torch.Tensor,        # [num_generation_tokens, num_query_heads, head_size]
        key_cache: torch.Tensor,    # [num_blocks, num_kv_heads, head_size/x, block_size, x]
        value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size, block_size]
        input_metadata: InputMetadata
    ) -> None:
        block_size = value_cache.shape[3]
        attention_ops.single_query_cached_kv_attention(
            output,
            query,
            key_cache,
            value_cache,
            self.scale,
            input_metadata.block_tables,
            input_metadata.context_lens,
            block_size,
            input_metadata.max_context_len
        )

    def varlen_query_cached_kv_attention(
        self,
        output: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> None:
        """Attend prompt suffix queries to a cached paged KV prefix."""
        assert input_metadata.cached_prompt_cu_seqlens is not None
        assert input_metadata.cached_prompt_block_tables is not None
        assert input_metadata.cached_prompt_context_lens is not None
        block_size = value_cache.shape[3]
        attention_ops.varlen_query_cached_kv_attention(
            output,
            query,
            key_cache,
            value_cache,
            input_metadata.cached_prompt_cu_seqlens,
            max(input_metadata.cached_prompt_query_lens),
            self.scale,
            input_metadata.cached_prompt_block_tables,
            input_metadata.cached_prompt_context_lens,
            block_size,
            input_metadata.max_cached_prompt_context_len,
            True,
        )

    @nvtx_function("full_attention")
    def forward(
        self,
        query: torch.Tensor,  # [num_tokens, num_query_heads * head_size]
        key: torch.Tensor,    # [num_tokens, num_kv_heads * head_size]
        value: torch.Tensor,  # [num_tokens, num_kv_heads * head_size]
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        query, key, value = self._reshape_qkv(query, key, value)

        output = torch.empty_like(query)

        # Prompts without a cache hit keep the original xFormers fast path.
        num_fresh_prompt_tokens = input_metadata.num_fresh_prompt_tokens
        if num_fresh_prompt_tokens > 0:
            assert input_metadata.attn_bias is not None
            self.multi_query_kv_attention(
                output[:num_fresh_prompt_tokens],
                query[:num_fresh_prompt_tokens],
                key[:num_fresh_prompt_tokens],
                value[:num_fresh_prompt_tokens],
                input_metadata.attn_bias
            )
        
        # Wait until the cache op is done.
        if cache_event is not None:
            cache_event.wait()

        # Reshape the keys and values and store them in the cache.
        # When key_cache and value_cache are not provided, the new key
        # and value vectors will not be cached.
        num_valid_tokens = input_metadata.num_valid_tokens
        if (num_valid_tokens > 0 and key_cache is not None
            and value_cache is not None):
            # The stride is 3 because the key and value are sliced from qkv.
            # NOTE: 这里把对应层的kv cache整个（指针）传进去了
            # 然后根据slot_mapping把算好的key和value通过这个函数放置到kv cache的对应位置
            cache_ops.reshape_and_cache(
                key[:num_valid_tokens],
                value[:num_valid_tokens],
                key_cache,
                value_cache,
                input_metadata.slot_mapping
            )

        # Cached prompt suffixes need both the reused prefix and the K/V just
        # written for the current suffix. The CUDA kernel applies causality per
        # query token while reading the paged block table.
        num_cached_prompt_tokens = input_metadata.num_cached_prompt_tokens
        if num_cached_prompt_tokens > 0:
            assert key_cache is not None and value_cache is not None
            cached_start = num_fresh_prompt_tokens
            cached_end = cached_start + num_cached_prompt_tokens
            self.varlen_query_cached_kv_attention(
                output[cached_start:cached_end],
                query[cached_start:cached_end],
                key_cache,
                value_cache,
                input_metadata,
            )
        
        if input_metadata.num_generation_tokens > 0:
            assert key_cache is not None and value_cache is not None, (
                "key_cache and value_cache must be provided when "
                "generating tokens."
            )
            self.single_query_cached_kv_attention(
                output[input_metadata.num_prompt_tokens:num_valid_tokens],
                query[input_metadata.num_prompt_tokens:num_valid_tokens],
                key_cache,
                value_cache,
                input_metadata
            )

        # Reshape the output tensor.
        # NOTE(woosuk): The output tensor may include paddings.
        return output.view(-1, self.num_heads * self.head_size)


class PagedAttentionWithRoPE(PagedAttention):
    """PagedAttention with GPT-NeoX style rotary embedding."""

    def __init__(
        self,
        num_heads: int, 
        head_size: int, 
        scale: float,
        rotary_dim: int,
        max_position: int = 8192,       # max model len
        base: float = 10000,
        num_kv_heads: Optional[int] = None,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads=num_kv_heads,
        )

        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2) / rotary_dim))
        t = torch.arange(max_position).float()
        freqs = torch.einsum('i,j -> ij', t, inv_freq.float())
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)

        torch_dtype = torch.get_default_dtype()
        cache = cache.to(torch_dtype)
        self.register_buffer("cos_sin_cache", cache, persistent=False)
    
    def forward(
        self,
        positions: torch.Tensor,                # [num_tokens]
        query: torch.Tensor,  # [num_tokens, num_query_heads * head_size]
        key: torch.Tensor,    # [num_tokens, num_kv_heads * head_size]
        value: torch.Tensor,  # [num_tokens, num_kv_heads * head_size]
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        # Apply rotary embedding to the query and key before passing them
        # to the attention op.
        pos_encoding_ops.rotary_embedding_neox(
            positions,
            query,
            key,
            self.head_size,
            self.cos_sin_cache,
        )
        return super().forward(
            query,
            key,
            value,
            key_cache,
            value_cache,
            input_metadata,
            cache_event,
        )
