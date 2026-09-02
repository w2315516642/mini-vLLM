"""Paged block attention used by the DSpark draft model."""

from typing import Tuple

import torch
import torch.nn as nn

from minivllm import attention_ops, cache_ops
from minivllm.spec_decode.draft_metadata import DraftAttentionMetadata


class DraftPagedAttention(nn.Module):
    """Write current draft K/V and attend to context plus the full block."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float,
    ) -> None:
        super().__init__()
        if num_heads <= 0 or num_kv_heads <= 0 or num_heads % num_kv_heads:
            raise ValueError("Invalid DSpark GQA head layout")
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.scale = float(scale)

    def write_context(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        cache: Tuple[torch.Tensor, torch.Tensor],
        slot_mapping: torch.Tensor,
    ) -> None:
        key_cache, value_cache = cache
        cache_ops.reshape_and_cache(
            key.view(-1, self.num_kv_heads, self.head_dim),
            value.view(-1, self.num_kv_heads, self.head_dim),
            key_cache,
            value_cache,
            slot_mapping,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cache: Tuple[torch.Tensor, torch.Tensor],
        metadata: DraftAttentionMetadata,
    ) -> torch.Tensor:
        if query.shape != (metadata.num_tokens, self.num_heads, self.head_dim):
            raise ValueError("Packed draft query shape does not match metadata")
        if key.shape != value.shape or key.shape != (
            metadata.num_tokens,
            self.num_kv_heads,
            self.head_dim,
        ):
            raise ValueError("Packed draft K/V shape does not match metadata")
        self.write_context(key, value, cache, metadata.slot_mapping)
        output = torch.empty_like(query)
        key_cache, value_cache = cache
        attention_ops.varlen_query_cached_kv_attention(
            output,
            query,
            key_cache,
            value_cache,
            metadata.cu_seqlens_q,
            metadata.max_query_len,
            self.scale,
            metadata.block_tables,
            metadata.context_lens,
            value_cache.shape[-1],
            metadata.max_context_len,
            False,
        )
        return output
