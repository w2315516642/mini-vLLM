"""Packed metadata used by DSpark's variable-length draft blocks."""

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class DraftAttentionMetadata:
    """Describe block-parallel queries sharing the target block table."""

    query_lens: Sequence[int]
    cu_seqlens_q: torch.Tensor
    context_lens: torch.Tensor
    block_tables: torch.Tensor
    slot_mapping: torch.Tensor

    def __post_init__(self) -> None:
        query_lens = tuple(int(length) for length in self.query_lens)
        object.__setattr__(self, "query_lens", query_lens)
        if not query_lens or any(length <= 0 for length in query_lens):
            raise ValueError("Draft query lengths must be positive")
        if self.cu_seqlens_q.shape != (len(query_lens) + 1,):
            raise ValueError("cu_seqlens_q must contain one boundary per request")
        if self.context_lens.shape != (len(query_lens),):
            raise ValueError("context_lens must contain one value per request")
        if self.block_tables.ndim != 2 or self.block_tables.shape[0] != len(query_lens):
            raise ValueError("block_tables must have one row per request")
        if self.slot_mapping.shape != (sum(query_lens),):
            raise ValueError("slot_mapping must contain one slot per draft query")

    @property
    def num_tokens(self) -> int:
        return sum(self.query_lens)

    @property
    def max_query_len(self) -> int:
        return max(self.query_lens)

    @property
    def max_context_len(self) -> int:
        return int(self.context_lens.max().item())
