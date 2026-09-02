"""Rank pairing rules for equal-TP prefill and decode replicas."""

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

from minivllm.distributed.kv_transfer.layout import CacheLayout


@dataclass(frozen=True)
class RankTransferPair:
    rank: int
    source: CacheLayout
    target: CacheLayout


@dataclass(frozen=True)
class PDTransferTopology:
    """One-to-one rank mapping used by the first PD implementation."""

    pairs: Tuple[RankTransferPair, ...]

    @classmethod
    def build(
        cls,
        source_layouts: Sequence[CacheLayout],
        target_layouts: Sequence[CacheLayout],
    ) -> "PDTransferTopology":
        if not source_layouts or not target_layouts:
            raise ValueError("P and D must each expose at least one rank")
        if len(source_layouts) != len(target_layouts):
            raise ValueError("P and D must use equal tensor parallel sizes")
        source_by_rank = cls._index_by_rank(source_layouts, "P")
        target_by_rank = cls._index_by_rank(target_layouts, "D")
        expected = set(range(len(source_layouts)))
        if set(source_by_rank) != expected or set(target_by_rank) != expected:
            raise ValueError("P and D ranks must be contiguous from zero")
        pairs = tuple(
            RankTransferPair(
                rank=rank,
                source=source_by_rank[rank],
                target=target_by_rank[rank],
            )
            for rank in sorted(expected)
        )
        for pair in pairs:
            if pair.source.block_size != pair.target.block_size:
                raise ValueError(
                    f"rank {pair.rank} uses different P/D block sizes"
                )
            if pair.source.region_keys() != pair.target.region_keys():
                raise ValueError(
                    f"rank {pair.rank} exposes different P/D cache regions"
                )
        return cls(pairs)

    @staticmethod
    def _index_by_rank(
        layouts: Sequence[CacheLayout],
        role: str,
    ) -> Dict[int, CacheLayout]:
        result = {}
        endpoint_ids = set()
        hostnames = set()
        for layout in layouts:
            endpoint = layout.endpoint
            if endpoint.rank in result:
                raise ValueError(f"{role} contains duplicate rank {endpoint.rank}")
            if endpoint.endpoint_id in endpoint_ids:
                raise ValueError(f"{role} contains duplicate endpoint IDs")
            if endpoint.hostname in hostnames:
                raise ValueError(f"{role} contains duplicate transfer addresses")
            result[endpoint.rank] = layout
            endpoint_ids.add(endpoint.endpoint_id)
            hostnames.add(endpoint.hostname)
        return result
