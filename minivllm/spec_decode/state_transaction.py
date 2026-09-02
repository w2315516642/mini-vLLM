"""Plan the deterministic replay used to commit hybrid-model state."""

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SpeculativeReplayItem:
    seq_id: int
    sequence_data: object
    token_ids: Tuple[int, ...]
    positions: Tuple[int, ...]
    block_table: Tuple[int, ...]
    multimodal_inputs: Optional[object]

    @property
    def context_len(self) -> int:
        return self.positions[-1] + 1


def build_speculative_replay_plan(
    metadata_list: Iterable[object],
    committed_tokens: Mapping[int, int],
) -> Tuple[SpeculativeReplayItem, ...]:
    """Select only target inputs whose state survived block verification."""
    plan = []
    for metadata in metadata_list:
        if not metadata.is_speculative:
            continue
        seq_id = next(iter(metadata.seq_data))
        if seq_id not in committed_tokens:
            continue
        start = metadata.num_computed_tokens[seq_id]
        drafts: Sequence[int] = metadata.speculative_token_blocks[seq_id]
        all_inputs = [metadata.seq_data[seq_id].get_token_ids()[start], *drafts]
        query_len = int(committed_tokens[seq_id])
        if not 1 <= query_len <= len(all_inputs):
            raise ValueError("Invalid committed speculative prefix length")
        plan.append(SpeculativeReplayItem(
            seq_id=seq_id,
            sequence_data=metadata.seq_data[seq_id],
            token_ids=tuple(int(token) for token in all_inputs[:query_len]),
            positions=tuple(range(start, start + query_len)),
            block_table=tuple(metadata.block_tables[seq_id]),
            multimodal_inputs=metadata.multi_modal_inputs,
        ))
    if len(plan) != len(committed_tokens):
        planned_ids = {item.seq_id for item in plan}
        missing = sorted(set(committed_tokens) - planned_ids)
        raise ValueError(f"Missing speculative replay metadata for {missing}")
    return tuple(plan)


__all__ = ["SpeculativeReplayItem", "build_speculative_replay_plan"]
