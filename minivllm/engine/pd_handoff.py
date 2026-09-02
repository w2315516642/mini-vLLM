"""Control-plane payload used to rebuild a request on the decode engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from minivllm.multimodal import MultiModalInputs, PositionIds
from minivllm.sampling_params import SamplingParams
from minivllm.sequence import Sequence, SequenceGroup, SequenceStatus


_SAMPLING_FIELDS = (
    "n",
    "best_of",
    "presence_penalty",
    "frequency_penalty",
    "temperature",
    "top_p",
    "top_k",
    "use_beam_search",
    "stop",
    "ignore_eos",
    "max_tokens",
    "logprobs",
)


def sampling_params_to_dict(params: SamplingParams) -> Dict[str, Any]:
    return {
        field_name: (
            list(getattr(params, field_name))
            if field_name == "stop"
            else getattr(params, field_name)
        )
        for field_name in _SAMPLING_FIELDS
    }


def sampling_params_from_dict(value: Mapping[str, Any]) -> SamplingParams:
    missing = [field_name for field_name in _SAMPLING_FIELDS if field_name not in value]
    if missing:
        raise ValueError(f"sampling parameters are missing fields: {missing}")
    return SamplingParams(**{name: value[name] for name in _SAMPLING_FIELDS})


@dataclass(frozen=True)
class MultiModalPositionHandoff:
    """Small multimodal state needed after P has encoded all visual inputs."""

    token_type_ids: Tuple[int, ...]
    position_ids: PositionIds
    rope_delta: int

    @classmethod
    def from_inputs(
        cls, value: Optional[MultiModalInputs]
    ) -> Optional["MultiModalPositionHandoff"]:
        if value is None:
            return None
        if value.position_ids is None:
            raise ValueError("multimodal positions must be prepared before handoff")
        return cls(value.token_type_ids, value.position_ids, value.rope_delta)

    def to_inputs(self) -> MultiModalInputs:
        return MultiModalInputs(
            token_type_ids=self.token_type_ids,
            position_ids=self.position_ids,
            rope_delta=self.rope_delta,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_type_ids": list(self.token_type_ids),
            "position_ids": [list(row) for row in self.position_ids],
            "rope_delta": self.rope_delta,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "MultiModalPositionHandoff":
        rows = tuple(tuple(int(item) for item in row) for row in value["position_ids"])
        if len(rows) != 3:
            raise ValueError("multimodal position_ids must contain three rows")
        return cls(
            token_type_ids=tuple(int(item) for item in value["token_type_ids"]),
            position_ids=rows,  # type: ignore[arg-type]
            rope_delta=int(value["rope_delta"]),
        )


@dataclass(frozen=True)
class SequenceHandoff:
    """Model-independent sequence progress at the P/D boundary.

    A hybrid source slot contains state *after* ``num_computed_tokens``. P has
    already sampled ``output_token_ids[0]``, so D continues by consuming that
    token. This direct handoff does not replay the final prompt token.
    """

    seq_id: int
    prompt: Optional[str]
    prompt_token_ids: Tuple[int, ...]
    output_token_ids: Tuple[int, ...]
    output_logprobs: Tuple[Mapping[int, float], ...]
    num_computed_tokens: int
    source_block_ids: Tuple[int, ...]
    source_state_slot: Optional[int] = None
    speculative_token_id: Optional[int] = None

    def __post_init__(self) -> None:
        if self.seq_id < 0:
            raise ValueError("seq_id must be non-negative")
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        if len(self.output_token_ids) != len(self.output_logprobs):
            raise ValueError("each output token requires one logprob mapping")
        if self.num_computed_tokens <= 0:
            raise ValueError("num_computed_tokens must be positive")
        if self.num_computed_tokens > len(self.prompt_token_ids):
            raise ValueError("PD handoff cannot include computed decode tokens")
        if not self.source_block_ids:
            raise ValueError("source_block_ids must not be empty")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq_id": self.seq_id,
            "prompt": self.prompt,
            "prompt_token_ids": list(self.prompt_token_ids),
            "output_token_ids": list(self.output_token_ids),
            "output_logprobs": [
                {str(token): value for token, value in logprobs.items()}
                for logprobs in self.output_logprobs
            ],
            "num_computed_tokens": self.num_computed_tokens,
            "source_block_ids": list(self.source_block_ids),
            "source_state_slot": self.source_state_slot,
            "speculative_token_id": self.speculative_token_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SequenceHandoff":
        return cls(
            seq_id=int(value["seq_id"]),
            prompt=value.get("prompt"),
            prompt_token_ids=tuple(int(item) for item in value["prompt_token_ids"]),
            output_token_ids=tuple(int(item) for item in value["output_token_ids"]),
            output_logprobs=tuple(
                {int(token): float(score) for token, score in item.items()}
                for item in value["output_logprobs"]
            ),
            num_computed_tokens=int(value["num_computed_tokens"]),
            source_block_ids=tuple(int(item) for item in value["source_block_ids"]),
            source_state_slot=(
                None
                if value.get("source_state_slot") is None
                else int(value["source_state_slot"])
            ),
            speculative_token_id=(
                None
                if value.get("speculative_token_id") is None
                else int(value["speculative_token_id"])
            ),
        )


@dataclass(frozen=True)
class RequestHandoff:
    """Serializable request state sealed by P and consumed once by D."""

    request_id: str
    sequences: Tuple[SequenceHandoff, ...]
    sampling_params: Mapping[str, Any]
    arrival_time: float
    multimodal: Optional[MultiModalPositionHandoff] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id must not be empty")
        if not self.sequences:
            raise ValueError("a request handoff requires at least one sequence")
        seq_ids = [sequence.seq_id for sequence in self.sequences]
        if len(set(seq_ids)) != len(seq_ids):
            raise ValueError("handoff sequence IDs must be unique")

    @classmethod
    def from_sequence_group(
        cls,
        seq_group: SequenceGroup,
        block_tables: Mapping[int, Tuple[int, ...]],
        state_slots: Optional[Mapping[int, int]] = None,
    ) -> "RequestHandoff":
        params = seq_group.sampling_params
        if params.n != 1 or params.best_of != 1 or params.use_beam_search:
            raise ValueError("PD handoff currently supports one non-beam sequence")
        sequences = []
        for sequence in seq_group.seqs:
            if sequence.num_computed_tokens != sequence.get_prompt_len():
                raise ValueError(
                    "a request can be sealed only after its entire prompt "
                    "has been computed"
                )
            if not sequence.get_output_token_ids():
                raise ValueError("prefill must sample the first output token")
            try:
                source_blocks = block_tables[sequence.seq_id]
            except KeyError as exc:
                raise ValueError("source block table is missing a sequence") from exc
            sequences.append(
                SequenceHandoff(
                    seq_id=sequence.seq_id,
                    prompt=sequence.prompt,
                    prompt_token_ids=tuple(sequence.data.prompt_token_ids),
                    output_token_ids=tuple(sequence.data.output_token_ids),
                    output_logprobs=tuple(sequence.output_logprobs),
                    num_computed_tokens=sequence.num_computed_tokens,
                    source_block_ids=tuple(source_blocks),
                    source_state_slot=(state_slots or {}).get(sequence.seq_id),
                    speculative_token_id=sequence.speculative_token_id,
                )
            )
        return cls(
            request_id=seq_group.request_id,
            sequences=tuple(sequences),
            sampling_params=sampling_params_to_dict(params),
            arrival_time=seq_group.arrival_time,
            multimodal=MultiModalPositionHandoff.from_inputs(
                seq_group.multi_modal_inputs
            ),
        )

    def rebuild_sequence_group(self, block_size: int) -> SequenceGroup:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        sequences = []
        for snapshot in self.sequences:
            sequence = Sequence(
                seq_id=snapshot.seq_id,
                prompt=snapshot.prompt,
                prompt_token_ids=list(snapshot.prompt_token_ids),
                block_size=block_size,
            )
            for token_id, logprobs in zip(
                snapshot.output_token_ids, snapshot.output_logprobs
            ):
                sequence.append_token_id(token_id, dict(logprobs))
            sequence.num_computed_tokens = snapshot.num_computed_tokens
            sequence.speculative_token_id = snapshot.speculative_token_id
            sequence.status = SequenceStatus.WAITING
            sequences.append(sequence)
        return SequenceGroup(
            request_id=self.request_id,
            seqs=sequences,
            sampling_params=sampling_params_from_dict(self.sampling_params),
            arrival_time=self.arrival_time,
            multi_modal_inputs=(
                None if self.multimodal is None else self.multimodal.to_inputs()
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "sequences": [sequence.to_dict() for sequence in self.sequences],
            "sampling_params": dict(self.sampling_params),
            "arrival_time": self.arrival_time,
            "multimodal": (
                None if self.multimodal is None else self.multimodal.to_dict()
            ),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RequestHandoff":
        multimodal = value.get("multimodal")
        return cls(
            request_id=str(value["request_id"]),
            sequences=tuple(
                SequenceHandoff.from_dict(item) for item in value["sequences"]
            ),
            sampling_params=dict(value["sampling_params"]),
            arrival_time=float(value["arrival_time"]),
            multimodal=(
                None
                if multimodal is None
                else MultiModalPositionHandoff.from_dict(multimodal)
            ),
            metadata=dict(value.get("metadata", {})),
        )
