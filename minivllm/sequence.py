from enum import Enum, auto
from typing import Optional, List, Dict

from minivllm.kv_cache.block import LogicalTokenBlock
from minivllm.sampling_params import SamplingParams

class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED_STOP = auto()
    FINISHED_LENGTH_CAPPED = auto()

    @staticmethod
    def is_finished(status: "SequenceStatus") -> bool:
        return status in [
            SequenceStatus.FINISHED_STOP,
            SequenceStatus.FINISHED_LENGTH_CAPPED
        ]


class Sequence:
    def __init__(
        self, 
        seq_id: int,
        prompt: str,
        prompt_token_ids: List[int],
        block_size: int,
    ) -> None:

        self.seq_id = seq_id
        self.block_size = block_size

        self.prompt = prompt

        self.prompt_token_ids = prompt_token_ids
        self.output_token_ids: List[int] = []
        self.cumulative_logprob = 0.0

        self.output_tokens: List[str] = []
        self.output_text = ""
        # 这个应该是给beam search这类或者算困惑度用的
        self.output_logprobs: List[Dict[int, float]] = []

        self.logical_token_blocks: List[LogicalTokenBlock] = []
        self._append_tokens_to_blocks(prompt_token_ids)
        self.status = SequenceStatus.WAITING

    def append_logical_block(self) -> None:
        block = LogicalTokenBlock(
            block_id=len(self.logical_token_blocks),
            block_size=self.block_size
        )
        self.logical_token_blocks.append(block)

    def _append_tokens_to_blocks(self, token_ids: List[int]) -> None:
        while token_ids:
            if not self.logical_token_blocks:
                self.append_logical_block()
            
            last_block = self.logical_token_blocks[-1]
            if last_block.is_full():
                self.append_logical_block()
                last_block = self.logical_token_blocks[-1]
            
            num_empty_slots = last_block.get_num_empty_slots()
            last_block.append_tokens(token_ids[:num_empty_slots])
            token_ids = token_ids[num_empty_slots:]

    def append_token_id(
        self, 
        token_id: int,
        logprobs: Dict[int, float],
    ) -> None:
        assert token_id in logprobs
        self._append_tokens_to_blocks([token_id])
        
        self.output_token_ids.append(token_id)
        self.cumulative_logprob += logprobs[token_id]
        self.output_logprobs.append(logprobs)
        
        self.num_tokens += 1
    
    def get_len(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    def get_output_len(self) -> int:
        return len(self.output_token_ids)

    def get_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids

    def get_output_token_ids(self) -> List[int]:
        return self.output_token_ids

    def get_last_token_id(self) -> List[int]:
        if not self.output_token_ids:
            return self.prompt_token_ids[-1]
        return self.output_token_ids[-1]
    
    def get_cumulative_logprob(self) -> float:
        return self.cumulative_logprob

    def is_finished(self):
        return SequenceStatus.is_finished(self.status)


class SequenceGroup:
    def __init__(
        self, 
        request_id: str,
        seqs: List[Sequence],
        sampling_params: SamplingParams
    ) -> None:
        self.request_id = request_id
        self.seqs = seqs
        self.sampling_params = sampling_params

    def get_seqs(
        self,
        status: Optional[SequenceStatus] = None,
    ) -> List[Sequence]:
        if status is None:
            return self.seqs
        else:
            return [seq for seq in self.seqs if seq.status == status]

    def num_seqs(
        self,
        status: Optional[SequenceStatus] = None,
    ) -> int:
        return len(self.get_seqs(status))

    def find(self, seq_id: int) -> Sequence:
        for seq in self.seqs:
            if seq.seq_id == seq_id:
                return seq
        raise ValueError(f"Sequence {seq_id} is not found.")

    def is_finished(self) -> bool:
        return all(seq.is_finished() for seq in self.seqs)