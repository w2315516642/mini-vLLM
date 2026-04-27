from copy import copy
from enum import Enum, auto
from typing import List
from itertools import count


class SequenceStatus(Enum):
    WAITING = auto()
    PREFILL = auto()
    DECODE = auto()
    FINISHED = auto()


class SequenceGroup:
    def __init__(self, request_id: int):
        self.sequences: List[Sequence] = []
        self.request_id = request_id

    def add_sequence(self, token_ids: List[int], attention_mask: List[int]):
        self.sequences.append(Sequence(token_ids, attention_mask))

    @property
    def is_finished(self):
        for sequence in self.sequences:
            if sequence.is_finished:
                return True
        return False

class Sequence:
    counter = count()

    def __init__(self, token_ids: List[int], attention_mask: List[int]) -> None:

        self.request_id = next(Sequence.counter)

        self.token_ids = copy(token_ids)
        self.attention_mask = attention_mask
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.status = SequenceStatus.PREFILL

        self.kv_cache = None

    def __len__(self):
        return self.num_tokens

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def is_prefill(self):
        return self.status == SequenceStatus.PREFILL

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def output_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def next_token(self):
        if self.is_prefill:
            ans = self.token_ids[:self.num_prompt_tokens]
        else:
            ans = [self.token_ids[-1]]
        return [ans]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.num_tokens += 1

    def decoding(self):
        self.status = SequenceStatus.DECODE
    
    def finished(self):
        self.status = SequenceStatus.FINISHED