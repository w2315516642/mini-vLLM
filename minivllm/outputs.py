from enum import Enum
from typing import Optional, List, Dict

from minivllm.sequence import SequenceGroup, SequenceStatus


class RequestOutputKind(str, Enum):
    """Whether a consumer receives the whole result or only its new suffix."""

    CUMULATIVE = "cumulative"
    DELTA = "delta"
    FINAL_ONLY = "final_only"


# 单条output
class CompletionOutput:
    """The output data of one completion output of a request.

    Args:
        index: The index of the output in the request.
        text: The generated output text.
        token_ids: The token IDs of the generated output text.
        cumulative_logprob: The cumulative log probability of the generated
            output text.
        logprobs: The log probabilities of the top probability words at each
            position if the logprobs are requested.
        finish_reason: The reason why the sequence is finished.
    """

    def __init__(
        self,
        index: int,
        text: str,
        token_ids: List[int],
        cumulative_logprob: float,
        logprobs: Optional[List[Dict[int, float]]],
        finish_reason: Optional[str] = None,
    ) -> None:
        self.index = index
        self.text = text
        self.token_ids = token_ids
        self.cumulative_logprob = cumulative_logprob
        self.logprobs = logprobs
        self.finish_reason = finish_reason

    def finished(self) -> bool:
        return self.finish_reason is not None
    
    def __repr__(self) -> str:
        return (f"CompletionOutput(index={self.index}, "
                f"text={self.text!r}, "
                f"token_ids={self.token_ids}, "
                f"cumulative_logprob={self.cumulative_logprob}, "
                f"logprobs={self.logprobs}, "
                f"finish_reason={self.finish_reason})")


class RequestOutput:
    """The output data of a request to the LLM.

    Args:
        request_id: The unique ID of the request.
        prompt: The prompt string of the request.
        prompt_token_ids: The token IDs of the prompt.
        outputs: The output sequences of the request.
        finished: Whether the entire request (not just this update) is finished.
    """

    def __init__(
        self,
        request_id: str,
        prompt: str,
        prompt_token_ids: List[int],
        outputs: List[CompletionOutput],
        finished: Optional[bool] = None,
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.outputs = outputs
        # A delta can contain only one finished completion of an active group.
        self.finished = (
            all(output.finished() for output in outputs)
            if finished is None else finished
        )

    @classmethod
    def from_seq_group(cls, seq_group: SequenceGroup) -> "RequestOutput":
        # Get the top-n sequence.
        n = seq_group.sampling_params.n
        seqs = seq_group.get_seqs()
        assert n <= len(seqs)
        sorted_seqs = sorted(
            seqs, key=lambda seq: seq.get_cumulative_logprob(), reverse=True)
        top_n_seqs = sorted_seqs[:n]

        # Create the outputs.
        outputs: List[CompletionOutput] = []
        for seq in top_n_seqs:
            # The sequence keeps sampled logprobs even when not requested.
            logprobs = None
            if seq_group.sampling_params.logprobs is not None:
                logprobs = [dict(item) for item in seq.output_logprobs]
            finished_reason = SequenceStatus.get_finished_reason(seq.status)
            output = CompletionOutput(
                seqs.index(seq),
                seq.output_text,
                list(seq.get_output_token_ids()),
                seq.get_cumulative_logprob(),
                logprobs,
                finished_reason
            )
            outputs.append(output)
        
        prompt = top_n_seqs[0].prompt
        # These snapshots may outlive the next engine step in a stream.
        prompt_token_ids = list(top_n_seqs[0].data.prompt_token_ids)
        return cls(
            seq_group.request_id, prompt, prompt_token_ids, outputs,
            finished=seq_group.is_finished(),
        )

    def __repr__(self) -> str:
        return (f"RequestOutput(request_id={self.request_id}, "
                f"prompt={self.prompt!r}, "
                f"prompt_token_ids={self.prompt_token_ids}, "
                f"outputs={self.outputs})")

    def is_finished(self) -> bool:
        return self.finished
