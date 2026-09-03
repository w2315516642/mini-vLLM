"""Convert cumulative engine snapshots into client-facing stream updates.

This layer owns output offsets, not decoding or scheduling. Both the local LLM
and the PD client use it, so moving a request from P to D does not reset offsets.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from minivllm.outputs import CompletionOutput, RequestOutput, RequestOutputKind
from minivllm.sampling_params import SamplingParams


@dataclass
class _OutputOffset:
    num_tokens: int = 0
    num_chars: int = 0
    finished: bool = False


class OutputProcessor:
    def __init__(
        self,
        sampling_params: SamplingParams,
        output_kind: RequestOutputKind = RequestOutputKind.DELTA,
    ) -> None:
        self.output_kind = RequestOutputKind(output_kind)
        if self.output_kind != RequestOutputKind.FINAL_ONLY and (
            sampling_params.best_of != sampling_params.n
            or sampling_params.use_beam_search
        ):
            raise ValueError(
                "Streaming requires best_of == n and use_beam_search=False; "
                "use FINAL_ONLY for candidates that can be replaced"
            )
        self._stop_buffer_length = max(
            (len(stop) - 1 for stop in sampling_params.stop), default=0
        )
        self._offsets: Dict[str, Dict[int, _OutputOffset]] = {}

    def process(self, output: RequestOutput) -> Optional[RequestOutput]:
        if self.output_kind == RequestOutputKind.FINAL_ONLY:
            return output if output.is_finished() else None

        offsets = self._offsets.setdefault(output.request_id, {})
        completions = []
        delta = self.output_kind == RequestOutputKind.DELTA
        for completion in output.outputs:
            offset = offsets.setdefault(completion.index, _OutputOffset())
            if delta and offset.finished:
                continue
            text = completion.text
            if not completion.finished():
                # A byte-fallback token may end in incomplete UTF-8. Also hold
                # back the suffix that could still become an excluded stop.
                text = text.rstrip("\ufffd")
                text = text[:max(0, len(text) - self._stop_buffer_length)]
            start_token = offset.num_tokens if delta else 0
            start_char = offset.num_chars if delta else 0
            token_ids = completion.token_ids[start_token:]
            new_text = text[start_char:]
            if (
                delta and not token_ids and not new_text
                and not completion.finished()
            ):
                continue
            logprobs = completion.logprobs
            completions.append(CompletionOutput(
                completion.index,
                new_text,
                token_ids,
                completion.cumulative_logprob,
                None if logprobs is None else [
                    dict(item) for item in logprobs[start_token:]
                ],
                completion.finish_reason,
            ))
            offset.num_tokens = len(completion.token_ids)
            offset.num_chars = len(text)
            offset.finished = completion.finished()

        if output.is_finished():
            del self._offsets[output.request_id]
        if not completions and not output.is_finished():
            return None
        return RequestOutput(
            output.request_id,
            output.prompt,
            list(output.prompt_token_ids),
            completions,
            finished=output.is_finished(),
        )
