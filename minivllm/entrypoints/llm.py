from contextlib import closing
from typing import Any, Iterator, List, Mapping, Optional, Sequence, Union

from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from minivllm.engine.arg_utils import EngineArgs
from minivllm.engine.llm_engine import LLMEngine
from minivllm.engine.output_processor import OutputProcessor
from minivllm.multimodal import MultiModalInputs
from minivllm.outputs import RequestOutput, RequestOutputKind
from minivllm.sampling_params import SamplingParams
from minivllm.utils import Counter


class LLM:
    """An LLM for generating texts from given prompts and sampling parameters.

    This class includes a tokenizer, a language model (possibly distributed
    across multiple GPUs), and GPU memory space allocated for intermediate
    states (aka KV cache). Given a batch of prompts and sampling parameters,
    this class generates texts from the model, using an intelligent batching
    mechanism and efficient memory management.

    NOTE: This is a synchronous inference interface. A single caller owns the
    engine loop, including while a streaming generator is suspended.
    NOTE: For the comprehensive list of arguments, see `EngineArgs`.

    Args:
        model: The name or path of a HuggingFace Transformers model.
        tensor_parallel_size: The number of GPUs to use for distributed
            execution with tensor parallelism.
        dtype: The data type for the model weights and activations. Currently,
            we support `float32`, `float16`, and `bfloat16`. If `auto`, we use
            the `torch_dtype` attribute specified in the model config file.
            However, if the `torch_dtype` in the config is `float32`, we will
            use `float16` instead.
        seed: The seed to initialize the random number generator for sampling.
    """

    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        seed: int = 0,
        **kwargs,
    ) -> None:
        if "disable_log_stats" not in kwargs:
            kwargs['disable_log_stats'] = True
        engine_args = EngineArgs(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            seed=seed,
            **kwargs,
        )
        self.llm_engine = LLMEngine.from_engine_args(engine_args)
        self.request_conuter = Counter()
        self._processor = None
        self._generation_active = False

    def get_tokenizer(
        self,
    ) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        return self.llm_engine.tokenizer

    def get_processor(self):
        """Lazily load the model processor used by image/video requests."""
        if not self.llm_engine.model_config.is_multimodal:
            raise ValueError("The selected model does not support vision inputs")
        if self._processor is None:
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(
                self.llm_engine.model_config.model,
                cache_dir=self.llm_engine.model_config.download_dir,
            )
        return self._processor
    
    def generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        sampling_params: Optional[SamplingParams] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        use_tqdm: bool = True,
        multi_modal_inputs: Optional[
            List[Union[MultiModalInputs, Mapping[str, Any]]]
        ] = None,
    ) -> List[RequestOutput]:
        """Generates the completions for the input prompts.

        NOTE: This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: A list of prompts to generate completions for.
            sampling_params: The sampling parameters for text generation. If
                None, we use the default sampling parameters.
            prompt_token_ids: A list of token IDs for the prompts. If None, we
                use the tokenizer to convert the prompts to token IDs.
            use_tqdm: Whether to use tqdm to display the progress bar.

        Returns:
            A list of `RequestOutput` objects containing the generated
            completions in the same order as the input prompts.
        """
        outputs = list(self.generate_stream(
            prompts, sampling_params, prompt_token_ids, use_tqdm,
            multi_modal_inputs, output_kind=RequestOutputKind.FINAL_ONLY,
        ))
        return sorted(outputs, key=lambda output: int(output.request_id))

    def generate_stream(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        sampling_params: Optional[SamplingParams] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        use_tqdm: bool = False,
        multi_modal_inputs: Optional[
            List[Union[MultiModalInputs, Mapping[str, Any]]]
        ] = None,
        *,
        output_kind: RequestOutputKind = RequestOutputKind.DELTA,
    ) -> Iterator[RequestOutput]:
        """Yield updates tagged by request_id and completion index.

        DELTA contains only new text/tokens; a verified speculative block may
        produce multiple tokens at once. Close the generator (for example with
        contextlib.closing) when stopping early to release outstanding requests.
        """
        if sampling_params is None:
            sampling_params = SamplingParams()
        processor = OutputProcessor(sampling_params, output_kind)
        with closing(self._generate(
            prompts, sampling_params, prompt_token_ids, use_tqdm,
            multi_modal_inputs,
        )) as stream:
            for output in stream:
                update = processor.process(output)
                if update is not None:
                    yield update

    def _generate(
        self, prompts, sampling_params, prompt_token_ids, use_tqdm,
        multi_modal_inputs,
    ) -> Iterator[RequestOutput]:
        """Own submission, stepping and cleanup for one synchronous batch."""
        if self._generation_active:
            raise RuntimeError(
                "Finish or close the active generation before starting another"
            )
        if (
            prompts is None
            and prompt_token_ids is None
            and multi_modal_inputs is None
        ):
            raise ValueError(
                "Either prompts or prompt_token_ids must be provided."
            )
        if isinstance(prompts, str):
            prompts = [prompts]
        if prompts is not None and prompt_token_ids is not None:
            if len(prompts) != len(prompt_token_ids):
                raise ValueError(
                    "The length of prompts and prompt_token_ids must be the same."
                )
        if multi_modal_inputs is not None:
            expected = (
                len(prompts)
                if prompts is not None
                else len(prompt_token_ids)
                if prompt_token_ids is not None
                else len(multi_modal_inputs)
            )
            if len(multi_modal_inputs) != expected:
                raise ValueError(
                    "multi_modal_inputs must contain one item per prompt"
                )

        # Add requests to the engine.
        if prompts is not None:
            num_requests = len(prompts)
        elif prompt_token_ids is not None:
            num_requests = len(prompt_token_ids)
        else:
            num_requests = len(multi_modal_inputs)
        if num_requests == 0:
            return
        pending = set()
        self._generation_active = True
        try:
            for i in range(num_requests):
                prompt = prompts[i] if prompts is not None else None
                token_ids = (
                    prompt_token_ids[i] if prompt_token_ids is not None else None
                )
                request_inputs = None
                if multi_modal_inputs is not None:
                    request_inputs = multi_modal_inputs[i]
                    if isinstance(request_inputs, Mapping):
                        processed_ids, request_inputs = (
                            MultiModalInputs.from_processor_output(request_inputs)
                        )
                        if (
                            token_ids is not None
                            and tuple(token_ids) != processed_ids
                        ):
                            raise ValueError(
                                "prompt_token_ids do not match processor input_ids"
                            )
                        token_ids = list(processed_ids)
                pending.add(self._add_request(
                    prompt, sampling_params, token_ids, request_inputs,
                ))
            with closing(self._run_engine(use_tqdm)) as stream:
                for output in stream:
                    if output.is_finished():
                        pending.discard(output.request_id)
                    yield output
        finally:
            self._generation_active = False
            for request_id in pending:
                self.llm_engine.abort_request(request_id)

    def chat(
        self,
        messages: Union[
            Sequence[Mapping[str, Any]],
            Sequence[Sequence[Mapping[str, Any]]],
        ],
        sampling_params: Optional[SamplingParams] = None,
        enable_thinking: bool = True,
        use_tqdm: bool = True,
    ) -> List[RequestOutput]:
        """Run Qwen processor chat templates for image and video messages."""
        processed = self._prepare_chat(messages, enable_thinking)
        return self.generate(
            sampling_params=sampling_params,
            multi_modal_inputs=processed,
            use_tqdm=use_tqdm,
        )

    def chat_stream(
        self,
        messages: Union[
            Sequence[Mapping[str, Any]],
            Sequence[Sequence[Mapping[str, Any]]],
        ],
        sampling_params: Optional[SamplingParams] = None,
        enable_thinking: bool = True,
        *,
        output_kind: RequestOutputKind = RequestOutputKind.DELTA,
    ) -> Iterator[RequestOutput]:
        """Stream image/video chat using the same processor path as chat()."""
        return self.generate_stream(
            sampling_params=sampling_params,
            multi_modal_inputs=self._prepare_chat(messages, enable_thinking),
            output_kind=output_kind,
        )

    def _prepare_chat(self, messages, enable_thinking):
        conversations = list(messages)
        if not conversations:
            return []
        if isinstance(conversations[0], Mapping):
            conversations = [conversations]
        processor = self.get_processor()
        return [
            processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=enable_thinking,
            )
            for conversation in conversations
        ]

    def _add_request(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]],
        multi_modal_inputs: Optional[MultiModalInputs] = None,
    ) -> str:
        request_id = str(next(self.request_conuter))
        self.llm_engine.add_request(
            request_id,
            prompt,
            sampling_params,
            prompt_token_ids,
            multi_modal_inputs=multi_modal_inputs,
        )
        return request_id
        
    def _run_engine(self, use_tqdm: bool) -> Iterator[RequestOutput]:
        pbar = tqdm(
            total=self.llm_engine.get_num_unfinished_requests(),
            desc="Processed prompts", disable=not use_tqdm,
        )
        try:
            while self.llm_engine.has_unfinished_requests():
                for output in self.llm_engine.step():
                    if output.is_finished():
                        pbar.update(1)
                    yield output
        finally:
            pbar.close()
