"""Run after stage 8 TODOs pass; requires an existing local floating checkpoint."""
import argparse
from pathlib import Path

import torch

from minivllm.entrypoints.llm import LLM
from minivllm.sampling_params import SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()
    if not Path(args.model).is_dir():
        parser.error("--model must be an existing local checkpoint directory")
    try:
        llm = LLM(
            model=args.model, tensor_parallel_size=1, dtype="bfloat16",
            max_num_seqs=1, max_num_batched_tokens=256,
            gpu_memory_utilization=0.75, swap_space=0,
            enable_prefix_caching=False,
        )
        tokenizer = llm.get_tokenizer()
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}], tokenize=True,
            add_generation_prompt=True,
        )
        if len(prompt_ids) > 256:
            parser.error("The formatted prompt exceeds the 256-token smoke budget")
        params = SamplingParams(temperature=0, max_tokens=args.max_tokens)
        previous = None
        for iteration in range(2):
            outputs = llm.generate(prompt_token_ids=[prompt_ids],
                                   sampling_params=params, use_tqdm=False)
            if len(outputs) != 1 or not outputs[0].outputs:
                raise AssertionError("Generation returned no completion")
            completion = outputs[0].outputs[0]
            tokens = list(completion.token_ids)
            if not tokens:
                raise AssertionError("Generation returned no tokens")
            if previous is not None and previous != tokens:
                raise AssertionError("Repeated greedy generation changed after state reuse")
            previous = tokens
            for worker in llm.llm_engine.workers:
                if worker.hybrid_cache.num_active_slots:
                    raise AssertionError("Completed request retained a hybrid state slot")
            print(f"Run {iteration + 1}: {tokens}")
            print(tokenizer.decode(tokens, skip_special_tokens=True))
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
