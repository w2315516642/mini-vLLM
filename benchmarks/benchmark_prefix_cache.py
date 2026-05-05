import argparse
import time
import json
from typing import List, Dict
from pathlib import Path

import numpy as np
import torch    
from tqdm import tqdm

from minivllm import LLM, SamplingParams


def load_dataset(file_path: str, num_data: int = 10) -> List[List[int]]:
    prompt_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        cnt = 0
        for line in f:
            item = json.loads(line)
            prompt_list.append(item['prompts'])
            cnt += 1
            if cnt >= num_data:
                break
    return prompt_list


def main(args: argparse.Namespace):
    print(args)

    # Process all the requests in a single batch if possible.
    # NOTE(woosuk): If the request cannot be processed in a single batch,
    # the engine will automatically process the request in multiple batches.
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=args.batch_size * args.input_len,
    )

    sampling_params = SamplingParams(
        n=args.n,
        temperature=0.0 if args.use_beam_search else 1.0,
        top_p=1.0,
        use_beam_search=args.use_beam_search,
        ignore_eos=True,
        max_tokens=args.output_len
    )
    print(sampling_params)
    dummy_prompt_token_ids = [[0] * args.input_len] * args.batch_size

    def run_to_completion(token_ids: List[List[int]], profile: bool = False):
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        start_time = time.time()
        llm.generate(
            prompt_token_ids=token_ids,
            sampling_params=sampling_params,
            use_tqdm=False
        )
        end_time = time.time()
        latency = end_time - start_time
        if profile:
            torch.cuda.cudart().cudaProfilerStop()
        return latency

    print("Warming up...")
    run_to_completion(dummy_prompt_token_ids, profile=False)

    print("Loading dataset...")
    DATASETS_DIR = Path(__file__).parent / "datasets"
    data_path = DATASETS_DIR / "share_gpt_prompt.jsonl"
    prompt_lists = load_dataset(data_path, num_data=args.batch_size)

    max_len = len(max(prompt_lists, key=len))
    
    # Benchmark.
    latencies: List[float] = []
    encode_fn = llm.llm_engine.tokenizer.encode

    for i in tqdm(range(max_len), desc="Testing prompt"):
        prompts = [prompt[i] for prompt in prompt_lists if i < len(prompt)]
        # prompts = [prompt_lists[1][i]] * args.batch_size if i < len(prompt_lists[1]) else dummy_prompt_token_ids
        token_ids = [encode_fn(prompt)[:args.input_len] for prompt in prompts]
        print(f"max token_ids len: {len(max(token_ids, key=len))}")
        latencies.append(run_to_completion(token_ids, profile=True))

    for latency in latencies:
        print(f"Avg latency: {latency} seconds.")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Benchmark the latency of processing a single batch of '
                    'requests till completion.')
    parser.add_argument('--model', type=str, default='facebook/opt-125m')
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--input-len', type=int, default=1024)
    parser.add_argument('--output-len', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=3)
    parser.add_argument('--n', type=int, default=1,
                        help='Number of generated sequences per prompt.')
    parser.add_argument('--use-beam-search', action='store_true')
    parser.add_argument('--num-iters', type=int, default=3,
                        help='Number of iterations to run.')
    args = parser.parse_args()
    main(args)