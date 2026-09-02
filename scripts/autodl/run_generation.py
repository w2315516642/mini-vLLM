#!/usr/bin/env python3

import argparse
import json
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run or benchmark Qwen3.8 with optional DSpark."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--num-speculative-tokens", type=int, default=0)
    parser.add_argument("--speculative-adaptive", action="store_true")
    parser.add_argument("--prompt", default="用三句话说明投机解码。")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--requests", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.requests <= 0 or args.warmup < 0:
        raise ValueError("requests must be positive and warmup non-negative")

    from minivllm import LLM, SamplingParams

    engine_kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "num_speculative_tokens": args.num_speculative_tokens,
    }
    if args.draft_model:
        engine_kwargs.update(
            draft_model=args.draft_model,
            speculative_adaptive=args.speculative_adaptive,
        )

    load_started = time.perf_counter()
    llm = LLM(**engine_kwargs)
    load_seconds = time.perf_counter() - load_started
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    def generate_once():
        started = time.perf_counter()
        result = llm.generate(
            [args.prompt], sampling_params, use_tqdm=False
        )[0]
        return result, time.perf_counter() - started

    for _ in range(args.warmup):
        generate_once()

    latencies = []
    total_tokens = 0
    last_output = None
    for _ in range(args.requests):
        last_output, elapsed = generate_once()
        latencies.append(elapsed)
        total_tokens += len(last_output.outputs[0].token_ids)

    assert last_output is not None
    print(last_output.outputs[0].text)
    print(
        json.dumps(
            {
                "mode": "dspark" if args.draft_model else "target",
                "load_seconds": load_seconds,
                "requests": args.requests,
                "generated_tokens": total_tokens,
                "mean_latency_seconds": sum(latencies) / len(latencies),
                "generation_tokens_per_second": total_tokens / sum(latencies),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
