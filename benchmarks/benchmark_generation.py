"""Measure local streaming batches with or without DSpark."""

import argparse
import json
from pathlib import Path
from contextlib import closing
import time

from benchmarks.benchmark_utils import WorkItem, prepare_prompts, write_result
from benchmarks.streaming_metrics import RequestTrace, speculative_delta


def measure_batch(llm, prompts, params):
    traces = {}
    started = time.perf_counter()
    with closing(llm.generate_stream(
        prompt_token_ids=prompts, sampling_params=params, use_tqdm=False,
    )) as stream:
        for output in stream:
            now = time.perf_counter()
            trace = traces.setdefault(output.request_id, RequestTrace(
                output.request_id, len(output.prompt_token_ids), started))
            trace.observe(output, now, cumulative=False)
    ended = time.perf_counter()
    if len(traces) != len(prompts) or any(t.finished_at is None for t in traces.values()):
        raise RuntimeError("Batch ended without all final outputs")
    if any(t.output_tokens != params.max_tokens for t in traces.values()):
        raise RuntimeError("Generated token count differs from the fixed benchmark workload")
    return list(traces.values()), (started, ended)


def measure_continuous(llm, prompts, params, concurrency):
    """Closed-loop load: refill finished slots before the next engine step.

    Time includes admission, prefill and the final drain. Requests not yet
    admitted have no submission timestamp: this is not an open-loop load test.
    """
    from minivllm.engine.output_processor import OutputProcessor

    if concurrency <= 0 or not prompts:
        raise ValueError("Continuous load requires prompts and positive concurrency")
    engine = llm.llm_engine
    if llm._generation_active or engine.has_unfinished_requests():
        raise RuntimeError("Continuous benchmark requires an idle engine")
    processor = OutputProcessor(params)
    traces, pending = {}, set()
    next_prompt = 0
    started = time.perf_counter()
    llm._generation_active = True
    try:
        while next_prompt < len(prompts) or pending:
            while next_prompt < len(prompts) and len(pending) < concurrency:
                ids = prompts[next_prompt]
                submitted = time.perf_counter()
                request_id = llm._add_request(None, params, ids)
                pending.add(request_id)
                traces[request_id] = RequestTrace(request_id, len(ids), submitted)
                next_prompt += 1
            if not engine.has_unfinished_requests():
                raise RuntimeError("Engine ended without all final outputs")
            # Use the same delta conversion as generate_stream. Consume the
            # whole step before refilling; no extra GPU synchronization is added.
            for snapshot in engine.step():
                output = processor.process(snapshot)
                if output is None:
                    continue
                trace = traces[output.request_id]
                trace.observe(output, time.perf_counter(), cumulative=False)
                if output.is_finished():
                    pending.remove(output.request_id)
                    if trace.output_tokens != params.max_tokens:
                        raise RuntimeError("Generated token count differs from the fixed benchmark workload")
        ended = time.perf_counter()
    finally:
        llm._generation_active = False
        for request_id in pending:
            engine.abort_request(request_id)
    return list(traces.values()), (started, ended)


def main():
    from minivllm import LLM, SamplingParams
    from minivllm.engine.arg_utils import EngineArgs

    parser = argparse.ArgumentParser(description=__doc__)
    EngineArgs.add_cli_args(parser)
    parser.add_argument("--dataset", help="JSONL prompt records or ShareGPT JSON")
    parser.add_argument("--synthetic", action="store_true", help="Random-token smoke benchmark only")
    parser.add_argument("--input-len", type=int, default=2048)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--load-mode", choices=("batch", "continuous"), default="batch",
                        help="Fixed batches or closed-loop refill at --batch-size concurrency")
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prime-prefix", action="store_true", help="Prefill each measured batch before timing")
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage-profile-output", help="Opt-in worker stage timing JSON")
    parser.add_argument("--stage-profile-max-steps", type=int, default=2048)
    parser.add_argument("--nsys-capture-output", help="Enable short Decode capture; write window metadata")
    parser.add_argument("--nsys-skip-steps", type=int, default=5)
    parser.add_argument("--nsys-capture-steps", type=int, default=20)
    parser.set_defaults(max_num_seqs=16, disable_log_stats=True)
    args = parser.parse_args()
    if min(args.input_len, args.output_len, args.batch_size, args.num_batches) <= 0 or args.warmup < 0:
        parser.error("Lengths, batch size and batch count must be positive; warmup must be non-negative")
    if bool(args.dataset) == args.synthetic:
        parser.error("Choose exactly one of --dataset and --synthetic")
    if args.pd_role != "unified":
        parser.error("Use benchmark_serving for PD roles")
    if args.prime_prefix and not args.enable_prefix_caching:
        parser.error("--prime-prefix requires --enable-prefix-caching")
    if args.load_mode == "continuous" and args.prime_prefix:
        parser.error("Continuous load does not support per-batch prefix priming")
    if args.load_mode == "continuous" and args.batch_size > args.max_num_seqs:
        parser.error("Continuous concurrency must not exceed --max-num-seqs")
    if args.stage_profile_max_steps <= 0:
        parser.error("--stage-profile-max-steps must be positive")
    if args.stage_profile_output and args.prime_prefix:
        parser.error("Stage profiling does not include prefix priming; disable --prime-prefix")
    if args.nsys_skip_steps < 0 or args.nsys_capture_steps <= 0:
        parser.error("Nsight skip must be nonnegative and capture steps positive")
    if args.nsys_capture_output and (
        args.tensor_parallel_size != 1 or args.pipeline_parallel_size != 1
        or args.worker_use_ray or args.prime_prefix or args.stage_profile_output
    ):
        parser.error("Nsight capture requires one local GPU, no priming or stage-event profiling")
    engine_args = EngineArgs.from_cli_args(args)
    llm = LLM(**vars(engine_args))
    tokenizer = llm.get_tokenizer()
    count = args.batch_size * args.num_batches
    prompts = prepare_prompts(tokenizer, count, args.input_len, dataset=args.dataset,
                              synthetic=args.synthetic, seed=args.seed)
    params = SamplingParams(temperature=args.temperature, ignore_eos=True, max_tokens=args.output_len)
    # Warmup inputs differ from measured inputs, so warming CUDA does not
    # silently turn the first Prefix Cache measurement into a cache hit.
    warmup_prompts = prepare_prompts(tokenizer, args.batch_size, args.input_len,
                                     synthetic=True, seed=args.seed + 10000)
    for _ in range(args.warmup):
        llm.generate(prompt_token_ids=warmup_prompts, sampling_params=params, use_tqdm=False)
    before = [llm.llm_engine.get_runtime_stats()]
    if args.stage_profile_output:
        llm.llm_engine._run_workers("start_stage_profile",
            max_steps=args.stage_profile_max_steps)
    if args.nsys_capture_output:
        llm.llm_engine._run_workers("start_decode_capture",
            skip=args.nsys_skip_steps, steps=args.nsys_capture_steps)
    traces, windows = [], []
    try:
        offsets = range(0, count, args.batch_size) if args.load_mode == "batch" else ()
        if args.load_mode == "continuous":
            observed, window = measure_continuous(llm, prompts, params, args.batch_size)
            traces.extend(observed)
            windows.append(window)
        for offset in offsets:
            batch = prompts[offset:offset + args.batch_size]
            if args.prime_prefix:
                llm.generate(prompt_token_ids=batch, sampling_params=SamplingParams(
                    temperature=0.0, ignore_eos=True, max_tokens=1), use_tqdm=False)
            observed, window = measure_batch(llm, batch, params)
            traces.extend(observed)
            windows.append(window)
    finally:
        if args.nsys_capture_output:
            ranks = llm.llm_engine._run_workers("finish_decode_capture", get_all_outputs=True)
            target = Path(args.nsys_capture_output)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps({"config": vars(args), "ranks": ranks}, indent=2),
                              encoding="utf-8")
    after = [llm.llm_engine.get_runtime_stats()]
    if args.stage_profile_output:
        ranks = llm.llm_engine._run_workers("finish_stage_profile", get_all_outputs=True)
        target = Path(args.stage_profile_output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"config": vars(args), "ranks": ranks,
            "note": "Instrumented run. stream_ms includes idle gaps; host_ms includes host waits. "
                    "They overlap and must not be added. Warmup excluded; prefill included."},
            indent=2), encoding="utf-8")
    workload = [WorkItem(str(i), ids, args.output_len) for i, ids in enumerate(prompts)]
    write_result(args.output, traces=traces, windows=windows,
                 config={**vars(args), "benchmark": "generation", "ignore_eos": True,
                         "includes_prefill_and_drain": True},
                 workload=workload, speculative=speculative_delta(before, after),
                 servers=[{"config": after[0]["config"]}])


if __name__ == "__main__":
    main()
