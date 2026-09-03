"""Replay the same timed workload against unified replicas or a P/D pair."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

from benchmarks.benchmark_utils import WorkItem, prepare_prompts, write_result
from benchmarks.serving_runner import run_pd, run_unified
from benchmarks.streaming_metrics import speculative_delta


def main():
    from minivllm import SamplingParams
    from minivllm.engine.pd_rpc import RemoteEngineClient
    from minivllm.engine.pd_runtime import PDEngineBridge
    from minivllm.engine.tokenizer_utils import get_tokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", choices=["unified", "pd"], required=True)
    parser.add_argument("--server-info", nargs="+", required=True)
    parser.add_argument("--control-authkey", required=True)
    parser.add_argument("--rpc-timeout", type=float, default=300)
    parser.add_argument("--timeout", type=float, default=600, help="Total measured run timeout in seconds")
    parser.add_argument("--dataset")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--short-requests", type=int, default=8)
    parser.add_argument("--long-requests", type=int, default=8)
    parser.add_argument("--short-input-len", type=int, default=128)
    parser.add_argument("--long-input-len", type=int, default=4096)
    parser.add_argument("--short-output-len", type=int, default=256)
    parser.add_argument("--long-output-len", type=int, default=1)
    parser.add_argument("--long-delay", type=float, default=1.0)
    parser.add_argument("--long-interval", type=float, default=.1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if bool(args.dataset) == args.synthetic:
        parser.error("Choose exactly one of --dataset and --synthetic")
    if min(args.short_requests, args.short_input_len, args.long_input_len,
           args.short_output_len, args.long_output_len) <= 0:
        parser.error("Short request count and token lengths must be positive")
    if min(args.long_requests, args.warmup, args.long_delay, args.long_interval) < 0:
        parser.error("Counts and arrival delays must be non-negative")
    if args.timeout <= 0 or args.rpc_timeout <= 0:
        parser.error("Timeouts must be positive")
    servers = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.server_info]
    endpoints = [x["endpoint"] for x in servers]
    if len(set(endpoints)) != len(endpoints):
        parser.error("Server endpoints must be distinct")
    roles = [s["config"]["pd_role"] for s in servers]
    if args.topology == "pd" and roles != ["prefill", "decode"]:
        parser.error("PD requires prefill and decode info files, in that order")
    if args.topology == "unified" and any(role != "unified" for role in roles):
        parser.error("Unified benchmark requires unified server info files")
    models = {s["config"]["model"] for s in servers}
    if len(models) != 1:
        parser.error("All servers must use the same target model")
    tokenizer = get_tokenizer(next(iter(models)))
    workload = []
    for name, count, length, output_len, delay, interval in (
        ("decode", args.short_requests, args.short_input_len, args.short_output_len, 0., 0.),
        ("prefill_load", args.long_requests, args.long_input_len, args.long_output_len,
         args.long_delay, args.long_interval),
    ):
        if not count:
            continue
        prompts = prepare_prompts(tokenizer, count, length, dataset=args.dataset,
                                  synthetic=args.synthetic, seed=args.seed)
        workload.extend(WorkItem(f"{name}-{i}", ids, output_len, delay + i * interval, name)
                        for i, ids in enumerate(prompts))
    params_factory = lambda length: SamplingParams(temperature=0., ignore_eos=True, max_tokens=length)
    with ExitStack() as stack:
        clients = []
        for server in servers:
            client = RemoteEngineClient(server["endpoint"], args.control_authkey.encode(), args.rpc_timeout)
            stack.callback(client.close)
            if client.get_runtime_stats()["config"] != server["config"]:
                raise RuntimeError("Server configuration changed; regenerate its info file")
            clients.append(client)

        def run(items):
            if args.topology == "pd":
                return run_pd(*clients, items, params_factory, PDEngineBridge(*clients), args.timeout)
            return run_unified(clients, items, params_factory, args.timeout)

        warm_prompts = prepare_prompts(tokenizer, len(clients), args.short_input_len,
                                       synthetic=True, seed=args.seed + 10000)
        for iteration in range(args.warmup):
            run([WorkItem(f"warm-{iteration}-{i}", ids, 8) for i, ids in enumerate(warm_prompts)])
        before = [client.get_runtime_stats() for client in clients]
        traces, window = run(workload)
        after = [client.get_runtime_stats() for client in clients]
    config = {k: v for k, v in vars(args).items() if k != "control_authkey"}
    config.update(benchmark="serving", temperature=0., ignore_eos=True)
    write_result(args.output, traces=traces, windows=[window], config=config,
                 workload=workload, speculative=speculative_delta(before, after), servers=servers)


if __name__ == "__main__":
    main()
