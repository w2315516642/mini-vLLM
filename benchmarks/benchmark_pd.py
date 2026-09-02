"""Measure end-to-end latency through separately launched P/D engines."""

import argparse
import json
import statistics
import time

from minivllm.engine.pd_rpc import PDClient
from minivllm.sampling_params import SamplingParams


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(int(len(ordered) * fraction), len(ordered) - 1)
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill-control", required=True)
    parser.add_argument("--decode-control", required=True)
    parser.add_argument("--control-authkey", required=True)
    parser.add_argument("--prompt", default="Explain paged KV cache briefly.")
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()
    if args.requests <= 0 or args.warmup < 0:
        parser.error("requests must be positive and warmup non-negative")
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    client = PDClient(
        args.prefill_control,
        args.decode_control,
        args.control_authkey.encode("utf-8"),
    )
    latencies = []
    ttfts = []
    tpots = []
    transfers = []
    try:
        for index in range(args.warmup + args.requests):
            started = time.perf_counter()
            _, metrics = client.generate_with_metrics(args.prompt, params)
            elapsed = time.perf_counter() - started
            if index >= args.warmup:
                latencies.append(elapsed)
                ttfts.append(metrics.ttft_s)
                tpots.append(metrics.tpot_s)
                transfers.append(metrics.transfer_s)
    finally:
        client.close()
    print(
        json.dumps(
            {
                "requests": len(latencies),
                "mean_latency_s": statistics.mean(latencies),
                "p50_latency_s": percentile(latencies, 0.50),
                "p95_latency_s": percentile(latencies, 0.95),
                "mean_ttft_s": statistics.mean(ttfts),
                "mean_tpot_s": statistics.mean(tpots),
                "mean_transfer_s": statistics.mean(transfers),
                "throughput_requests_per_s": len(latencies) / sum(latencies),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
