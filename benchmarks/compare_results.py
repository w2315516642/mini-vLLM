"""Compare like-for-like benchmark results; refuse ambiguous denominators."""

import argparse
import json
from pathlib import Path

from benchmarks.benchmark_utils import selected_gpus


def compare_results(baseline, candidate, allow_config_difference=()):
    if baseline["schema_version"] != 1 or candidate["schema_version"] != 1:
        raise ValueError("Unsupported result schema")
    if baseline["workload"] != candidate["workload"]:
        raise ValueError("Workloads differ: token inputs, output lengths or arrival schedule changed")
    for key in ("benchmark", "synthetic", "temperature", "ignore_eos", "batch_size", "num_batches"):
        if baseline["config"].get(key) != candidate["config"].get(key):
            raise ValueError(f"Benchmark configuration differs: {key}")
    experiment = baseline["config"]["benchmark"]
    expected_changes = {"pd_role"}
    if experiment == "generation":
        expected_changes.update({"draft_model", "num_speculative_tokens", "speculative_adaptive",
                                 "speculative_min_survival", "speculative_cost_token_counts",
                                 "speculative_cost_latency_ms"})
    changes = expected_changes | set(allow_config_difference)
    configs = [server["config"] for r in (baseline, candidate) for server in r["servers"]]
    for config in configs[1:]:
        for key in set(configs[0]) | set(config):
            if key not in changes and configs[0].get(key) != config.get(key):
                raise ValueError(f"Engine configuration differs: {key}; allow this difference explicitly")

    hardware = []
    for result in (baseline, candidate):
        devices = []
        for server in result["servers"]:
            env = server.get("environment", result["environment"])
            devices.extend(selected_gpus(env, server["config"]["tensor_parallel_size"]))
            for key in ("torch", "torch_cuda"):
                if env.get(key) != baseline["servers"][0].get(
                    "environment", baseline["environment"]).get(key):
                    raise ValueError(f"Runtime versions differ: {key}")
        if len({row[0] for row in devices}) != len(devices):
            raise ValueError("Engine instances share a GPU; this is not a dedicated-resource PD comparison")
        hardware.append(sorted(devices))
    if hardware[0] != hardware[1]:
        raise ValueError("GPU UUID/model/memory/driver or total GPU count differs")
    if any(r["metrics"]["completed_requests"] != r["metrics"]["requests"] for r in (baseline, candidate)):
        raise ValueError("Cannot compare incomplete runs")
    if experiment == "serving" and baseline["config"].get("long_requests", 0):
        if not all(r["decode_requests_overlapping_prefill_load"] for r in (baseline, candidate)):
            raise ValueError("Prefill/decode load did not overlap in both experiments")

    def metric(result, name):
        if name == "output_tokens_per_s":
            return result["metrics"][name]
        root = result["cohorts"].get("decode", result["metrics"])
        key, aggregation = name.split(".")
        return root[key][aggregation] if root[key] else None

    rows = {}
    for name, higher_is_better in (
        ("ttft_ms.mean", False), ("decode_tokens_per_s.mean", True),
        ("output_tokens_per_s", True), ("tpot_ms.mean", False), ("itl_ms.p99", False),
    ):
        old, new = metric(baseline, name), metric(candidate, name)
        if old is None or new is None or old <= 0 or new <= 0:
            continue
        rows[name] = {"baseline": old, "candidate": new,
                      "speedup": new / old if higher_is_better else old / new,
                      "improvement_percent": 100 * ((new / old - 1) if higher_is_better else (1 - new / old))}
    return {"metrics": rows, "allowed_config_differences": sorted(allow_config_difference),
            "candidate_speculative": candidate["speculative"],
            "candidate_handoff_ms": candidate["metrics"]["handoff_ms"],
            "warnings": baseline["warnings"] + candidate["warnings"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--allow-config-difference", action="append", default=[])
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        result = compare_results(
            json.loads(Path(args.baseline).read_text(encoding="utf-8")),
            json.loads(Path(args.candidate).read_text(encoding="utf-8")),
            args.allow_config_difference)
    except ValueError as exc:
        parser.error(str(exc))
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
