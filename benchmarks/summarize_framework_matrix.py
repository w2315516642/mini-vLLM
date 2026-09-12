"""Summarize measured ratios and token agreement, without inventing kernel causes."""
import argparse
import json
from pathlib import Path


def compare_outputs(left, right):
    if left["workload"] != right["workload"]:
        raise ValueError("Cannot compare different workloads")
    a = {r["request_id"]: r["token_ids"] for r in left["requests"]}
    b = {r["request_id"]: r["token_ids"] for r in right["requests"]}
    if a.keys() != b.keys():
        raise ValueError("Request IDs differ")
    differences = []
    for rid in sorted(a, key=int):
        if a[rid] != b[rid]:
            first = next((i for i, pair in enumerate(zip(a[rid], b[rid])) if pair[0] != pair[1]),
                         min(len(a[rid]), len(b[rid])))
            differences.append({"request_id": rid, "first_difference": first})
    return {"requests": len(a), "exact_matches": len(a) - len(differences), "differences": differences}


def summarize(directory):
    files = {}
    for path in sorted(directory.glob("*-b*-r*.json")):
        if path.name.endswith("-commands.json"):
            continue
        files[path.stem] = json.loads(path.read_text())
    if not files:
        raise ValueError("No matrix result files")
    report = ["# Framework Comparison", "", "| Run | token/s | TPOT ms | TTFT ms | Accepted/round |",
              "|---|---:|---:|---:|---:|"]
    checks = {}
    for name, data in files.items():
        m = data["metrics"]
        accepted = data["speculative"]["accepted_tokens_per_round"]
        report.append(f'| {name} | {m["output_tokens_per_s"]:.3f} | {m["tpot_ms"]["mean"]:.3f} | '
                      f'{m["ttft_ms"]["mean"]:.3f} | {accepted if accepted is not None else "N/A"} |')
    report += ["", "## Ratios and Correctness", ""]
    for name, data in files.items():
        base = None
        if "-dspark-" in name:
            base = name.replace("-dspark-", "-target-")
        if base in files:
            checks[name + " vs " + base] = compare_outputs(data, files[base])
            ratio = data["metrics"]["output_tokens_per_s"] / files[base]["metrics"]["output_tokens_per_s"]
            report.append(f"- {name} / {base}: **{ratio:.3f}x** throughput.")
        if name.startswith("vllm-"):
            base = name.replace("vllm-", "mini-", 1)
            if base in files:
                checks[name + " vs " + base] = compare_outputs(data, files[base])
                ratio = data["metrics"]["output_tokens_per_s"] / files[base]["metrics"]["output_tokens_per_s"]
                report.append(f"- {name} / {base}: **{ratio:.3f}x** throughput.")
    for pair, check in checks.items():
        report.append(f'- {pair}: {check["exact_matches"]}/{check["requests"]} exact token matches.')
    report += ["", "Instrumented runs are not baseline performance. Client inflight is not GPU batch size.",
               "A throughput difference alone does not identify a kernel bottleneck; inspect profiles and shapes."]
    return "\n".join(report) + "\n", checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report, checks = summarize(args.directory)
    (args.directory / "summary.md").write_text(report, encoding="utf-8")
    (args.directory / "token_comparison.json").write_text(json.dumps(checks, indent=2))
    print(report)


if __name__ == "__main__":
    main()
