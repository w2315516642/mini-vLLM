"""Summarize worker stage timelines, separating prefill from decode steps."""

import argparse
import json
from collections import defaultdict


def summarize(steps):
    groups = {}
    for label in ("prefill_or_mixed", "decode"):
        rows = [s for s in steps if bool(s["counts"]["prefill_requests"])
                == (label == "prefill_or_mixed")]
        totals = defaultdict(lambda: {"stream_ms": 0.0, "host_ms": 0.0})
        counts = defaultdict(int)
        for row in rows:
            for name, values in row["stages"].items():
                for metric, value in values.items():
                    totals[name][metric] += value
            for name, value in row["counts"].items():
                counts[name] += value
        groups[label] = {"steps": len(rows), "count_totals": dict(counts),
                         "stage_totals": dict(totals)}
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    for path in args.paths:
        with open(path, encoding="utf-8") as source:
            result = json.load(source)
        print(json.dumps({"path": path, "note": result["note"], "ranks": [
            {"rank": i, "limit_reached": rank["limit_reached"],
             "summary": summarize(rank["steps"])}
            for i, rank in enumerate(result["ranks"])]}, indent=2))


if __name__ == "__main__":
    main()
