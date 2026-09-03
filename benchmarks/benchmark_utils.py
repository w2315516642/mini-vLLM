"""Workload preparation and result files shared by local and RPC benchmarks."""

from dataclasses import dataclass, asdict
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time

from benchmarks.streaming_metrics import summarize


@dataclass(frozen=True)
class WorkItem:
    request_id: str
    token_ids: list[int]
    output_len: int
    arrival_s: float = 0.0
    cohort: str = "default"


def prompts_from_records(records):
    for item in records:
        if isinstance(item, str):
            yield item
        elif "prompt" in item:
            yield item["prompt"]
        elif "prompts" in item:
            yield from item["prompts"]
        elif "conversations" in item:
            turns = item["conversations"]
            if turns:
                yield turns[0]["value"]
        else:
            raise ValueError("Dataset requires prompt, prompts, or ShareGPT conversations")


def prepare_prompts(tokenizer, count, length, *, dataset=None, synthetic=False, seed=42):
    if count <= 0 or length <= 0:
        raise ValueError("Prompt count and length must be positive")
    rng = random.Random(seed)
    if synthetic:
        # Random IDs exercise shapes, not a meaningful DSpark acceptance rate.
        special = set(tokenizer.all_special_ids)
        vocabulary = sorted(set(tokenizer.get_vocab().values()) - special)
        if not vocabulary:
            raise ValueError("Tokenizer has no non-special tokens")
        return [rng.choices(vocabulary, k=length) for _ in range(count)]
    if not dataset:
        raise ValueError("Provide --dataset, or explicitly use --synthetic for a smoke benchmark")
    path = Path(dataset)
    with path.open(encoding="utf-8") as source:
        if path.suffix == ".jsonl":
            records = [json.loads(line) for line in source if line.strip()]
        else:
            records = json.load(source)
    if not isinstance(records, list):
        raise ValueError("Dataset must be a JSON array or JSONL records")
    rng.shuffle(records)
    candidates = []
    for prompt in prompts_from_records(records):
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) >= length:
            candidates.append(ids[:length])
            if len(candidates) == count:
                break
    if not candidates:
        raise ValueError(f"No dataset prompt has at least {length} tokens; no padding is applied")
    rng.shuffle(candidates)
    # Resampling is deterministic; the result records the number of unique inputs.
    return [list(candidates[i % len(candidates)]) for i in range(count)]


def workload_info(workload):
    payload = json.dumps([asdict(x) for x in workload], sort_keys=True, separators=(",", ":"))
    return {"sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "requests": len(workload),
            "unique_prompts": len({tuple(x.token_ids) for x in workload}),
            "input_lengths": sorted({len(x.token_ids) for x in workload}),
            "output_lengths": sorted({x.output_len for x in workload})}


def command_output(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def environment_info():
    import torch
    return {
        "hostname": socket.gethostname(), "python": sys.version.split()[0],
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(command_output(["git", "status", "--porcelain"])),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "gpu_inventory": command_output([
            "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version", "--format=csv,noheader"]),
    }


def selected_gpus(environment, tp_size):
    """Resolve explicitly visible devices without creating a CUDA context."""
    inventory = environment.get("gpu_inventory")
    visible = environment.get("cuda_visible_devices")
    if not inventory or not visible:
        raise ValueError("GPU inventory/CUDA_VISIBLE_DEVICES missing; cannot verify hardware equality")
    devices = [part.strip() for part in visible.split(",")]
    if len(devices) != tp_size:
        raise ValueError("Visible GPU count must equal TP size for an auditable benchmark")
    rows = [[part.strip() for part in row] for row in csv.reader(inventory.splitlines())]
    selected = []
    for device in devices:
        if device.isdigit() and environment.get("cuda_device_order") != "PCI_BUS_ID":
            raise ValueError("Set CUDA_DEVICE_ORDER=PCI_BUS_ID when selecting GPUs by index")
        matches = [row for row in rows if row[0] == device or row[1] == device]
        if len(matches) != 1:
            raise ValueError(f"Cannot uniquely resolve GPU {device}")
        selected.append(tuple(matches[0][1:]))
    return selected


def write_result(path, *, traces, windows, config, workload, speculative, servers):
    metrics, records = summarize(traces, windows)
    cohorts = {}
    for name in sorted({trace.cohort for trace in traces}):
        cohorts[name] = summarize([t for t in traces if t.cohort == name], windows)[0]
    warnings = []
    if config.get("synthetic"):
        warnings.append("Synthetic random tokens: not a natural-workload DSpark/resume result.")
    if metrics["itl_ms"] and metrics["itl_ms"]["count"] < 10000:
        warnings.append("Fewer than 10000 ITL samples; inspect sample count before quoting P99.")
    decode_traces = [t for t in traces if t.cohort == "decode"]
    load_traces = [t for t in traces if t.cohort == "prefill_load"]
    overlaps = sum(any(d.events and p.events and
                       max(d.events[0][0], p.submitted_at) < min(d.events[-1][0], p.events[0][0])
                       for p in load_traces) for d in decode_traces)
    if load_traces and not overlaps:
        warnings.append("No decode/prefill-load overlap observed; tune --long-delay/output lengths.")
    result = {"schema_version": 1, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "config": config, "environment": environment_info(), "servers": servers,
              "workload": workload_info(workload), "metrics": metrics,
              "cohorts": cohorts, "speculative": speculative, "warnings": warnings,
              "decode_requests_overlapping_prefill_load": overlaps}
    origin = min(start for start, _ in windows)
    result["measurement_windows_s"] = [[start - origin, end - origin] for start, end in windows]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw_path = target.with_suffix(".requests.jsonl")
    with raw_path.open("w", encoding="utf-8") as raw:
        for record in records:
            raw.write(json.dumps(record, ensure_ascii=False) + "\n")
    result["request_records"] = str(raw_path)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(target), "metrics": metrics,
                      "speculative": speculative, "warnings": warnings}, ensure_ascii=False, indent=2))
    return result
