#!/usr/bin/env bash
set -euo pipefail
: "${DATASET:?Set DATASET to the prepared held-out JSONL}"
source "$(dirname -- "$0")/common.sh"
enter_repo
export TARGET_MODEL
original_draft="${ORIGINAL_DRAFT_MODEL:-${MODEL_ROOT}/Qwen3.8-27B-DSpark}"
finetuned_draft="${FINETUNED_DRAFT_MODEL:-${MODEL_ROOT}/Qwen3.8-27B-DSpark-ft20}"
export CUDA_DEVICES=0 TP_SIZE=1 DTYPE=bfloat16
export BENCH_MODE=dspark-static LOAD_MODE=batch
export BATCH_SIZE=1 MAX_NUM_SEQS=1 NUM_BATCHES=100
export INPUT_LEN=512 OUTPUT_LEN=128 NUM_SPECULATIVE_TOKENS=3
export MAX_NUM_BATCHED_TOKENS=2048 GPU_MEMORY_UTILIZATION=0.85
export WARMUP=2 SEED=42 SYNTHETIC=0 PREFIX_PRIME=0
for round in 1 2; do
  # Reverse run order to reduce systematic warm-cache/order bias.
  order=(original ft20)
  if [[ "$round" == 2 ]]; then order=(ft20 original); fi
  for variant in "${order[@]}"; do
    export DRAFT_MODEL="${original_draft}"
    if [[ "$variant" == ft20 ]]; then DRAFT_MODEL="${finetuned_draft}"; fi
    export BENCH_OUTPUT="build/benchmarks/heldout-${variant}-b1-r${round}.json"
    if [[ -e "$BENCH_OUTPUT" ]]; then
      echo "Refusing to overwrite $BENCH_OUTPUT" >&2
      exit 1
    fi
    bash scripts/autodl/benchmark_generation.sh
  done
done
