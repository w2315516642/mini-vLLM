#!/usr/bin/env bash

set -euo pipefail
source "$(dirname -- "$0")/common.sh"
activate_minivllm_env
configure_cuda
enter_repo
export CUDA_DEVICE_ORDER=PCI_BUS_ID

mode="${BENCH_MODE:-target}"
mode_args=()
case "${mode}" in
  target) ;;
  dspark-static|dspark-adaptive)
    mode_args+=(--draft-model "${DRAFT_MODEL}" --num-speculative-tokens "${NUM_SPECULATIVE_TOKENS:-7}")
    if [[ "${mode}" == "dspark-adaptive" ]]; then
      mode_args+=(--speculative-adaptive)
    fi
    ;;
  *) echo "BENCH_MODE must be target, dspark-static, or dspark-adaptive" >&2; exit 2 ;;
esac
data_args=()
if [[ "${SYNTHETIC:-0}" == "1" ]]; then
  data_args+=(--synthetic)
else
  : "${DATASET:?Set DATASET to a JSON/JSONL file, or SYNTHETIC=1 for smoke testing}"
  data_args+=(--dataset "${DATASET}")
fi
if [[ "${PREFIX_PRIME:-0}" == "1" ]]; then
  mode_args+=(--enable-prefix-caching --prime-prefix)
fi
exec env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES:-0,1}" python -m benchmarks.benchmark_generation \
  --model "${TARGET_MODEL}" --dtype "${DTYPE:-bfloat16}" \
  --tensor-parallel-size "${TP_SIZE:-2}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
  --max-num-seqs "${MAX_NUM_SEQS:-${BATCH_SIZE:-1}}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}" \
  --input-len "${INPUT_LEN:-2048}" --output-len "${OUTPUT_LEN:-128}" \
  --batch-size "${BATCH_SIZE:-1}" --num-batches "${NUM_BATCHES:-10}" \
  --warmup "${WARMUP:-2}" --seed "${SEED:-42}" \
  --output "${BENCH_OUTPUT:-build/benchmarks/${mode}-b${BATCH_SIZE:-1}.json}" \
  "${data_args[@]}" "${mode_args[@]}" "$@"
