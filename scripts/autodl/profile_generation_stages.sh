#!/usr/bin/env bash
set -euo pipefail

# Match the B16 width scan; each invocation profiles just one configuration.
export CUDA_DEVICES="${CUDA_DEVICES:-0}" TP_SIZE="${TP_SIZE:-1}"
export BATCH_SIZE="${BATCH_SIZE:-16}" MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
export INPUT_LEN="${INPUT_LEN:-512}" OUTPUT_LEN="${OUTPUT_LEN:-128}"
export NUM_BATCHES="${NUM_BATCHES:-3}" WARMUP="${WARMUP:-2}"
name="${BENCH_MODE:-target}-b${BATCH_SIZE}-k${NUM_SPECULATIVE_TOKENS:-7}"
export BENCH_OUTPUT="${BENCH_OUTPUT:-build/benchmarks/profile-${name}.json}"
exec bash "$(dirname -- "$0")/benchmark_generation.sh" \
  --stage-profile-output "${STAGE_PROFILE_OUTPUT:-build/benchmarks/stages-${name}.json}" \
  --stage-profile-max-steps "${STAGE_PROFILE_MAX_STEPS:-2048}" "$@"
