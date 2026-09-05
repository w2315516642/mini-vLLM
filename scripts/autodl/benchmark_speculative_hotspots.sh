#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/common.sh"
activate_minivllm_env
configure_cuda
enter_repo
exec env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES:-0}" \
  python -m benchmarks.benchmark_speculative_hotspots "$@"
