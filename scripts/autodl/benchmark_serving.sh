#!/usr/bin/env bash

set -euo pipefail
source "$(dirname -- "$0")/common.sh"
activate_minivllm_env
configure_cuda
enter_repo

topology="${BENCH_TOPOLOGY:-unified}"
case "${topology}" in
  unified) infos=(build/benchmarks/server-u0.json build/benchmarks/server-u1.json) ;;
  pd) infos=(build/benchmarks/server-p.json build/benchmarks/server-d.json) ;;
  *) echo "BENCH_TOPOLOGY must be unified or pd" >&2; exit 2 ;;
esac
data_args=()
if [[ "${SYNTHETIC:-0}" == "1" ]]; then
  data_args+=(--synthetic)
else
  : "${DATASET:?Set DATASET, or SYNTHETIC=1 for smoke testing}"
  data_args+=(--dataset "${DATASET}")
fi
exec python -m benchmarks.benchmark_serving \
  --topology "${topology}" --server-info "${infos[@]}" \
  --control-authkey "${PD_AUTHKEY:-mini-vllm}" \
  --short-requests "${SHORT_REQUESTS:-8}" --long-requests "${LONG_REQUESTS:-8}" \
  --short-input-len "${SHORT_INPUT_LEN:-128}" --long-input-len "${LONG_INPUT_LEN:-4096}" \
  --short-output-len "${SHORT_OUTPUT_LEN:-256}" --long-output-len "${LONG_OUTPUT_LEN:-1}" \
  --long-delay "${LONG_DELAY:-1.0}" --long-interval "${LONG_INTERVAL:-0.1}" \
  --warmup "${WARMUP:-2}" --seed "${SEED:-42}" \
  --output "${BENCH_OUTPUT:-build/benchmarks/${topology}.json}" \
  "${data_args[@]}" "$@"
