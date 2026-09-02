#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-vllm}"
CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
MODEL_ROOT="${MODEL_ROOT:-/root/autodl-tmp/models}"
TARGET_MODEL="${TARGET_MODEL:-${MODEL_ROOT}/Qwen3.8-27B-FP8}"
DRAFT_MODEL="${DRAFT_MODEL:-${MODEL_ROOT}/Qwen3.8-27B-DSpark}"
PD_SMOKE_MODEL="${PD_SMOKE_MODEL:-${MODEL_ROOT}/TinyLlama-1.1B-Chat-v1.0}"

activate_minivllm_env() {
  if [[ -f "${CONDA_SH}" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  else
    echo "找不到 conda，请设置 CONDA_SH。" >&2
    return 1
  fi
  conda activate "${CONDA_ENV}"
}

configure_cuda() {
  export CUDA_HOME
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  export TMPDIR="${TMPDIR:-/tmp}"
  mkdir -p "${TMPDIR}"
}

enter_repo() {
  cd "${REPO_ROOT}"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "缺少命令：$1" >&2
    return 1
  fi
}
