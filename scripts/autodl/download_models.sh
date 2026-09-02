#!/usr/bin/env bash

set -euo pipefail
source "$(dirname -- "$0")/common.sh"

activate_minivllm_env
configure_cuda
mkdir -p "${MODEL_ROOT}"

PIP_INDEX="${MINIVLLM_PIP_INDEX:-https://pypi.org/simple}"
TARGET_SOURCE="${TARGET_SOURCE:-modelscope}"
DOWNLOAD_TARGET="${DOWNLOAD_TARGET:-1}"
DOWNLOAD_DRAFT="${DOWNLOAD_DRAFT:-1}"
DOWNLOAD_PD_SMOKE_MODEL="${DOWNLOAD_PD_SMOKE_MODEL:-0}"

python -m pip install modelscope huggingface_hub -i "${PIP_INDEX}"

if [[ "${DOWNLOAD_TARGET}" == "1" ]]; then
  if [[ "${TARGET_SOURCE}" == "modelscope" ]]; then
    modelscope download --model Qwen/Qwen3.8-27B-FP8 \
      --local_dir "${TARGET_MODEL}"
  elif [[ "${TARGET_SOURCE}" == "huggingface" ]]; then
    hf download Qwen/Qwen3.8-27B-FP8 --local-dir "${TARGET_MODEL}"
  else
    echo "TARGET_SOURCE 只能是 modelscope 或 huggingface" >&2
    exit 2
  fi
fi

if [[ "${DOWNLOAD_DRAFT}" == "1" ]]; then
  hf download RadixArk/Qwen3.8-27B-DSpark \
    --local-dir "${DRAFT_MODEL}"
fi

if [[ "${DOWNLOAD_PD_SMOKE_MODEL}" == "1" ]]; then
  hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --local-dir "${PD_SMOKE_MODEL}"
fi

for model_path in "${TARGET_MODEL}" "${DRAFT_MODEL}" "${PD_SMOKE_MODEL}"; do
  if [[ -e "${model_path}" ]]; then
    du -sh "${model_path}"
  fi
done
