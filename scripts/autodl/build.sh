#!/usr/bin/env bash

set -euo pipefail
source "$(dirname -- "$0")/common.sh"

activate_minivllm_env
configure_cuda
enter_repo

PIP_INDEX="${MINIVLLM_PIP_INDEX:-https://pypi.org/simple}"
MAX_JOBS="${MAX_JOBS:-8}"
INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS:-1}"

python - <<'PY'
import torch
from torch.utils.cpp_extension import CUDA_HOME

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("detected CUDA_HOME:", CUDA_HOME)
if not torch.cuda.is_available():
    raise SystemExit("构建前必须让 PyTorch 能访问 GPU")
PY
nvcc -V

python -m pip install ninja packaging setuptools wheel -i "${PIP_INDEX}"
if [[ "${INSTALL_REQUIREMENTS}" == "1" ]]; then
  python -m pip install -r requirements.txt -i "${PIP_INDEX}"
fi

MAX_JOBS="${MAX_JOBS}" python -m pip install \
  -v -e . --no-build-isolation -i "${PIP_INDEX}"

python - <<'PY'
from minivllm import attention_ops, cache_ops, dspark_ops, gated_delta_net_ops

print("CUDA extensions imported successfully")
PY
