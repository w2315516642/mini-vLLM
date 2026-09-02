#!/usr/bin/env bash

set -euo pipefail
source "$(dirname -- "$0")/common.sh"

activate_minivllm_env
configure_cuda
enter_repo

echo "== Git =="
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git status --short --branch
  git rev-parse --short HEAD
else
  echo "当前目录不是此 shell 可识别的 Git worktree：${REPO_ROOT}"
fi

echo "== CUDA =="
echo "CUDA_HOME=${CUDA_HOME}"
require_command nvcc
nvcc -V
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader

echo "== Python =="
python - <<'PY'
import importlib
import torch
import transformers
import xformers

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("xformers:", xformers.__version__)
if not torch.cuda.is_available():
    raise SystemExit("PyTorch 当前看不到 CUDA")

for name in (
    "minivllm.cache_ops",
    "minivllm.attention_ops",
    "minivllm.gated_delta_net_ops",
    "minivllm.dspark_ops",
):
    try:
        importlib.import_module(name)
    except ImportError as exc:
        print(f"extension: {name}: 未构建 ({exc})")
    else:
        print(f"extension: {name}: OK")
PY

echo "== Disk =="
df -h "${MODEL_ROOT}" 2>/dev/null || df -h "$(dirname -- "${MODEL_ROOT}")"
