import filelock
import os
import glob
import json
from typing import Iterable, List, Optional, Tuple

from huggingface_hub import snapshot_download
import numpy as np
from safetensors import safe_open
import torch
from tqdm.auto import tqdm


class Disabledtqdm(tqdm):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, disable=True)


def hf_model_weights_iterator(
    model_name_or_path: str,
    cache_dir: Optional[str] = None,
    use_np_cache: bool = False,
) -> Iterable[Tuple[str, torch.Tensor]]:
    # Prepare file lock directory to prevent multiple processes from
    # downloading the same model weights at the same time.
    lock_dir = cache_dir if cache_dir is not None else "/tmp"
    os.makedirs(lock_dir, exist_ok=True)
    lock_file_name = model_name_or_path.replace("/", "-") + ".lock"
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name))

    is_local = os.path.isdir(model_name_or_path)
    if not is_local:
        with lock:
            hf_loader = snapshot_download(
                model_name_or_path,
                allow_patterns=["*.safetensors", "*.json", "*.bin"],
                cache_dir=cache_dir,
                tqdm_class=Disabledtqdm)
    else:
        hf_loader = model_name_or_path
    
    hf_safetensor_files = sorted(
        glob.glob(os.path.join(hf_loader, "*.safetensors"))
    )
    hf_bin_files = sorted(glob.glob(os.path.join(hf_loader, "*.bin")))

    # Prefer safetensors when a repository publishes both formats. Opening one
    # shard at a time avoids materializing a second copy of a large checkpoint.
    if hf_safetensor_files:
        if use_np_cache:
            raise ValueError(
                "NumPy weight caching is not supported for safetensors "
                "checkpoints"
            )
        for safetensor_file in hf_safetensor_files:
            with safe_open(safetensor_file, framework="pt", device="cpu") as f:
                for name in f.keys():
                    yield name, f.get_tensor(name)
        return

    if not hf_bin_files:
        raise RuntimeError(
            f"No .safetensors or .bin weights found in {hf_loader}"
        )

    if use_np_cache:
        # Convert the model weights from torch tensors to numpy arrays for
        # faster loading.
        np_folder = os.path.join(hf_loader, "np")
        os.makedirs(np_folder, exist_ok=True)
        weight_names_file = os.path.join(np_folder, "weight_names.json")
        with lock:
            if not os.path.exists(weight_names_file):
                weight_names = []
                for bin_file in hf_bin_files:
                    state = torch.load(bin_file, map_location="cpu")
                    for name, param in state.items():
                        param_path = os.path.join(np_folder, name)
                        with open(param_path, 'wb') as f:
                            np.save(f, param.cpu().detach().numpy())
                        weight_names.append(name)
                with open(weight_names_file, 'w') as f:
                    json.dump(weight_names, f)
        
        with open(weight_names_file, 'r') as f:
            weight_names = json.load(f)

        for name in weight_names:
            param_path = os.path.join(np_folder, name)
            with open(param_path, 'rb') as f:
                param = np.load(f)
            yield name, torch.from_numpy(param)
    else:
        for bin_file in hf_bin_files:
            state = torch.load(bin_file, map_location="cpu")
            for name, param in state.items():
                yield name, param


def load_tensor_parallel_weights(
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    param_name: str,
    column_parallel_weight_name: List[str],
    row_parallel_weight_name: List[str],
    tensor_model_parallel_rank: int,
) -> None:
    for p in column_parallel_weight_name:
        if p in param_name:
            shard_size = param.shape[0]
            loaded_weight = loaded_weight[
                shard_size * tensor_model_parallel_rank:
                shard_size * (tensor_model_parallel_rank + 1)
            ]
            break
    for p in row_parallel_weight_name:
        if p in param_name:
            shard_size = param.shape[1]
            loaded_weight = loaded_weight[
                :,
                shard_size * tensor_model_parallel_rank:
                shard_size * (tensor_model_parallel_rank + 1)
            ]
            break
    assert param.shape == loaded_weight.shape
    param.data.copy_(loaded_weight)


def initialize_dummy_weights(
    model: torch.nn.Module,
    low: float = -1e-3,
    high: float = 1e-3
) -> None:
    """Initialize model weights with random values.

    The model weights must be randomly initialized for accurate performance
    measurements. Additionally, the model weights should not cause NaNs in the
    forward pass. We empirically found that initializing the weights with
    values between -1e-3 and 1e-3 works well for most models. (by original author)
    """
    float8_dtypes = {
        dtype
        for dtype in (
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e5m2", None),
        )
        if dtype is not None
    }
    for name, param in model.state_dict().items():
        if name.endswith("weight_scale_inv"):
            param.data.fill_(1.0)
        elif param.dtype in float8_dtypes:
            initialized = torch.empty_like(param, dtype=torch.float32)
            initialized.uniform_(low, high)
            param.data.copy_(initialized.to(param.dtype))
        elif param.is_floating_point():
            param.data.uniform_(low, high)
