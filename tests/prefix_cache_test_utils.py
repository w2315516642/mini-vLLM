import importlib.util
import sys
import types
from enum import Enum, auto
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_prefix_cache_modules():
    """Load scheduler/cache code without importing CUDA and Ray modules."""
    minivllm = types.ModuleType("minivllm")
    minivllm.__path__ = [str(ROOT / "minivllm")]
    sys.modules["minivllm"] = minivllm

    utils = types.ModuleType("minivllm.utils")
    utils.__path__ = [str(ROOT / "minivllm" / "utils")]
    sys.modules["minivllm.utils"] = utils

    kv_cache = types.ModuleType("minivllm.kv_cache")
    kv_cache.__path__ = [str(ROOT / "minivllm" / "kv_cache")]
    sys.modules["minivllm.kv_cache"] = kv_cache

    hasher = _load_module(
        "minivllm.utils.hasher", "minivllm/utils/hasher.py")

    class Device(Enum):
        CPU = auto()
        GPU = auto()

    utils.Device = Device
    utils.BlockHash = hasher.BlockHash
    utils.BlockHasher = hasher.BlockHasher

    sampling_params = types.ModuleType("minivllm.sampling_params")
    sampling_params.SamplingParams = object
    sys.modules["minivllm.sampling_params"] = sampling_params

    configs = types.ModuleType("minivllm.configs")
    configs.CacheConfig = object
    configs.SchedulerConfig = object
    sys.modules["minivllm.configs"] = configs

    loguru = types.ModuleType("loguru")
    loguru.logger = types.SimpleNamespace(warning=lambda *args, **kwargs: None)
    sys.modules["loguru"] = loguru

    _load_module("minivllm.kv_cache.block", "minivllm/kv_cache/block.py")
    sequence = _load_module("minivllm.sequence", "minivllm/sequence.py")
    block_manager = _load_module(
        "minivllm.kv_cache.block_manager",
        "minivllm/kv_cache/block_manager.py",
    )
    _load_module("minivllm.kv_cache.policy", "minivllm/kv_cache/policy.py")
    scheduler = _load_module(
        "minivllm.kv_cache.scheduler", "minivllm/kv_cache/scheduler.py")
    return hasher, sequence, block_manager, scheduler


hasher, sequence, block_manager, scheduler = _load_prefix_cache_modules()
hasher.init_none_hash(hasher.sha256)


def make_seq(seq_id, token_ids, block_size=4):
    return sequence.Sequence(
        seq_id=seq_id,
        prompt="",
        prompt_token_ids=token_ids,
        block_size=block_size,
        block_hasher=hasher.get_seq_block_hasher(hasher.sha256),
    )


def make_group(request_id, seqs):
    return sequence.SequenceGroup(request_id, seqs, None, 0.0)


def make_scheduler(max_tokens=32, max_seqs=8, num_gpu_blocks=16):
    scheduler_config = types.SimpleNamespace(
        max_num_batched_tokens=max_tokens,
        max_num_seqs=max_seqs,
    )
    cache_config = types.SimpleNamespace(
        block_size=4,
        num_gpu_blocks=num_gpu_blocks,
        num_cpu_blocks=4,
        enable_prefix_caching=True,
    )
    return scheduler.Scheduler(scheduler_config, cache_config, log_stats=False)
