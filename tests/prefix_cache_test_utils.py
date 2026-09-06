import importlib.util
import sys
import types
from enum import Enum, auto
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@patch.dict(sys.modules)
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


@patch.dict(sys.modules)
def load_worker_module():
    """Load Worker with small stubs for dependencies unrelated to input prep."""
    configs_config = types.ModuleType("minivllm.configs.config")
    for name in ("ModelConfig", "CacheConfig", "ParallelConfig", "SchedulerConfig"):
        setattr(configs_config, name, object)
    sys.modules["minivllm.configs.config"] = configs_config

    xformers = types.ModuleType("xformers")
    xformers_ops = types.ModuleType("xformers.ops")
    xformers_fmha = types.ModuleType("xformers.ops.fmha")
    xformers_bias = types.ModuleType("xformers.ops.fmha.attn_bias")

    class BlockDiagonalCausalMask:
        @classmethod
        def from_seqlens(cls, seqlens):
            return tuple(seqlens)

    xformers_bias.BlockDiagonalCausalMask = BlockDiagonalCausalMask
    sys.modules["xformers"] = xformers
    sys.modules["xformers.ops"] = xformers_ops
    sys.modules["xformers.ops.fmha"] = xformers_fmha
    sys.modules["xformers.ops.fmha.attn_bias"] = xformers_bias

    input_metadata = _load_module(
        "minivllm.model_executor.input_metadata",
        "minivllm/model_executor/input_metadata.py",
    )
    model_executor = types.ModuleType("minivllm.model_executor")
    model_executor.__path__ = [str(ROOT / "minivllm" / "model_executor")]
    model_executor.InputMetadata = input_metadata.InputMetadata
    model_executor.set_random_seed = lambda *args, **kwargs: None
    model_executor.get_model = lambda *args, **kwargs: None
    sys.modules["minivllm.model_executor"] = model_executor

    parallel_state = types.ModuleType(
        "minivllm.model_executor.parallel_utils.parallel_state")
    parallel_state.initialize_all_reduce_launcher = lambda *args, **kwargs: None
    parallel_state.initialize_model_parallel = lambda *args, **kwargs: None
    parallel_state.get_tensor_model_parallel_rank = lambda: 0
    parallel_state.get_tensor_model_parallel_world_size = lambda: 1
    parallel_state.get_all_reduce_launcher = lambda: None
    sys.modules[
        "minivllm.model_executor.parallel_utils.parallel_state"
    ] = parallel_state

    worker_package = types.ModuleType("minivllm.worker")
    worker_package.__path__ = [str(ROOT / "minivllm" / "worker")]
    sys.modules["minivllm.worker"] = worker_package
    cache_engine = types.ModuleType("minivllm.worker.cache_engine")
    cache_engine.CacheEngine = object
    sys.modules["minivllm.worker.cache_engine"] = cache_engine

    device = types.ModuleType("minivllm.utils.device")
    device.get_gpu_memory = lambda *args, **kwargs: 0
    sys.modules["minivllm.utils.device"] = device

    return _load_module("minivllm.worker.worker", "minivllm/worker/worker.py")
