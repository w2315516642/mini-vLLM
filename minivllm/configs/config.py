import torch
from typing import Optional
from loguru import logger
from transformers import AutoConfig, PretrainedConfig

from minivllm.utils.device import get_cpu_memory

_GiB = 1 << 30

class ModelConfig:
    def __init__(
        self,
        model: str,
        download_dir: Optional[str],
        use_np_weights: bool,
        use_dummy_weights: bool,
        dtype: str,
        seed: int,
    ) -> None:
        self.model = model
        self.download_dir = download_dir
        self.use_np_weights = use_np_weights
        self.use_dummy_weights = use_dummy_weights
        self.seed = seed

        self.hf_config: PretrainedConfig = AutoConfig.from_pretrained(model)
        self.dtype = _get_and_verify_dtype(self.hf_config, dtype)

    def verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:
        "tensor并行切的是注意力头吗"
        total_num_attention_heads = self.hf_config.num_attention_heads
        tensor_parallel_size = parallel_config.tensor_parallel_size
        if total_num_attention_heads % tensor_parallel_size != 0:
            raise ValueError(
                f"Total number of attention heads ({total_num_attention_heads})"
                " must be divisible by tensor parallel size "
                f"({tensor_parallel_size}).")

        "流水线并行切的是层数，n级就要分成n的倍数"
        total_num_hidden_layers = self.hf_config.num_hidden_layers
        pipeline_parallel_size = parallel_config.pipeline_parallel_size
        if total_num_hidden_layers % pipeline_parallel_size != 0:
            raise ValueError(
                f"Total number of hidden layers ({total_num_hidden_layers}) "
                "must be divisible by pipeline parallel size "
                f"({pipeline_parallel_size}).")

    def get_hidden_size(self) -> int:
        return self.hf_config.hidden_size
    
    def get_head_size(self) -> int:
        # FIXME: 对 GQA、MQA 这种是不成立的
        return self.hf_config.hidden_size // self.hf_config.num_attention_heads

    def get_num_heads(self, parallel_config: "ParallelConfig") -> int:
        total_num_attention_heads = self.hf_config.num_attention_heads
        return total_num_attention_heads // parallel_config.tensor_parallel_size

    def get_num_layers(self, parallel_config: "ParallelConfig") -> int:
        total_num_hidden_layers = self.hf_config.num_hidden_layers
        return total_num_hidden_layers // parallel_config.pipeline_parallel_size


class CacheConfig:
    " 对 KVCache 的配置 "
    def __init__(
        self,
        block_size: int, 
        gpu_memory_utilization: float,
        swap_space: int,
        prefix_caching_hash_fn: str = "sha256",
    ) -> None:
        self.block_size = block_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.swap_space_bytes = swap_space * _GiB
        self._verify_args()

        self.num_gpu_blocks = None
        self.num_cpu_blocks = None

        self.prefix_caching_hash_fn = prefix_caching_hash_fn

    def _verify_args(self) -> None:
        if self.gpu_memory_utilization > 1.0:
            raise ValueError(
                "GPU memory utilization muse be less than 1.0, got "
                f"{self.gpu_memory_utilization}"
            )

    def verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:
        total_cpu_memory = get_cpu_memory()
        """
            这里是假设所有GPU都在同一个节点上，但实际上GPU可能分布在不同节点（实例）
            并且这里默认用的是TP，没有考虑PP需要的GPU数量（因为还没实现PP）
        """
        num_gpus_per_node = parallel_config.tensor_parallel_size
        cpu_memory_usage = self.swap_space_bytes * num_gpus_per_node

        msg = (
            f"{cpu_memory_usage / _GiB} GiB out of "
            f"the {total_cpu_memory / _GiB} GiB total CPU memory is "
            "allocated for the swap space"
        )
        if cpu_memory_usage > 0.7 * total_cpu_memory:
            raise ValueError("Too large swap space. " + msg)
        if cpu_memory_usage > 0.4 * total_cpu_memory:
            logger.warning("Possibly too large swap space. " + msg)


class ParallelConfig:
    """ 分布式推理的设置 """
    def __init__(
        self,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        worker_use_ray: bool,
    ) -> None:
        self.pipeline_parallel_size = pipeline_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.worker_use_ray = worker_use_ray

        self.world_size = pipeline_parallel_size * tensor_parallel_size
        if self.world_size > 1:
            self.worker_use_ray = True
        self._verify_args()

    def _verify_args(self) -> None:
        if self.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "Pipeline parallelism is not supported yet."
            )


class SchedulerConfig:
    """ 调度器配置

        Args:
            max_num_batched_tokens: 所有处理中序列拼接到一起后的最大序列长度
            max_num_seqs: 可同时处理的最多序列数量
    """
    def __init__(
        self,
        max_num_batched_tokens: int,
        max_num_seqs: int,
    ) -> None:
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs


_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.float16,
    "float16": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def _get_and_verify_dtype(
    config: PretrainedConfig, 
    dtype: str
) -> torch.dtype:
    "'torch_dtype'在新版本的transformers里面被弃用了"
    config_dtype = getattr(config, "dtype", None)
    if config_dtype is None:
        config_dtype = torch.float32
    
    dtype = dtype.lower()
    if dtype == "auto":
        if config_dtype == torch.float32:
            torch_dtype = torch.float16
        else:
            torch_dtype = config_dtype
    else:
        if dtype not in _STR_DTYPE_TO_TORCH_DTYPE:
            raise ValueError(f"Unknown dtype: {dtype}")
        torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE[dtype]

    # 实际用的dtype和设置的dtype不符合
    if torch_dtype != config_dtype:
        if torch_dtype == torch.float32:
            # 精度上升，f16 -> f32
            logger.info(f"Config dtype {config_dtype} not equal to torch dtype {torch_dtype}")
        elif config_dtype == torch.float32:
            # 精度下降，f32 -> f16
            logger.info(f"Config dtype {config_dtype} not equal to torch dtype {torch_dtype}")
        else:
            # bf16 和 fp16 之间转换
            logger.warning(f"Casting {config_dtype} to {torch_dtype}")

    if torch_dtype == torch.bfloat16:
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] < 8:
            gpu_name = torch.cuda.get_device_name()
            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU has compute capability "
                f"{compute_capability[0]}.{compute_capability[1]}.")
    return torch_dtype