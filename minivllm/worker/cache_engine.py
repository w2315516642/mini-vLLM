from typing import Dict, Tuple, List, Optional
import torch

from minivllm import cache_ops
from minivllm.configs import CacheConfig, ModelConfig, ParallelConfig
from minivllm.configs.model_architecture import FULL_ATTENTION
from minivllm.worker.hybrid_cache import GatedDeltaNetStateSpec

KVCache = Tuple[torch.Tensor, torch.Tensor]
LayerKVCache = Optional[KVCache]

class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        self.head_size = model_config.get_head_size() 
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.layer_types = model_config.architecture.layer_types
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        self.dtype = model_config.dtype

        self.block_size = cache_config.block_size   
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks

        self.gpu_cache = self.allocate_gpu_cache()
        self.cpu_cache = self.allocate_cpu_cache()

        # Initialize the stream for caching operations.
        self.cache_stream = torch.cuda.Stream()
        assert self.cache_stream != torch.cuda.current_stream()
        # Initialize the events for stream synchronization.
        self.events = [
            torch.cuda.Event() if layer_type == FULL_ATTENTION else None
            for layer_type in self.layer_types
        ]

    def get_key_block_shape(self) -> Tuple[int, int, int, int]:
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        # 要凑齐 16 字节，需要连续读取多少个元素
        x = 16 // element_size
        return (
            self.num_kv_heads,
            self.head_size // x,
            self.block_size,
            x,
        )
    
    def get_value_block_shape(self) -> Tuple[int, int, int]:
        return (
            self.num_kv_heads,
            self.head_size,
            self.block_size,
        )

    def allocate_gpu_cache(self) -> List[LayerKVCache]:
        gpu_cache: List[LayerKVCache] = []
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        for layer_type in self.layer_types:
            if layer_type != FULL_ATTENTION:
                gpu_cache.append(None)
                continue
            key_block = torch.empty(
                # 申请 num_gpu_blocks 个 block
                # NOTE: 这里假设每个block形状固定
                size=(self.num_gpu_blocks, *key_block_shape),
                dtype=self.dtype,
                device="cuda"
            )
            value_block = torch.empty(
                size=(self.num_gpu_blocks, *value_block_shape),
                dtype=self.dtype,
                device="cuda"
            )
            gpu_cache.append((key_block, value_block))
        return gpu_cache

    def allocate_cpu_cache(self) -> List[LayerKVCache]:
        cpu_cache: List[LayerKVCache] = []
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        for layer_type in self.layer_types:
            if layer_type != FULL_ATTENTION:
                cpu_cache.append(None)
                continue
            key_block = torch.empty(
                # CPU swap space has its own independently configured capacity.
                # NOTE: 这里假设每个block形状固定
                size=(self.num_cpu_blocks, *key_block_shape),
                dtype=self.dtype,
                pin_memory=True
            )
            value_block = torch.empty(
                size=(self.num_cpu_blocks, *value_block_shape),
                dtype=self.dtype,
                pin_memory=True
            )
            cpu_cache.append((key_block, value_block))
        return cpu_cache

    def _swap(
        self,
        src: List[LayerKVCache],
        dst: List[LayerKVCache],
        src_to_dst: Dict[int, int],
    ) -> None:
        with torch.cuda.stream(self.cache_stream):
            for i in range(self.num_layers):
                if src[i] is None:
                    continue
                src_key_cache, src_value_cache = src[i]
                dst_key_cache, dst_value_cache = dst[i]
                # Copy the key blocks
                cache_ops.swap_blocks(
                    src_key_cache, dst_key_cache, src_to_dst
                )
                cache_ops.swap_blocks(
                    src_value_cache, dst_value_cache, src_to_dst
                )
                self.events[i].record(stream=self.cache_stream)

    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.cpu_cache, self.gpu_cache, src_to_dst)

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.gpu_cache, self.cpu_cache, src_to_dst)

    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        full_caches = [cache for cache in self.gpu_cache if cache is not None]
        key_caches = [key_cache for key_cache, _ in full_caches]
        value_caches = [value_cache for _, value_cache in full_caches]
        # NOTE(by original author woosuk): 
        # This operation implicitly synchronizes the CPU and GPU.
        cache_ops.copy_blocks(key_caches, value_caches, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        block_size: int,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size() 
        architecture = getattr(model_config, "architecture", None)
        num_layers = (
            architecture.num_full_attention_layers
            if architecture is not None
            else model_config.get_num_layers(parallel_config)
        )
        num_kv_heads = model_config.get_num_kv_heads(parallel_config)

        # Size of key cache in bytes.
        key_cache_block = block_size * num_kv_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        dtype_size = _get_dtype_size(model_config.dtype)
        return dtype_size * total

    @staticmethod
    def get_state_cache_size(
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        max_num_seqs: int,
    ) -> int:
        """Return persistent FP32 GDN state bytes allocated by one worker."""
        num_linear_layers = (
            model_config.architecture.num_linear_attention_layers
        )
        if num_linear_layers == 0:
            return 0
        spec = GatedDeltaNetStateSpec.from_text_config(
            model_config.architecture.text_config
        )
        conv_elements = max_num_seqs * torch.tensor(
            spec.conv_state_shape
        ).prod().item()
        recurrent_elements = max_num_seqs * torch.tensor(
            spec.recurrent_state_shape
        ).prod().item()
        return int(
            num_linear_layers
            * (conv_elements + recurrent_elements)
            * torch.tensor([], dtype=torch.float32).element_size()
        )


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
