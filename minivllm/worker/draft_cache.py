"""KV cache owned by the DSpark drafter."""

from typing import Dict, List, Tuple

import torch

from minivllm import cache_ops


DraftKVCache = Tuple[torch.Tensor, torch.Tensor]


class DraftCacheEngine:
    """Allocate draft-layer K/V using target-model physical block ids.

    The scheduler remains the single owner of block allocation. Draft cache
    tensors mirror its physical block count, so swap/copy maps can be applied
    to target and draft caches together.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str = "cuda",
    ) -> None:
        values = {
            "num_layers": num_layers,
            "num_gpu_blocks": num_gpu_blocks,
            "block_size": block_size,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
        }
        invalid = [name for name, value in values.items() if value <= 0]
        if invalid or num_cpu_blocks < 0:
            raise ValueError("Invalid draft cache dimensions: " + ", ".join(invalid))
        self.num_layers = int(num_layers)
        self.num_gpu_blocks = int(num_gpu_blocks)
        self.num_cpu_blocks = int(num_cpu_blocks)
        self.block_size = int(block_size)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.dtype = dtype
        self.device = torch.device(device)
        self.gpu_cache = self._allocate(self.num_gpu_blocks, self.device)
        self.cpu_cache = self._allocate(
            self.num_cpu_blocks,
            torch.device("cpu"),
            pin_memory=self.device.type == "cuda",
        )
        self.cache_stream = (
            torch.cuda.Stream() if self.device.type == "cuda" else None
        )

    def _allocate(
        self,
        num_blocks: int,
        device: torch.device,
        *,
        pin_memory: bool = False,
    ) -> List[DraftKVCache]:
        x = 16 // torch.tensor([], dtype=self.dtype).element_size()
        caches = []
        for _ in range(self.num_layers):
            key = torch.empty(
                num_blocks,
                self.num_kv_heads,
                self.head_dim // x,
                self.block_size,
                x,
                dtype=self.dtype,
                device=device,
                pin_memory=pin_memory,
            )
            value = torch.empty(
                num_blocks,
                self.num_kv_heads,
                self.head_dim,
                self.block_size,
                dtype=self.dtype,
                device=device,
                pin_memory=pin_memory,
            )
            caches.append((key, value))
        return caches

    @property
    def bytes_per_block(self) -> int:
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        return (
            self.num_layers
            * 2
            * self.num_kv_heads
            * self.head_dim
            * self.block_size
            * element_size
        )

    def _swap(
        self,
        source: List[DraftKVCache],
        destination: List[DraftKVCache],
        mapping: Dict[int, int],
    ) -> None:
        if not mapping:
            return
        if self.cache_stream is None:
            raise RuntimeError("Draft cache swap requires CUDA")
        with torch.cuda.stream(self.cache_stream):
            for source_cache, destination_cache in zip(source, destination):
                for source_tensor, destination_tensor in zip(
                    source_cache, destination_cache
                ):
                    cache_ops.swap_blocks(source_tensor, destination_tensor, mapping)

    def swap_in(self, mapping: Dict[int, int]) -> None:
        self._swap(self.cpu_cache, self.gpu_cache, mapping)

    def swap_out(self, mapping: Dict[int, int]) -> None:
        self._swap(self.gpu_cache, self.cpu_cache, mapping)

    def copy(self, mapping: Dict[int, List[int]]) -> None:
        if not mapping:
            return
        cache_ops.copy_blocks(
            [cache[0] for cache in self.gpu_cache],
            [cache[1] for cache in self.gpu_cache],
            mapping,
        )
