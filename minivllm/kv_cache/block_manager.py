from typing import Optional, List, Dict, Tuple, Set
from collections import OrderedDict

from minivllm.kv_cache.block import PhysicalTokenBlock
from minivllm.sequence import Sequence, SequenceGroup, SequenceStatus
from minivllm.utils import Device, BlockHash

class BlockAllocator:
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    def __init__(
        self,
        device: Device,
        block_size: int,
        num_blocks: int,
    ) -> None:
        self.device = device
        self.block_size = block_size
        self.num_blocks = num_blocks

        self.free_blocks: List[PhysicalTokenBlock] = []
        for i in range(num_blocks):
            block = PhysicalTokenBlock(
                device, block_id=i, block_size=block_size
            )
            self.free_blocks.append(block)
        
    def allocate(self) -> PhysicalTokenBlock:
        if not self.free_blocks:
            raise ValueError("Out of memory! No free blocks are available.")
        block = self.free_blocks.pop()
        block.ref_count = 1
        return block
    
    def free(self, block: PhysicalTokenBlock) -> None:
        if block.ref_count == 0:
            raise ValueError(f"Double free! {block} is already freed.")
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block)
    
    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)


class BlockHashManager:
    """Maps prefix hashes to GPU blocks and maintains LRU order.

    Every entry owns one reference to its block. A block with ref_count == 1
    is therefore only kept alive by the prefix cache and can be evicted.
    """

    def __init__(self) -> None:
        self.hash_to_block: OrderedDict[
            BlockHash, PhysicalTokenBlock
        ] = OrderedDict()

    def add_blocks(
        self,
        block_hashes: List[BlockHash],
        blocks: List[PhysicalTokenBlock],
    ) -> None:
        assert len(block_hashes) == len(blocks), (
            f"Length of block_hashes should be equal to blocks, "
            f"but got {len(block_hashes)} and {len(blocks)}"
        )

        for block_hash, block in zip(block_hashes, blocks):
            if block_hash in self.hash_to_block:
                self.hash_to_block.move_to_end(block_hash)
                continue
            assert block.device == Device.GPU
            # This reference belongs to the cache, not to a sequence.
            block.ref_count += 1
            self.hash_to_block[block_hash] = block

    def get_block(
        self, 
        block_hash: BlockHash
    ) -> PhysicalTokenBlock | None:
        if block_hash not in self.hash_to_block:
            return None
        
        self.hash_to_block.move_to_end(block_hash)
        return self.hash_to_block[block_hash]

    def find_longest_cache_hit(
        self,
        block_hashes: List[BlockHash],
        max_num_tokens: int,
    ) -> List[PhysicalTokenBlock]:
        hit_blocks: List[PhysicalTokenBlock] = []
        curr_num_tokens = 0
        for block_hash in block_hashes:
            block = self.get_block(block_hash)
            if block is None:
                break
            
            curr_num_tokens += block.block_size
            if curr_num_tokens > max_num_tokens:
                break

            hit_blocks.append(block)
        return hit_blocks

    def evict(self, block_hash: BlockHash) -> PhysicalTokenBlock | None:
        evicted_block = self.hash_to_block.pop(block_hash, None)
        return evicted_block

    def evict_lru(self) -> PhysicalTokenBlock | None:
        """Remove the oldest block that is only owned by the cache."""
        for block_hash, block in self.hash_to_block.items():
            if block.ref_count == 1:
                del self.hash_to_block[block_hash]
                return block
        return None

    def get_num_evictable_blocks(self) -> int:
        return sum(
            block.ref_count == 1 for block in self.hash_to_block.values()
        )

    def clear(self) -> List[PhysicalTokenBlock]:
        blocks = list(self.hash_to_block.values())
        self.hash_to_block.clear()
        return blocks

    def __len__(self) -> int:
        return len(self.hash_to_block)


BlockTable = List[PhysicalTokenBlock]

class BlockSpaceManager:
    """Manages the mapping between logical and physical token blocks."""

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
        enable_prefix_caching: bool = True,
    ) -> None:
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        self.num_cpu_blocks = num_cpu_blocks
        self.watermark = watermark
        self.enable_prefix_caching = enable_prefix_caching
        assert watermark >= 0.0

        self.watermark_blocks = int(watermark * num_gpu_blocks)
        self.gpu_allocator = BlockAllocator(
            Device.GPU, block_size, num_gpu_blocks)
        self.cpu_allocator = BlockAllocator(
            Device.CPU, block_size, num_cpu_blocks)
        self.hash_manager = BlockHashManager()

        # Mapping: seq_id -> BlockTable.
        self.block_tables: Dict[int, BlockTable] = {}

    def _get_num_available_gpu_blocks(
        self,
        protected_blocks: Optional[BlockTable] = None,
    ) -> int:
        """Count free and evictable blocks, excluding blocks about to be used."""
        num_available_blocks = (
            self.gpu_allocator.get_num_free_blocks()
            + self.hash_manager.get_num_evictable_blocks()
        )
        if protected_blocks:
            # Cache-only hits are normally evictable, but allocation will add
            # sequence references to them. They cannot fund new block slots.
            num_available_blocks -= sum(
                block.ref_count == 1 for block in protected_blocks
            )
        return num_available_blocks

    def _allocate_gpu_block(self) -> PhysicalTokenBlock:
        """Allocate a block, evicting one cache-only LRU block if needed."""
        if self.gpu_allocator.get_num_free_blocks() == 0:
            evicted_block = self.hash_manager.evict_lru()
            if evicted_block is None:
                raise ValueError("Out of memory! No evictable GPU blocks.")
            # Drop the reference owned by the prefix cache.
            self.gpu_allocator.free(evicted_block)
        return self.gpu_allocator.allocate()

    def cache_blocks(self, seq: Sequence) -> None:
        if not self.enable_prefix_caching:
            return
        block_table = self.block_tables[seq.seq_id]
        # Logical blocks exist before model execution. Only the blocks covered
        # by num_computed_tokens are guaranteed to contain valid KV data.
        num_cacheable_blocks = min(
            seq.num_computed_tokens // self.block_size,
            len(seq.block_hashes),
            len(block_table),
        )
        if num_cacheable_blocks <= seq.num_cached_blocks:
            return

        start = seq.num_cached_blocks
        self.hash_manager.add_blocks(
            seq.block_hashes[start:num_cacheable_blocks],
            block_table[start:num_cacheable_blocks],
        )
        seq.num_cached_blocks = num_cacheable_blocks
    
    def can_allocate(self, seq_group: SequenceGroup) -> bool:
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.
        seq = seq_group.get_seqs()[0]
        hit_blocks = self.find_longest_cache_hit(seq)
        num_hit_blocks = len(hit_blocks)
        num_required_blocks = len(seq.logical_token_blocks) - num_hit_blocks
        num_available_blocks = self._get_num_available_gpu_blocks(hit_blocks)
        # Use watermark to avoid frequent cache eviction.
        return (num_available_blocks - num_required_blocks
                >= self.watermark_blocks)

    def get_num_cached_tokens(self, seq_group: SequenceGroup) -> int:
        """Return the reusable prefix length without allocating block tables."""
        seq = seq_group.get_seqs()[0]
        return len(self.find_longest_cache_hit(seq)) * self.block_size

    def find_longest_cache_hit(self, seq: Sequence) -> BlockTable:
        if not self.enable_prefix_caching:
            return []
        # Allocate new physical token blocks that will store the prompt tokens.
        # 这里找prompt命中的缓存块，找不到的再从gpu_allocator要
        # Keep the final known token for logits. With full-block reuse, an
        # aligned prompt therefore recomputes its final block.
        max_num_tokens = max(seq.get_len() - 1, 0)
        hit_blocks = self.hash_manager.find_longest_cache_hit(
            seq.block_hashes, max_num_tokens)
        return list(hit_blocks)

    def allocate(self, seq_group: SequenceGroup) -> None:
        # NOTE: Here we assume that all sequences in the group have the same prompt.
        seq = seq_group.get_seqs()[0]
        block_table = self.find_longest_cache_hit(seq)

        num_computed_tokens = len(block_table) * self.block_size
        num_need_blocks = len(seq.logical_token_blocks) - len(block_table)
        assert num_need_blocks >= 0, (
            f"Number of logical_token_blocks {len(seq.logical_token_blocks)} "
            f"should be greater than or equal to hit_blocks {len(block_table)}."
        )
        hit_blocks = block_table.copy()
        num_seqs = seq_group.num_seqs()

        if num_need_blocks > self._get_num_available_gpu_blocks(hit_blocks):
            raise ValueError(
                "Out of memory! Prefix hits leave too few blocks for the "
                "uncached suffix."
            )

        # Pin hits before allocating suffix blocks. Otherwise an empty free
        # list could evict and immediately reuse a block selected for this
        # sequence's prefix.
        for block in hit_blocks:
            block.ref_count += num_seqs

        new_blocks: BlockTable = []
        for _ in range(num_need_blocks):
            block = self._allocate_gpu_block()
            block_table.append(block)
            new_blocks.append(block)

        # New blocks get their first sequence reference from allocate().
        for block in new_blocks:
            block.ref_count += num_seqs - 1
        
        # Assign the block table for each sequence.
        # 一个seq group里面所有seq共享相同的初始block table（因为prompt相同）
        for seq in seq_group.get_seqs():
            seq.num_computed_tokens = num_computed_tokens
            seq.num_cached_blocks = len(hit_blocks)
            self.block_tables[seq.seq_id] = block_table.copy()
    
    def can_append_slot(self, seq_group: SequenceGroup) -> bool:
        # Simple heuristic: If there is at least one free block
        # for each sequence, we can append.
        num_available_blocks = self._get_num_available_gpu_blocks()
        num_seqs = seq_group.num_seqs(status=SequenceStatus.RUNNING)
        return num_seqs <= num_available_blocks

    def append_slot(self, seq: Sequence) -> Optional[Tuple[int, int]]:
        """Allocate a physical slot for a new token."""
        logical_blocks = seq.logical_token_blocks
        block_table = self.block_tables[seq.seq_id]

        if len(block_table) < len(logical_blocks):
            # The sequence has a new logical block.
            # Allocate a new physical block.
            block = self._allocate_gpu_block()
            block_table.append(block)
            return None
        
        # We want to append the token to the last physical block.
        last_block = block_table[-1]
        assert last_block.device == Device.GPU
        if last_block.ref_count == 1:
            # Not shared with other sequences. Appendable.
            return None
        else:
            # The last block is shared with other sequences.
            # Copy on Write: Allocate a new block and copy the tokens.
            new_block = self._allocate_gpu_block()
            block_table[-1] = new_block
            self.gpu_allocator.free(last_block)
            return last_block.block_id, new_block.block_id
    
    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        # NOTE: fork does not allocate a new physical block.
        # Thus, it is always safe from OOM.
        src_block_table = self.block_tables[parent_seq.seq_id]
        # copy的是table，不是block本身
        self.block_tables[child_seq.seq_id] = src_block_table.copy()
        for block in src_block_table:
            block.ref_count += 1
    
    def _get_physical_blocks(
        self,
        seq_group: SequenceGroup,
    ) -> List[PhysicalTokenBlock]:
        # TODO
        # NOTE: Here, we assume that the physical blocks are only shared by
        # the sequences in the same group.
        blocks: Set[PhysicalTokenBlock] = set()
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                continue
            block_table = self.block_tables[seq.seq_id]
            for block in block_table:
                blocks.add(block)
        return list(blocks)

    def can_swap_in(self, seq_group: SequenceGroup) -> bool:
        blocks = self._get_physical_blocks(seq_group)
        num_swapped_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
        num_available_blocks = self._get_num_available_gpu_blocks()
        # NOTE: Conservatively, we assume that every sequence will allocate
        # at least one free block right after the swap-in.
        # NOTE: This should match the logic in can_append_slot().
        num_required_blocks = len(blocks) + num_swapped_seqs
        return (num_available_blocks - num_required_blocks
                >= self.watermark_blocks)

    def swap_in(self, seq_group: SequenceGroup) -> Dict[int, int]:
        # CPU block -> GPU block.
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                continue
            new_block_table: BlockTable = []
            block_table = self.block_tables[seq.seq_id]

            for cpu_block in block_table:
                if cpu_block in mapping:
                    gpu_block = mapping[cpu_block]
                    gpu_block.ref_count += 1
                else:
                    gpu_block = self._allocate_gpu_block()
                    mapping[cpu_block] = gpu_block
                new_block_table.append(gpu_block)
                # Free the CPU block swapped in to GPU.
                self.cpu_allocator.free(cpu_block)
            self.block_tables[seq.seq_id] = new_block_table
        
        block_id_mapping = {
            cpu_block.block_id: gpu_block.block_id
            for cpu_block, gpu_block in mapping.items()
        }
        return block_id_mapping

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        blocks = self._get_physical_blocks(seq_group)
        return len(blocks) <= self.cpu_allocator.get_num_free_blocks()

    def swap_out(self, seq_group: SequenceGroup) -> Dict[int, int]:
        # GPU block -> CPU block.
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                continue
            new_block_table: BlockTable = []
            block_table = self.block_tables[seq.seq_id]

            for gpu_block in block_table:
                if gpu_block in mapping:
                    cpu_block = mapping[gpu_block]
                    cpu_block.ref_count += 1
                else:
                    cpu_block = self.cpu_allocator.allocate()
                    mapping[gpu_block] = cpu_block
                new_block_table.append(cpu_block)
                # Free the GPU block swapped out to CPU.
                self.gpu_allocator.free(gpu_block)
            self.block_tables[seq.seq_id] = new_block_table
        
        block_id_mapping = {
            gpu_block.block_id: cpu_block.block_id
            for gpu_block, cpu_block in mapping.items()
        }
        return block_id_mapping

    def _free_block_table(self, block_table: BlockTable) -> None:
        for block in block_table:
            if block.device == Device.CPU:
                self.cpu_allocator.free(block)
            else:
                self.gpu_allocator.free(block)
    
    def free(self, seq: Sequence) -> None:
        if seq.seq_id not in self.block_tables:
            return
        block_table = self.block_tables[seq.seq_id]
        self._free_block_table(block_table)
        # 资源块引用减少不一定释放，但是block table是一定要释放的了
        del self.block_tables[seq.seq_id]

    def reset(self) -> None:
        for block_table in self.block_tables.values():
            self._free_block_table(block_table)
        self.block_tables.clear()
        # Sequence references are gone; now release the cache-owned references.
        for block in self.hash_manager.clear():
            self.gpu_allocator.free(block)

    def get_block_table(self, seq: Sequence) -> List[int]:
        block_table = self.block_tables[seq.seq_id]
        # 返回虚拟block
        return [block.block_id for block in block_table]

    def get_num_free_gpu_blocks(self) -> int:
        return self.gpu_allocator.get_num_free_blocks()

    def get_num_free_cpu_blocks(self) -> int:
        return self.cpu_allocator.get_num_free_blocks()
