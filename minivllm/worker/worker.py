import torch
from typing import Tuple, List, Dict

from minivllm.configs.config import (
    ModelConfig,
    CacheConfig,
    ParallelConfig,
    SchedulerConfig
)

from minivllm.model_executor import InputMetadata, set_random_seed

from minivllm.sampling_params import SamplingParams
from minivllm.sequence import SequenceData, SequenceGroupMetadata
from minivllm.worker.cache_engine import CacheEngine
from minivllm.utils.device import get_gpu_memory

class Worker:
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        rank: int,
        distributed_init_method: str,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.rank = rank
        self.distributed_init_method = distributed_init_method

        _init_distributed_environment(parallel_config, rank, 
                                      distributed_init_method)

        # TODO: initializing methods
        set_random_seed(self.model_config.seed)
        self.model = get_model(model_config)


        # 各类 cache 配置，在后续 self.init_cache_engine() 函数中设置
        self.cache_config = None
        self.block_size = None
        self.cache_engine = None
        self.cache_events = None
        self.gpu_cache = None

    @torch.inference_mode()
    def profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        cpu_swap_size: int,
    ) -> Tuple[int, int]:
        "进行一次推理以统计使用的内存大小，据此计算并返回该worker可用的最大block数"
        # 重置当前内存使用情况
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        sampling_params = SamplingParams(top_p=0.99,
                                         top_k=self.model.config.vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs
        seqs: List[SequenceGroupMetadata] = []
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs +
                        (group_id < max_num_batched_tokens % max_num_seqs))
            seq_data = SequenceData([0] * seq_len)
            seq = SequenceGroupMetadata(
                request_id=group_id,
                is_prompt=True,
                seq_data={group_id, seq_data},
                sampling_params=sampling_params,
                block_tables=None
            )
            seqs.append(seq)

        input_tokens, input_positions, input_metadata = self._prepare_inputs(seqs)
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        self.model(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=[(None, None)] * num_layers,
            input_metadata=input_metadata,
            cache_events=None
        )

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
        total_gpu_memory = get_gpu_memory()
        assert total_gpu_memory > peak_memory, (
            f"Total gpu memory {total_gpu_memory} is less than "
            f"max gpu memory usage {peak_memory}")
        cache_block_size = CacheEngine.get_cache_block_size(
            block_size, self.model_config, self.parallel_config)
        num_gpu_blocks = int((total_gpu_memory * gpu_memory_utilization
                              - peak_memory) // cache_block_size)
        num_cpu_blocks = int(cpu_swap_size // cache_block_size)
        torch.cuda.empty_cache()

        set_random_seed(self.model_config.seed)
        return num_gpu_blocks, num_cpu_blocks

    def init_cache_engine(self, cache_config: CacheConfig) -> None:
        self.cache_config = cache_config
        self.block_size = cache_config.block_size
        self.cache_engine = CacheEngine(
            self.cache_config, self.model_config, self.parallel_config
        )
        self.cache_events = self.cache_engine.events
        self.gpu_cache = self.cache_engine.gpu_cache

    def _prepare_inputs(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata]:
        seq_groups: List[Tuple[List[int], SamplingParams]] = []
        input_token_ids: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []

        # 添加prompt tokens
        prompt_lens: List[int] = []
        for seq_group_metadata in seq_group_metadata_list:
            if not seq_group_metadata.is_prompt:
                continue

            seq_ids = List(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            seq_id = seq_ids[0]

            # NOTE: 这里是把所有的prompt token ids都一起取出来了，不涉及到 chunked prefill
            # TODO: 这里的逻辑需要修改，以添加 chunked prefill 功能
            seq_data = seq_group_metadata.seq_data[seq_data]
            prompt_token_ids = seq_data.get_token_ids()
            prompt_len = len(prompt_token_ids)
            prompt_lens.append(prompt_len)

            # 拼接tokens
            input_token_ids.extend(prompt_token_ids)
            # NOTE: Here we assume that the first token in prompts 
            # is always the first token in the sequence. 
            input_positions.extend(range(len(prompt_token_ids)))

            if seq_group_metadata.block_tables is None:
                # 在内存测试的时候还没有分配blocks，因此用不到这个映射
                slot_mapping.extend([0] * prompt_len)
                continue
            
            # Mapping token to slot id
            block_table = seq_group_metadata.block_tables[seq_id]
            for i in range(prompt_len):
                block_number = block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)
        
        # Add generation token
        max_context_len = 0
        max_num_blocks_per_seq = 0
        context_lens: List[int] = []
        # TODO: 这里chunked prefill要改，prompt的处理也需要block_tables
        generation_block_tables: List[List[int]] = []
        for seq_group_metadata in seq_group_metadata_list:
            if seq_group_metadata.is_prompt:
                continue
            
            seq_ids = List(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_token_ids.append(generation_token)

                context_len = seq_data.get_len()
                position = context_len - 1
                input_positions.append(position)

                block_table = seq_group_metadata.block_tables[seq_id]
                generation_block_tables.append(block_table)

                max_context_len = max(max_context_len, context_len)
                max_num_blocks_per_seq = max(
                    max_num_blocks_per_seq, len(block_table)
                )
                context_lens.append(context_len)

                block_number = block_table[position // self.block_size]
                block_offset = position % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)
        
        # Padding to multiple of 8, used for Tensor Cores
        input_token_ids = _pad_to_alignment(input_token_ids, multiple_of=8)
        input_positions = _pad_to_alignment(input_positions, multiple_of=8)

        # Convert to tensor
        tokens_tensor = torch.as_tensor(input_token_ids, dtype=torch.long, device="cuda")
        positions_tensor = torch.as_tensor(input_positions, dtype=torch.long, device="cuda")
        slot_mapping_tensor = torch.as_tensor(slot_mapping, dtype=torch.int, device="cuda")
        context_lens_tensor = torch.as_tensor(context_lens, dtype=torch.int, device="cuda")
        padded_block_tables = [
            _pad_to_max(block_table, max_num_blocks_per_seq)
            for block_table in generation_block_tables
        ]    
        block_tables_tensor = torch.as_tensor(padded_block_tables, dtype=torch.int, device="cuda")

        seq_data: Dict[int, SequenceData] = {}
        for seq_group_metadata in seq_group_metadata_list:
            seq_data.update(seq_group_metadata.seq_data)
        
        input_metadata = InputMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
            slot_mapping=slot_mapping_tensor,
            context_lens=context_lens_tensor,
            max_context_len=max_context_len,
            block_tables=block_tables_tensor
        )
        return tokens_tensor, positions_tensor, input_metadata


def _init_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: str
) -> None:
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=parallel_config.world_size,
        rank=rank,
        init_method=distributed_init_method
    )

    # warm up
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    # TODO: initialize_model_parallel()


def _pad_to_alignment(x: List[int], multiple_of: int) -> List[int]:
    return x + [0] * ((-len(x)) % multiple_of)

def _pad_to_max(x: List[int], max_len: int) -> List[int]:
    return x + [0] * (max_len - len(x))