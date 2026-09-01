import torch
from typing import Tuple, List, Dict

from torch.nn import parallel

from minivllm.configs.config import (
    ModelConfig,
    CacheConfig,
    ParallelConfig,
    SchedulerConfig
)

from minivllm.model_executor import InputMetadata, set_random_seed, get_model

from minivllm.model_executor.parallel_utils.parallel_state import (
    initialize_all_reduce_launcher, initialize_model_parallel)
from minivllm.sampling_params import SamplingParams
from minivllm.sequence import SequenceData, SequenceGroupMetadata, SequenceOutputs
from minivllm.worker.cache_engine import CacheEngine
from minivllm.worker.hybrid_cache import (
    GatedDeltaNetStateSpec,
    HybridCache,
)
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

        set_random_seed(self.model_config.seed)
        self.model = get_model(model_config)
        initialize_all_reduce_launcher(
            self.scheduler_config.max_num_batched_tokens,
            self.model_config.get_hidden_size(),
            self.model_config.dtype
        )

        # 各类 cache 配置，在后续 self.init_cache_engine() 函数中设置
        self.cache_config = None
        self.block_size = None
        self.cache_engine = None
        self.cache_events = None
        self.gpu_cache = None
        self.hybrid_cache = None

    @torch.inference_mode()
    def profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        cpu_swap_space: int,
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
                seq_data={group_id: seq_data},
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
        state_cache_size = CacheEngine.get_state_cache_size(
            self.model_config,
            self.parallel_config,
            self.scheduler_config.max_num_seqs,
        )
        num_gpu_blocks = int((total_gpu_memory * gpu_memory_utilization
                              - peak_memory - state_cache_size)
                             // cache_block_size)
        num_cpu_blocks = int(cpu_swap_space // cache_block_size)
        torch.cuda.empty_cache()

        set_random_seed(self.model_config.seed)
        return num_gpu_blocks, num_cpu_blocks

    def init_cache_engine(self, cache_config: CacheConfig) -> None:
        if (
            cache_config.enable_prefix_caching
            and self.model_config.architecture.num_linear_attention_layers
        ):
            raise ValueError(
                "Prefix caching for hybrid models requires recurrent-state "
                "snapshots and is not supported yet"
            )
        self.cache_config = cache_config
        self.block_size = cache_config.block_size
        self.cache_engine = CacheEngine(
            self.cache_config, self.model_config, self.parallel_config
        )
        self.cache_events = self.cache_engine.events
        if self.model_config.architecture.num_linear_attention_layers:
            full_attention_caches = {
                layer_idx: cache
                for layer_idx, cache in enumerate(self.cache_engine.gpu_cache)
                if cache is not None
            }
            self.hybrid_cache = HybridCache(
                layer_types=self.model_config.architecture.layer_types,
                full_attention_caches=full_attention_caches,
                state_spec=GatedDeltaNetStateSpec.from_text_config(
                    self.model_config.architecture.text_config
                ),
                max_num_seqs=self.scheduler_config.max_num_seqs,
                device="cuda",
            )
            self.gpu_cache = self.hybrid_cache
        else:
            self.gpu_cache = self.cache_engine.gpu_cache

    def _prepare_inputs(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata]:
        seq_groups: List[Tuple[List[int], SamplingParams]] = []
        input_token_ids: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []
        prompt_lens: List[int] = []
        fresh_prompt_lens: List[int] = []
        cached_prompt_query_lens: List[int] = []
        cached_prompt_context_lens: List[int] = []
        cached_prompt_block_tables: List[List[int]] = []
        prompt_seq_ids: List[int] = []
        generation_seq_ids: List[int] = []

        prompt_metadata = [
            metadata for metadata in seq_group_metadata_list
            if metadata.is_prompt
        ]
        # Keep each attention input kind contiguous. Fresh prompts retain the
        # existing xFormers path; only prompts with a cached prefix use varlen.
        fresh_prompts: List[SequenceGroupMetadata] = []
        cached_prompts: List[SequenceGroupMetadata] = []
        for metadata in prompt_metadata:
            seq_id = next(iter(metadata.seq_data))
            if metadata.num_computed_tokens[seq_id] == 0:
                fresh_prompts.append(metadata)
            else:
                cached_prompts.append(metadata)

        def append_prompt(
            metadata: SequenceGroupMetadata,
            has_cached_prefix: bool,
        ) -> None:
            seq_ids = list(metadata.seq_data.keys())
            seq_id = seq_ids[0]
            seq_data = metadata.seq_data[seq_id]
            start = metadata.num_computed_tokens[seq_id]
            query_len = metadata.num_scheduled_tokens[seq_id]
            end = start + query_len
            assert end == seq_data.get_len()

            seq_groups.append((seq_ids, metadata.sampling_params))
            prompt_seq_ids.append(seq_id)
            prompt_lens.append(query_len)
            input_token_ids.extend(seq_data.get_token_ids()[start:end])
            input_positions.extend(range(start, end))

            if metadata.block_tables is None:
                # Cache profiling does not allocate physical blocks.
                slot_mapping.extend([0] * query_len)
                return

            block_table = metadata.block_tables[seq_id]
            for position in range(start, end):
                slot_mapping.append(
                    _get_slot_id(block_table, position, self.block_size))

            if has_cached_prefix:
                cached_prompt_query_lens.append(query_len)
                cached_prompt_context_lens.append(end)
                cached_prompt_block_tables.append(block_table)

        for metadata in fresh_prompts:
            seq_id = next(iter(metadata.seq_data))
            fresh_prompt_lens.append(metadata.num_scheduled_tokens[seq_id])
            append_prompt(metadata, has_cached_prefix=False)
        for metadata in cached_prompts:
            append_prompt(metadata, has_cached_prefix=True)

        # Decode inputs always contain one token per running sequence.
        context_lens: List[int] = []
        generation_block_tables: List[List[int]] = []
        for metadata in seq_group_metadata_list:
            if metadata.is_prompt:
                continue

            seq_ids = list(metadata.seq_data.keys())
            seq_groups.append((seq_ids, metadata.sampling_params))

            for seq_id in seq_ids:
                generation_seq_ids.append(seq_id)
                seq_data = metadata.seq_data[seq_id]
                position = metadata.num_computed_tokens[seq_id]
                query_len = metadata.num_scheduled_tokens[seq_id]
                assert query_len == 1 and position + 1 == seq_data.get_len()
                input_token_ids.append(seq_data.get_token_ids()[position])
                input_positions.append(position)

                block_table = metadata.block_tables[seq_id]
                generation_block_tables.append(block_table)
                context_lens.append(position + 1)
                slot_mapping.append(
                    _get_slot_id(block_table, position, self.block_size))
        
        # Padding to multiple of 8, used for Tensor Cores
        input_token_ids = _pad_to_alignment(input_token_ids, multiple_of=8)
        input_positions = _pad_to_alignment(input_positions, multiple_of=8)

        # Convert to tensor
        tokens_tensor = torch.as_tensor(input_token_ids, dtype=torch.long, device="cuda")
        positions_tensor = torch.as_tensor(input_positions, dtype=torch.long, device="cuda")
        slot_mapping_tensor = torch.as_tensor(slot_mapping, dtype=torch.int, device="cuda")
        context_lens_tensor = torch.as_tensor(context_lens, dtype=torch.int, device="cuda")
        block_tables_tensor = _make_block_table_tensor(generation_block_tables)

        cached_prompt_cu_seqlens = [0]
        for query_len in cached_prompt_query_lens:
            cached_prompt_cu_seqlens.append(
                cached_prompt_cu_seqlens[-1] + query_len)
        cached_prompt_cu_seqlens_tensor = torch.as_tensor(
            cached_prompt_cu_seqlens, dtype=torch.int, device="cuda")
        cached_prompt_context_lens_tensor = torch.as_tensor(
            cached_prompt_context_lens, dtype=torch.int, device="cuda")
        cached_prompt_block_tables_tensor = _make_block_table_tensor(
            cached_prompt_block_tables)

        seq_data: Dict[int, SequenceData] = {}
        for seq_group_metadata in seq_group_metadata_list:
            seq_data.update(seq_group_metadata.seq_data)
        
        input_metadata = InputMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
            slot_mapping=slot_mapping_tensor,
            context_lens=context_lens_tensor,
            max_context_len=max(context_lens, default=0),
            block_tables=block_tables_tensor,
            fresh_prompt_lens=fresh_prompt_lens,
            cached_prompt_query_lens=cached_prompt_query_lens,
            cached_prompt_cu_seqlens=cached_prompt_cu_seqlens_tensor,
            cached_prompt_context_lens=cached_prompt_context_lens_tensor,
            cached_prompt_block_tables=cached_prompt_block_tables_tensor,
            max_cached_prompt_context_len=max(
                cached_prompt_context_lens, default=0),
            prompt_seq_ids=prompt_seq_ids,
            generation_seq_ids=generation_seq_ids,
        )
        return tokens_tensor, positions_tensor, input_metadata

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> Dict[int, SequenceOutputs]:
        # Issue cache operations.
        issued_cache_op = False
        if blocks_to_swap_in:
            self.cache_engine.swap_in(blocks_to_swap_in)
            issued_cache_op = True
        if blocks_to_swap_out:
            self.cache_engine.swap_out(blocks_to_swap_out)
            issued_cache_op = True
        if blocks_to_copy:
            self.cache_engine.copy(blocks_to_copy)
            issued_cache_op = True
        
        if issued_cache_op:
            cache_events = self.cache_events
        else:
            cache_events = None
        
        # If there is no input, we don't need to execute the model.
        if not seq_group_metadata_list:
            if cache_events is not None:
                for event in cache_events:
                    if event is not None:
                        event.wait()
            return {}
        
        # Prepare input tensors.
        input_tokens, input_positions, input_metadata = self._prepare_inputs(
            seq_group_metadata_list=seq_group_metadata_list)
        if self.hybrid_cache is not None:
            input_metadata.state_slot_mapping = self.hybrid_cache.acquire(
                input_metadata.prompt_seq_ids
                + input_metadata.generation_seq_ids
            )
        
        # Execute the model.
        # output: List[int (seq_id), SequenceOutputs]
        output = self.model(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=self.gpu_cache,
            input_metadata=input_metadata,
            cache_events=cache_events,
        )
        return output


def _init_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: str
) -> None:
    """Initialize the distributed environment."""
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=parallel_config.world_size,
        rank=rank,
        init_method=distributed_init_method
    )
    # A small all_reduce for warmup.
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    initialize_model_parallel(parallel_config.tensor_parallel_size,
                              parallel_config.pipeline_parallel_size)


def _pad_to_alignment(x: List[int], multiple_of: int) -> List[int]:
    return x + [0] * ((-len(x)) % multiple_of)

def _pad_to_max(x: List[int], max_len: int) -> List[int]:
    return x + [0] * (max_len - len(x))


def _get_slot_id(
    block_table: List[int],
    position: int,
    block_size: int,
) -> int:
    block_number = block_table[position // block_size]
    block_offset = position % block_size
    return block_number * block_size + block_offset


def _make_block_table_tensor(
    block_tables: List[List[int]],
) -> torch.Tensor:
    if not block_tables:
        return torch.empty((0, 0), dtype=torch.int, device="cuda")
    max_num_blocks = max(len(block_table) for block_table in block_tables)
    padded_block_tables = [
        _pad_to_max(block_table, max_num_blocks)
        for block_table in block_tables
    ]
    return torch.as_tensor(
        padded_block_tables, dtype=torch.int, device="cuda")
