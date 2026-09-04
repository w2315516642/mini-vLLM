import math
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

import torch

from torch.nn import parallel

from minivllm.configs.config import (
    ModelConfig,
    CacheConfig,
    ParallelConfig,
    SchedulerConfig
)
from minivllm.configs.pd_config import KVTransferBackend, PDConfig, PDRole
from minivllm.distributed.kv_transfer import (
    CacheLayout,
    P2PTransferBackend,
    TransferPlan,
    register_cache_layout,
)

from minivllm.model_executor import (
    InputMetadata,
    get_dspark_model,
    get_model,
    set_random_seed,
)

from minivllm.model_executor.parallel_utils.parallel_state import (
    initialize_all_reduce_launcher, initialize_model_parallel)
from minivllm.sampling_params import SamplingParams
from minivllm.sequence import SequenceData, SequenceGroupMetadata, SequenceOutputs
from minivllm.spec_decode.draft_metadata import DraftAttentionMetadata
from minivllm.spec_decode.dspark_context import TargetHiddenStateCollector
from minivllm.worker.cache_engine import CacheEngine
from minivllm.worker.draft_cache import DraftCacheEngine
from minivllm.worker.gdn_replay import GatedDeltaNetReplay, replay_buffer_bytes
from minivllm.worker.hybrid_cache import (
    GatedDeltaNetStateSpec,
    HybridCache,
)
from minivllm.utils.device import get_gpu_memory


@dataclass(frozen=True)
class _DraftRequest:
    """One output token that can seed the next DSpark proposal block."""

    seq_id: int
    output: SequenceOutputs
    block_table: List[int]
    next_position: int
    anchor_token_id: int
    sampling_params: SamplingParams
    output_history: List[int]
    draft_width: int

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
        pd_config: Optional[PDConfig] = None,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.pd_config = pd_config or PDConfig()

        _init_distributed_environment(parallel_config, rank, 
                                      distributed_init_method)

        set_random_seed(self.model_config.seed)
        self.model = get_model(model_config)
        self.draft_model = None
        self.draft_config = None
        self.draft_cache_engine = None
        self.draft_gpu_cache = None
        self._draft_probabilities: Dict[int, torch.Tensor] = {}
        if self.scheduler_config.draft_model is not None:
            self.draft_model = get_dspark_model(
                self.scheduler_config.draft_model,
                dtype=self.model_config.dtype,
                cache_dir=self.scheduler_config.draft_download_dir,
                use_dummy_weights=self.model_config.use_dummy_weights,
            )
            self.draft_config = self.draft_model.config
            self._validate_dspark_target()
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
        self.transfer_backend = None
        self.transfer_layout = None

    def _validate_dspark_target(self) -> None:
        """Fail early when a draft checkpoint cannot share target modules."""
        if not hasattr(self.model, "model") or not hasattr(self.model, "lm_head"):
            raise ValueError("DSpark requires a Qwen target with embedding and LM head")
        target_backbone = self.model.model
        if not hasattr(target_backbone, "embed_tokens") or not hasattr(
            target_backbone, "layers"
        ):
            raise ValueError("DSpark target backbone is missing required Qwen modules")
        if self.draft_config.hidden_size != self.model_config.get_hidden_size():
            raise ValueError("DSpark and target hidden sizes must match")
        if self.draft_config.vocab_size > int(self.model.config.vocab_size):
            raise ValueError("DSpark vocabulary cannot exceed the target vocabulary")
        if self.draft_config.mask_token_id >= int(self.model.config.vocab_size):
            raise ValueError("DSpark mask token is outside the target embedding table")
        target_layers = len(target_backbone.layers)
        if any(
            layer_id < 0 or layer_id >= target_layers
            for layer_id in self.draft_config.target_layer_ids
        ):
            raise ValueError("DSpark auxiliary layer ids exceed the target depth")
        requested_width = self.scheduler_config.num_speculative_tokens
        if requested_width <= 0 or requested_width > self.draft_config.block_size:
            raise ValueError(
                "DSpark speculative width must be between 1 and its block size"
            )

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
        draft_cache_block_size = self._get_draft_cache_block_size(block_size)
        cache_block_size += draft_cache_block_size
        state_cache_size = CacheEngine.get_state_cache_size(
            self.model_config,
            self.parallel_config,
            self.scheduler_config.max_num_seqs,
        )
        usable_cache_memory = (
            total_gpu_memory * gpu_memory_utilization
            - peak_memory
            - state_cache_size
            - self._get_gdn_replay_reserve()
            - self._get_draft_workspace_size(
                block_size, draft_cache_block_size
            )
            - self._get_draft_probability_reserve()
        )
        if usable_cache_memory <= 0:
            state_gib = state_cache_size / (1024 ** 3)
            draft_gib = (
                self._get_draft_workspace_size(
                    block_size, draft_cache_block_size
                )
                + self._get_draft_probability_reserve()
            ) / (1024 ** 3)
            raise ValueError(
                "Persistent runtime buffers do not fit in the configured GPU "
                f"memory budget: recurrent={state_gib:.2f} GiB, "
                f"GDN replay={self._get_gdn_replay_reserve() / (1024 ** 3):.2f} GiB, "
                f"DSpark={draft_gib:.2f} GiB for max_num_seqs="
                f"{self.scheduler_config.max_num_seqs}. Reduce "
                "--max-num-seqs or the speculative width."
            )
        num_gpu_blocks = int(usable_cache_memory // cache_block_size)
        num_cpu_blocks = int(cpu_swap_space // cache_block_size)
        torch.cuda.empty_cache()

        set_random_seed(self.model_config.seed)
        return num_gpu_blocks, num_cpu_blocks

    def _get_gdn_replay_reserve(self) -> int:
        num_layers = self.model_config.architecture.num_linear_attention_layers
        width = self.scheduler_config.num_speculative_tokens
        if not num_layers or not width:
            return 0
        spec = GatedDeltaNetStateSpec.from_text_config(
            self.model_config.architecture.text_config,
            self.parallel_config.tensor_parallel_size,
        )
        max_seqs = self.scheduler_config.max_num_seqs
        max_tokens = min(self.scheduler_config.max_num_batched_tokens,
                         max_seqs * (width + 1))
        return replay_buffer_bytes(spec, num_layers, max_seqs, max_tokens,
                                   self.model_config.dtype)

    def _get_draft_cache_block_size(self, block_size: int) -> int:
        if self.draft_config is None:
            return 0
        tp_size = self.parallel_config.tensor_parallel_size
        if self.draft_config.num_key_value_heads % tp_size:
            raise ValueError("DSpark KV heads must be divisible by TP size")
        return DraftCacheEngine.get_cache_block_size(
            num_layers=self.draft_config.num_hidden_layers,
            block_size=block_size,
            num_kv_heads=self.draft_config.num_key_value_heads // tp_size,
            head_dim=self.draft_config.head_dim,
            dtype=self.model_config.dtype,
        )

    def _get_draft_workspace_blocks(self, block_size: int) -> int:
        if self.draft_config is None:
            return 0
        blocks_per_request = math.ceil(
            self.draft_config.block_size / block_size
        )
        return self.scheduler_config.max_num_seqs * blocks_per_request

    def _get_draft_workspace_size(
        self,
        block_size: int,
        draft_cache_block_size: int,
    ) -> int:
        return (
            self._get_draft_workspace_blocks(block_size)
            * draft_cache_block_size
        )

    def _get_draft_probability_reserve(self) -> int:
        """Reserve worst-case FP32 q distributions for exact rejection."""
        if self.draft_config is None:
            return 0
        return (
            self.scheduler_config.max_num_seqs
            * self.scheduler_config.num_speculative_tokens
            * self.draft_config.vocab_size
            * torch.tensor([], dtype=torch.float32).element_size()
        )

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
                    self.model_config.architecture.text_config,
                    self.parallel_config.tensor_parallel_size,
                ),
                max_num_seqs=self.scheduler_config.max_num_seqs,
                device="cuda",
            )
            self.gpu_cache = self.hybrid_cache
        else:
            self.gpu_cache = self.cache_engine.gpu_cache
        if self.draft_config is not None:
            tp_size = self.parallel_config.tensor_parallel_size
            self.draft_cache_engine = DraftCacheEngine(
                num_layers=self.draft_config.num_hidden_layers,
                num_gpu_blocks=cache_config.num_gpu_blocks,
                num_cpu_blocks=cache_config.num_cpu_blocks,
                block_size=cache_config.block_size,
                num_kv_heads=(
                    self.draft_config.num_key_value_heads // tp_size
                ),
                head_dim=self.draft_config.head_dim,
                dtype=self.model_config.dtype,
                workspace_blocks=self._get_draft_workspace_blocks(
                    cache_config.block_size
                ),
            )
            self.draft_gpu_cache = self.draft_cache_engine.gpu_cache
        self._init_transfer_engine()

    def _init_transfer_engine(self) -> None:
        """Register rank-local persistent caches for P/D data movement."""
        if not self.pd_config.enabled:
            return
        if self.pd_config.backend != KVTransferBackend.TCP:
            raise ValueError(
                "LLMEngine PD mode currently requires the tcp backend; "
                "the memory backend is reserved for unit tests"
            )
        endpoint = self.pd_config.endpoint_for_rank(self.rank)
        self.transfer_backend = P2PTransferBackend(
            endpoint,
            timeout_s=self.pd_config.transfer_timeout_s,
        )
        full_attention_caches = {
            layer_idx: cache
            for layer_idx, cache in enumerate(self.cache_engine.gpu_cache)
            if cache is not None
        }
        state_pools = (
            None
            if self.hybrid_cache is None
            else self.hybrid_cache.get_state_pools()
        )
        draft_attention_caches = (
            None
            if self.draft_gpu_cache is None
            else {
                layer_idx: cache
                for layer_idx, cache in enumerate(self.draft_gpu_cache)
            }
        )
        self.transfer_layout = register_cache_layout(
            backend=self.transfer_backend,
            block_size=self.block_size,
            full_attention_caches=full_attention_caches,
            linear_state_pools=state_pools,
            draft_attention_caches=draft_attention_caches,
        )

    def get_transfer_layout(self) -> CacheLayout:
        if self.transfer_layout is None:
            raise RuntimeError("worker has no PD transfer layout")
        return self.transfer_layout

    def get_source_state_slots(self, seq_ids: List[int]) -> Dict[int, int]:
        """Return P slots after prefill has created recurrent state."""
        if self.pd_config.role != PDRole.PREFILL:
            raise RuntimeError("source state slots are available only on P")
        if self.hybrid_cache is None:
            return {}
        return {
            seq_id: self.hybrid_cache.get_state_slot(seq_id)
            for seq_id in seq_ids
        }

    def reserve_decode_state_slots(self, seq_ids: List[int]) -> Dict[int, int]:
        """Allocate D recurrent-state destinations before transfer starts."""
        if self.pd_config.role != PDRole.DECODE:
            raise RuntimeError("decode state slots are available only on D")
        if self.hybrid_cache is None:
            return {}
        self.hybrid_cache.acquire(seq_ids)
        return {
            seq_id: self.hybrid_cache.get_state_slot(seq_id)
            for seq_id in seq_ids
        }

    def execute_cache_transfer(
        self,
        plan: TransferPlan,
    ) -> Dict[str, object]:
        """Run one rank-local P-push batch and wait for the remote ACK."""
        if self.pd_config.role != PDRole.PREFILL:
            raise RuntimeError("cache transfers must be submitted by P workers")
        if self.transfer_backend is None:
            raise RuntimeError("worker transfer backend is not initialized")
        handle = self.transfer_backend.submit(plan)
        status = self.transfer_backend.wait(
            handle, self.pd_config.transfer_timeout_s
        )
        return {
            "transfer_id": handle.transfer_id,
            "status": status.value,
            "error": handle.error,
            "total_bytes": plan.total_bytes,
        }

    def close_transfer_engine(self) -> None:
        if self.transfer_backend is not None:
            self.transfer_backend.close()

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
        prompt_sample_indices: List[int] = []
        speculative_seq_ids: List[int] = []
        speculative_token_ids: List[int] = []
        speculative_token_blocks: List[List[int]] = []
        speculative_hidden_indices: List[Tuple[int, ...]] = []
        speculative_sampling_params: List[SamplingParams] = []
        multimodal_inputs = {}
        multimodal_token_maps: List[Tuple[int, int, int, int]] = []
        has_multimodal_positions = any(
            metadata.multi_modal_inputs is not None
            for metadata in seq_group_metadata_list
        )
        input_positions_3d: List[List[int]] = [[], [], []]

        def append_positions(
            metadata: SequenceGroupMetadata,
            positions: range,
        ) -> None:
            multimodal = metadata.multi_modal_inputs
            for position in positions:
                if multimodal is None:
                    position_values = (position, position, position)
                elif position < len(multimodal.token_type_ids):
                    assert multimodal.position_ids is not None
                    position_values = tuple(
                        row[position] for row in multimodal.position_ids
                    )
                else:
                    decode_position = position + multimodal.rope_delta
                    position_values = (
                        decode_position,
                        decode_position,
                        decode_position,
                    )
                if has_multimodal_positions:
                    for row, value in zip(
                        input_positions_3d, position_values
                    ):
                        row.append(value)
                else:
                    input_positions.append(position)

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
            if metadata.is_speculative:
                draft_tokens = metadata.speculative_token_blocks[seq_id]
                assert (
                    query_len == len(draft_tokens) + 1
                    and start + 1 == seq_data.get_len()
                )
                query_token_ids = [seq_data.get_token_ids()[start]] + draft_tokens
                speculative_seq_ids.append(seq_id)
                speculative_token_ids.append(draft_tokens[0])
                speculative_token_blocks.append(draft_tokens)
                speculative_hidden_indices.append(tuple(range(
                    len(input_token_ids), len(input_token_ids) + query_len
                )))
                speculative_sampling_params.append(metadata.sampling_params)
            else:
                assert 0 < query_len and end <= seq_data.get_len()
                query_token_ids = seq_data.get_token_ids()[start:end]

            if metadata.do_sample:
                seq_groups.append((seq_ids, metadata.sampling_params))
                prompt_sample_indices.append(
                    len(input_token_ids) + query_len - 1
                )
            prompt_seq_ids.append(seq_id)
            prompt_lens.append(query_len)
            packed_start = len(input_token_ids)
            input_token_ids.extend(query_token_ids)
            append_positions(metadata, range(start, end))

            multimodal = metadata.multi_modal_inputs
            if multimodal is not None:
                multimodal_inputs[seq_id] = multimodal
                image_index = sum(
                    token_type == 1
                    for token_type in multimodal.token_type_ids[:start]
                )
                video_index = sum(
                    token_type == 2
                    for token_type in multimodal.token_type_ids[:start]
                )
                for offset, position in enumerate(range(start, end)):
                    if position >= len(multimodal.token_type_ids):
                        continue
                    token_type = multimodal.token_type_ids[position]
                    if token_type == 1:
                        multimodal_token_maps.append(
                            (packed_start + offset, seq_id, 1, image_index)
                        )
                        image_index += 1
                    elif token_type == 2:
                        multimodal_token_maps.append(
                            (packed_start + offset, seq_id, 2, video_index)
                        )
                        video_index += 1

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
                append_positions(metadata, range(position, position + 1))
                if metadata.multi_modal_inputs is not None:
                    multimodal_inputs[seq_id] = metadata.multi_modal_inputs

                block_table = metadata.block_tables[seq_id]
                generation_block_tables.append(block_table)
                context_lens.append(position + 1)
                slot_mapping.append(
                    _get_slot_id(block_table, position, self.block_size))
        
        # Padding to multiple of 8, used for Tensor Cores
        input_token_ids = _pad_to_alignment(input_token_ids, multiple_of=8)
        if has_multimodal_positions:
            input_positions_3d = [
                _pad_to_alignment(row, multiple_of=8)
                for row in input_positions_3d
            ]
        else:
            input_positions = _pad_to_alignment(
                input_positions, multiple_of=8
            )

        # Convert to tensor
        tokens_tensor = torch.as_tensor(input_token_ids, dtype=torch.long, device="cuda")
        positions_tensor = torch.as_tensor(
            input_positions_3d if has_multimodal_positions else input_positions,
            dtype=torch.long,
            device="cuda",
        )
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
            prompt_sample_indices=prompt_sample_indices,
            speculative_seq_ids=speculative_seq_ids,
            speculative_token_ids=speculative_token_ids,
            speculative_token_blocks=speculative_token_blocks,
            speculative_hidden_indices=speculative_hidden_indices,
            enable_mtp=(
                getattr(
                    getattr(self, "scheduler_config", None),
                    "num_speculative_tokens",
                    0,
                )
                == 1
                and getattr(
                    getattr(self, "scheduler_config", None),
                    "draft_model",
                    None,
                )
                is None
            ),
            speculative_sampling_params=speculative_sampling_params,
            multimodal_inputs=multimodal_inputs,
            multimodal_token_maps=multimodal_token_maps,
        )
        return tokens_tensor, positions_tensor, input_metadata

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        state_seq_ids_to_release: Optional[List[int]] = None,
        state_copies: Optional[Dict[int, int]] = None,
    ) -> Dict[int, SequenceOutputs]:
        self.apply_hybrid_state_operations(
            state_seq_ids_to_release or [], state_copies or {})
        # Issue cache operations.
        issued_cache_op = False
        if blocks_to_swap_in:
            self.cache_engine.swap_in(blocks_to_swap_in)
            if self.draft_cache_engine is not None:
                self.draft_cache_engine.swap_in(blocks_to_swap_in)
            issued_cache_op = True
        if blocks_to_swap_out:
            self.cache_engine.swap_out(blocks_to_swap_out)
            if self.draft_cache_engine is not None:
                self.draft_cache_engine.swap_out(blocks_to_swap_out)
            issued_cache_op = True
        if blocks_to_copy:
            self.cache_engine.copy(blocks_to_copy)
            if self.draft_cache_engine is not None:
                self.draft_cache_engine.copy(blocks_to_copy)
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
            if self.draft_cache_engine is not None:
                self.draft_cache_engine.wait_for_cache_ops()
            return {}
        if self.draft_model is not None and any(
            metadata.multi_modal_inputs is not None
            for metadata in seq_group_metadata_list
        ):
            raise NotImplementedError(
                "DSpark proposals currently support text requests only"
            )
        
        # Prepare input tensors.
        input_tokens, input_positions, input_metadata = self._prepare_inputs(
            seq_group_metadata_list=seq_group_metadata_list)
        self._attach_saved_draft_probabilities(input_metadata)
        if self.hybrid_cache is not None:
            input_metadata.state_slot_mapping = self.hybrid_cache.acquire(
                input_metadata.prompt_seq_ids
                + input_metadata.generation_seq_ids
            )
        if self.hybrid_cache is not None and input_metadata.speculative_seq_ids:
            input_metadata.gdn_replay = GatedDeltaNetReplay(
                input_metadata,
                self.hybrid_cache.snapshot(input_metadata.speculative_seq_ids),
            )
        
        # Execute the model.
        # output: List[int (seq_id), SequenceOutputs]
        hidden_collector = (
            TargetHiddenStateCollector(self.draft_config.target_layer_ids)
            if self.draft_config is not None
            else None
        )
        model_kwargs = {}
        if hidden_collector is not None:
            model_kwargs["hidden_state_collector"] = hidden_collector
        output = self.model(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=self.gpu_cache,
            input_metadata=input_metadata,
            cache_events=cache_events,
            **model_kwargs,
        )
        if hidden_collector is not None:
            self.draft_cache_engine.wait_for_cache_ops()
            target_hidden = hidden_collector.concatenate()[
                :input_metadata.num_valid_tokens
            ]
            self.draft_model.materialize_context_kv(
                target_hidden,
                input_positions[:input_metadata.num_valid_tokens],
                input_metadata.slot_mapping,
                self.draft_gpu_cache,
            )
        partially_accepted = {
            seq_id: output[seq_id].num_computed_tokens
            for seq_id, draft_tokens in zip(
                input_metadata.speculative_seq_ids,
                input_metadata.speculative_token_blocks,
            )
            if output[seq_id].num_computed_tokens < len(draft_tokens) + 1
        }
        if partially_accepted:
            assert self.hybrid_cache is not None and input_metadata.gdn_replay is not None
            input_metadata.gdn_replay.commit(
                self.hybrid_cache,
                {seq_id: output[seq_id].num_computed_tokens
                 for seq_id in input_metadata.speculative_seq_ids},
            )
        # Release the transaction before allocating the next draft workspace.
        input_metadata.gdn_replay = None
        if self._should_attach_dspark_drafts():
            self._attach_dspark_drafts(output, seq_group_metadata_list)
        return output

    def _should_attach_dspark_drafts(self) -> bool:
        """Generate proposals where the next target step executes locally.

        A prefill worker still materializes Draft K/V and transfers it to D,
        but D creates the first proposal after consuming P's sampled anchor.
        This keeps the large stochastic q distributions out of the handoff.
        """
        return (
            self.draft_model is not None
            and self.pd_config.role != PDRole.PREFILL
        )

    def _attach_saved_draft_probabilities(
        self,
        input_metadata: InputMetadata,
    ) -> None:
        """Move the previous proposal distributions into target verification."""
        probability_blocks = []
        has_stochastic_request = False
        for seq_id, draft_tokens, sampling_params in zip(
            input_metadata.speculative_seq_ids,
            input_metadata.speculative_token_blocks,
            input_metadata.speculative_sampling_params,
        ):
            probabilities = self._draft_probabilities.pop(seq_id, None)
            if sampling_params.temperature == 0.0:
                probability_blocks.append(None)
                continue
            has_stochastic_request = True
            if probabilities is None:
                raise RuntimeError(
                    f"Missing DSpark draft probabilities for sequence {seq_id}"
                )
            if probabilities.ndim != 2 or probabilities.shape[0] < len(
                draft_tokens
            ):
                raise RuntimeError(
                    f"Incomplete DSpark draft probabilities for sequence {seq_id}"
                )
            probability_blocks.append(probabilities[:len(draft_tokens)])
        if has_stochastic_request:
            input_metadata.speculative_draft_probs = probability_blocks

    def _attach_dspark_drafts(
        self,
        outputs: Dict[int, SequenceOutputs],
        metadata_list: List[SequenceGroupMetadata],
    ) -> None:
        requests = self._collect_draft_requests(outputs, metadata_list)
        if not requests:
            return
        gamma = self.draft_config.block_size
        token_rows = [
            [request.anchor_token_id]
            + [self.draft_config.mask_token_id] * (gamma - 1)
            for request in requests
        ]
        token_ids = torch.tensor(
            token_rows, dtype=torch.long, device="cuda"
        ).flatten()
        positions = torch.tensor(
            [
                position
                for request in requests
                for position in range(
                    request.next_position, request.next_position + gamma
                )
            ],
            dtype=torch.long,
            device="cuda",
        )
        draft_metadata = self._build_draft_metadata(requests)
        proposal = self.draft_model.propose_paged(
            self.model.model.embed_tokens(token_ids),
            positions,
            self.draft_gpu_cache,
            draft_metadata,
            self.model.lm_head.weight,
            torch.tensor(
                [request.anchor_token_id for request in requests],
                dtype=torch.long,
                device="cuda",
            ),
            [request.sampling_params for request in requests],
            [request.output_history for request in requests],
        )
        for batch_index, request in enumerate(requests):
            width = request.draft_width
            confidence = None
            if proposal.confidence is not None:
                confidence = proposal.confidence[
                    batch_index, :width
                ].tolist()
            request.output.set_draft_tokens(
                proposal.token_ids[batch_index, :width].tolist(),
                confidence,
            )
            probabilities = proposal.draft_probs[batch_index]
            if probabilities is None:
                self._draft_probabilities.pop(request.seq_id, None)
            else:
                self._draft_probabilities[request.seq_id] = (
                    probabilities[:width].detach()
                )

    def _collect_draft_requests(
        self,
        outputs: Dict[int, SequenceOutputs],
        metadata_list: List[SequenceGroupMetadata],
    ) -> List[_DraftRequest]:
        requests = []
        gamma = min(
            self.scheduler_config.num_speculative_tokens,
            self.draft_config.block_size,
        )
        eos_token_id = getattr(self.model.config, "eos_token_id", None)
        eos_token_ids = (
            set(eos_token_id)
            if isinstance(eos_token_id, (list, tuple, set))
            else ({eos_token_id} if eos_token_id is not None else set())
        )
        for metadata in metadata_list:
            params = metadata.sampling_params
            supported = (
                params.best_of == 1
                and not params.use_beam_search
                and not params.stop
                and (
                    params.temperature > 0.0
                    or (
                        params.presence_penalty == 0.0
                        and params.frequency_penalty == 0.0
                    )
                )
            )
            for seq_id, seq_data in metadata.seq_data.items():
                output = outputs.get(seq_id)
                if output is None:
                    continue
                self._draft_probabilities.pop(seq_id, None)
                if not supported or not output.output_token_ids:
                    continue
                anchor_token_id = int(output.output_token_ids[-1])
                if not params.ignore_eos and anchor_token_id in eos_token_ids:
                    continue
                output_len = (
                    len(seq_data.output_token_ids)
                    + len(output.output_token_ids)
                )
                remaining = params.max_tokens - output_len
                draft_width = min(gamma, remaining - 1)
                if draft_width <= 0:
                    continue
                committed = (
                    output.num_computed_tokens
                    if output.num_computed_tokens is not None
                    else metadata.num_scheduled_tokens[seq_id]
                )
                next_position = metadata.num_computed_tokens[seq_id] + committed
                if next_position + self.draft_config.block_size > (
                    self.draft_config.max_position_embeddings
                ):
                    continue
                requests.append(_DraftRequest(
                    seq_id=seq_id,
                    output=output,
                    block_table=list(metadata.block_tables[seq_id]),
                    next_position=next_position,
                    anchor_token_id=anchor_token_id,
                    sampling_params=params,
                    output_history=(
                        list(seq_data.output_token_ids)
                        + list(output.output_token_ids)
                    ),
                    draft_width=draft_width,
                ))
        return requests

    def _build_draft_metadata(
        self,
        requests: List[_DraftRequest],
    ) -> DraftAttentionMetadata:
        gamma = self.draft_config.block_size
        blocks_per_request = math.ceil(gamma / self.block_size)
        workspace_ids = list(self.draft_cache_engine.workspace_block_ids)
        block_tables = []
        slot_mapping = []
        context_lens = []
        for batch_index, request in enumerate(requests):
            workspace_start = batch_index * blocks_per_request
            request_workspace = workspace_ids[
                workspace_start:workspace_start + blocks_per_request
            ]
            block_table = _extend_draft_block_table(
                request.block_table,
                request.next_position,
                gamma,
                self.block_size,
                request_workspace,
            )
            block_tables.append(block_table)
            for position in range(
                request.next_position, request.next_position + gamma
            ):
                slot_mapping.append(
                    _get_slot_id(block_table, position, self.block_size)
                )
            context_lens.append(request.next_position + gamma)
        cu_seqlens = [index * gamma for index in range(len(requests) + 1)]
        return DraftAttentionMetadata(
            query_lens=[gamma] * len(requests),
            cu_seqlens_q=torch.tensor(
                cu_seqlens, dtype=torch.int32, device="cuda"
            ),
            context_lens=torch.tensor(
                context_lens, dtype=torch.int32, device="cuda"
            ),
            block_tables=_make_block_table_tensor(block_tables),
            slot_mapping=torch.tensor(
                slot_mapping, dtype=torch.int32, device="cuda"
            ),
        )

    def apply_hybrid_state_operations(
        self,
        seq_ids_to_release: List[int],
        state_copies: Dict[int, int],
    ) -> None:
        """Apply scheduler lifecycle changes to persistent GDN states."""
        for child_seq_id, parent_seq_id in state_copies.items():
            probabilities = self._draft_probabilities.get(parent_seq_id)
            if probabilities is not None:
                self._draft_probabilities[child_seq_id] = probabilities.clone()
        for seq_id in seq_ids_to_release:
            self._draft_probabilities.pop(seq_id, None)
        copy_multimodal = getattr(self.model, "copy_multimodal_cache", None)
        release_multimodal = getattr(
            self.model, "release_multimodal_cache", None
        )
        if copy_multimodal is not None:
            for child_seq_id, parent_seq_id in state_copies.items():
                copy_multimodal(parent_seq_id, child_seq_id)
        if release_multimodal is not None:
            release_multimodal(seq_ids_to_release)
        if self.hybrid_cache is None:
            return
        # Beam children need the parent's post-forward state before either
        # sequence is released by a stopping rule in the same engine step.
        for child_seq_id, parent_seq_id in state_copies.items():
            self.hybrid_cache.copy(parent_seq_id, child_seq_id)
        self.hybrid_cache.release_existing(seq_ids_to_release)


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


def _extend_draft_block_table(
    block_table: List[int],
    start_position: int,
    query_len: int,
    block_size: int,
    workspace_block_ids: List[int],
) -> List[int]:
    """Append temporary draft-only blocks without changing target ownership."""
    if start_position < 0 or query_len <= 0 or block_size <= 0:
        raise ValueError("Invalid DSpark block-table range")
    extended = list(block_table)
    required_blocks = math.ceil((start_position + query_len) / block_size)
    missing_blocks = required_blocks - len(extended)
    if missing_blocks > len(workspace_block_ids):
        raise RuntimeError(
            "DSpark workspace cannot cover the next proposal block"
        )
    if missing_blocks > 0:
        extended.extend(workspace_block_ids[:missing_blocks])
    return extended
