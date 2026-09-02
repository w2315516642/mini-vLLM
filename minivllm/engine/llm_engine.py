from typing import Any, List, Optional
import time


from loguru import logger

from minivllm.configs import (
    ModelConfig, ParallelConfig, CacheConfig, PDConfig, PDRole,
    SchedulerConfig)
from minivllm.distributed.kv_transfer import CacheLayout, TransferPlan
from minivllm.engine.pd_coordinator import DecodeReservation
from minivllm.engine.pd_handoff import RequestHandoff
from minivllm.kv_cache.scheduler import Scheduler
from minivllm.multimodal import MultiModalInputs
from minivllm.engine.arg_utils import EngineArgs
from minivllm.engine.ray_utils import DeviceID, ray, initialize_cluster
from minivllm.engine.tokenizer_utils import detokenize_incrementally, get_tokenizer
from minivllm.outputs import RequestOutput
from minivllm.sampling_params import SamplingParams
from minivllm.sequence import Sequence, SequenceGroup, SequenceStatus
from minivllm.utils import (
    get_hash_fn_by_name, 
    init_none_hash, 
    get_seq_block_hasher, 
    Counter, 
    BlockHasher
)
from minivllm.worker.worker import Worker

class LLMEngine:
    """ 核心，负责模型初始化、资源分配和推理 """
    def __init__(
        self, 
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        pd_config: PDConfig,
        distributed_init_method: str,
        stage_devices: List[List[DeviceID]],
        log_stats: bool,
    ) -> None:
        self.model_config = model_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.pd_config = pd_config
        self.log_stats = log_stats
        self._verify_args()

        self.tokenizer = get_tokenizer(model_config.model)
        self.seq_counter = Counter()
        self._sealed_handoffs: List[RequestHandoff] = []

        # Create the parallel GPU workers.
        self.workers: List[Worker] = []
        assert len(stage_devices) == 1, "Only support one stage for now."
        for rank, node_resource, _ in stage_devices[0]:
            worker_cls = Worker
            runtime_env = {"env_vars": {"NCCL_IB_DISABLE": "1", "NCCL_IP_OVER_IB": "0"}}
            if self.parallel_config.worker_use_ray:
                worker_cls = ray.remote(
                    num_cpus=0,
                    num_gpus=1,
                    resources={node_resource: 1e-4},
                    runtime_env=runtime_env
                )(worker_cls).remote
            
            worker = worker_cls(
                model_config,
                parallel_config,
                scheduler_config,
                rank,
                distributed_init_method,
                pd_config,
            )
            self.workers.append(worker)
        
        self._init_cache()

        self.scheduler = Scheduler(scheduler_config, cache_config, log_stats)

        self.seq_block_hasher: Optional[BlockHasher] = None
        if self.cache_config.enable_prefix_caching:
            caching_hash_fn = get_hash_fn_by_name(
                self.cache_config.prefix_caching_hash_fn)
            init_none_hash(caching_hash_fn)
            self.seq_block_hasher = get_seq_block_hasher(caching_hash_fn)
        
    def _verify_args(self) -> None:
        self.model_config.verify_with_parallel_config(self.parallel_config)
        self.cache_config.verify_with_parallel_config(self.parallel_config)
        if self.pd_config.enabled and self.cache_config.enable_prefix_caching:
            raise ValueError(
                "prefix caching across P/D roles is not implemented yet"
            )

    def _init_cache(self) -> None:
        """Profiles the memory usage and initializes the KV cache."""
        # Get the maximum number of blocks that can be allocated on GPU and CPU.
        num_blocks = self._run_workers(
            "profile_num_available_blocks",
            get_all_outputs=True,
            block_size=self.cache_config.block_size,
            gpu_memory_utilization=self.cache_config.gpu_memory_utilization,
            cpu_swap_space=self.cache_config.swap_space_bytes,
        )

        # Since we use a shared centralized controller, we take the minimum
        # number of blocks across all workers to make sure all the memory
        # operators can be applied to all workers.
        num_gpu_blocks = min(b[0] for b in num_blocks)
        num_cpu_blocks = min(b[1] for b in num_blocks)
        # Log
        logger.info(f'# GPU blocks: {num_gpu_blocks}, '
                    f'# CPU blocks: {num_cpu_blocks}')
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks
        
        # Initialize the cache.
        self._run_workers("init_cache_engine", cache_config=self.cache_config)

    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_configs = engine_args.create_engine_configs()
        parallel_config = engine_configs[2]
        # Initialize the cluster.
        distributed_init_method, devices = initialize_cluster(parallel_config)
        # Create the LLM engine.
        engine = cls(*engine_configs, distributed_init_method, devices,
                    log_stats=not engine_args.disable_log_stats)
        return engine


    def step(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()
        if (not seq_group_metadata_list) and scheduler_outpus.is_empty():
            # Nothing to do.
            return []
        
        # Execute the model.
        output = self._run_workers(
            "execute_model",
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
            blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
            blocks_to_copy=scheduler_outputs.blocks_to_copy,
            state_seq_ids_to_release=(
                scheduler_outputs.state_seq_ids_to_release),
            state_copies=scheduler_outputs.state_copies,
        )
        # Update the scheduler with the model outputs.
        seq_groups = self.scheduler.update(output, scheduler_outputs)

        # Decode the sequences.
        self._decode_sequences(seq_groups)
        # Stop the sequences that meet the stopping criteria.
        self._stop_sequences(seq_groups)
        # Free the finished sequence groups.
        self.scheduler.free_finished_seq_groups()
        self._flush_pending_state_operations()

        # Create the outputs.
        request_outputs: List[RequestOutput] = []
        for seq_group in seq_groups:
            request_output = RequestOutput.from_seq_group(seq_group)
            request_outputs.append(request_output)
        return request_outputs

    def add_request(
        self,
        request_id: str,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        multi_modal_inputs: Optional[MultiModalInputs] = None,
    ) -> None:
        """Add a request to the engine's request pool.

        The request is added to the request pool and will be processed by the
        scheduler as `engine.step()` is called. The exact scheduling policy is
        determined by the scheduler.

        Args:
            request_id: The unique ID of the request.
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters for text generation.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.
            arrival_time: The arrival time of the request. If None, we use
                the current time.
        """
        if arrival_time is None:
            arrival_time = time.time()
        if prompt_token_ids is None:
            assert prompt is not None
            prompt_token_ids = self.tokenizer.encode(prompt)
        if multi_modal_inputs is not None:
            if not self.model_config.is_multimodal:
                raise ValueError(
                    "Multimodal inputs require a model with vision_config"
                )
            multi_modal_inputs = multi_modal_inputs.with_positions(
                prompt_token_ids,
                self.model_config.spatial_merge_size,
            )

        # Create the sequences.
        block_size = self.cache_config.block_size
        seqs: List[Sequence] = []
        for _ in range(sampling_params.best_of):
            seq_id = next(self.seq_counter)
            # 在这里把哈希函数传进去，让Sequence初始化的时候计算哈希
            seq = Sequence(
                seq_id=seq_id, 
                prompt=prompt, 
                prompt_token_ids=prompt_token_ids, 
                block_size=block_size, 
                block_hasher=self.seq_block_hasher,
            )
            seqs.append(seq)
        
        # Create the sequence group.
        seq_group = SequenceGroup(
            request_id,
            seqs,
            sampling_params,
            arrival_time,
            multi_modal_inputs=multi_modal_inputs,
        )

        # Add the sequence group to the scheduler.
        self.scheduler.add_seq_group(seq_group)

    def abort_request(self, request_id: str) -> None:
        """Aborts a request with the given ID."""
        self.scheduler.abort_seq_group(request_id)
        self._flush_pending_state_operations()

    def get_num_unfinished_requests(self) -> int:
        """Gets the number of unfinished requests."""
        return self.scheduler.get_num_unfinished_seq_groups()

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests."""
        return self.scheduler.has_unfinished_seqs()

    def _run_workers(
        self,
        method: str,
        get_all_outputs: bool = False,
        *args,
        **kwargs
    ) -> Any:
        """Runs the given method on all workers."""
        all_outputs = []
        for worker in self.workers:
            executor = getattr(worker, method)
            if self.parallel_config.worker_use_ray:
                executor = executor.remote

            output = executor(*args, **kwargs)
            all_outputs.append(output)
        
        if self.parallel_config.worker_use_ray:
            all_outputs = ray.get(all_outputs)
        
        if get_all_outputs:
            return all_outputs
        # Make sure all workers have the same results.
        output = all_outputs[0]
        for other_output in all_outputs[1:]:
            assert output == other_output
        return output

    def _run_worker_at(self, worker_index: int, method: str, *args, **kwargs):
        if worker_index < 0 or worker_index >= len(self.workers):
            raise IndexError(f"worker index {worker_index} is out of range")
        worker = self.workers[worker_index]
        executor = getattr(worker, method)
        if self.parallel_config.worker_use_ray:
            return ray.get(executor.remote(*args, **kwargs))
        return executor(*args, **kwargs)


    def step(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        seq_group_metadata_list, scheduler_outputs = self.scheduler.scheduler()
        if (not seq_group_metadata_list) and scheduler_outputs.is_empty():
            return []
        
        # Execute the model.
        output = self._run_workers(
            "execute_model",
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
            blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
            blocks_to_copy=scheduler_outputs.blocks_to_copy,
            state_seq_ids_to_release=(
                scheduler_outputs.state_seq_ids_to_release),
            state_copies=scheduler_outputs.state_copies,
        )
        # Update the scheduler with the model outputs.
        seq_groups = self.scheduler.update(output, scheduler_outputs)

        # Decode the sequences.
        self._decode_sequences(seq_groups)
        # Stop the sequences that meet the stopping criteria.
        self._stop_sequences(seq_groups)
        if self.pd_config.role == PDRole.PREFILL:
            self._seal_prefilled_requests(seq_groups)
        # Free the finished sequence groups.
        self.scheduler.free_finished_seq_groups()
        self._flush_pending_state_operations()
        
        # Create the outputs.
        request_outputs: List[RequestOutput] = []
        for seq_group in seq_groups:
            request_output = RequestOutput.from_seq_group(seq_group)
            request_outputs.append(request_output)
        return request_outputs

    def _seal_prefilled_requests(
        self,
        seq_groups: List[SequenceGroup],
    ) -> None:
        """Freeze completed prompts before the scheduler can decode them."""
        for seq_group in seq_groups:
            if seq_group not in self.scheduler.running:
                continue
            active = seq_group.get_seqs(status=SequenceStatus.RUNNING)
            if not active or any(
                seq.num_computed_tokens != seq.get_prompt_len()
                or seq.get_output_len() == 0
                for seq in active
            ):
                continue
            block_tables = {
                seq.seq_id: tuple(
                    self.scheduler.block_manager.get_block_table(seq)
                )
                for seq in active
            }
            seq_ids = [seq.seq_id for seq in active]
            rank_state_slots = self._run_workers(
                "get_source_state_slots",
                get_all_outputs=True,
                seq_ids=seq_ids,
            )
            state_slots = rank_state_slots[0]
            if any(item != state_slots for item in rank_state_slots[1:]):
                raise RuntimeError("P workers disagree on hybrid state slots")
            handoff = RequestHandoff.from_sequence_group(
                seq_group,
                block_tables=block_tables,
                state_slots=state_slots,
            )
            self.scheduler.seal_prefilled_seq_group(seq_group)
            self._sealed_handoffs.append(handoff)

    def pop_prefill_handoffs(self) -> List[RequestHandoff]:
        """Return newly sealed requests to the external PD coordinator."""
        if self.pd_config.role != PDRole.PREFILL:
            raise RuntimeError("only a prefill engine produces handoffs")
        handoffs = self._sealed_handoffs
        self._sealed_handoffs = []
        return handoffs

    def get_transfer_layouts(self) -> List[CacheLayout]:
        if not self.pd_config.enabled:
            raise RuntimeError("unified engines have no transfer layouts")
        return self._run_workers(
            "get_transfer_layout", get_all_outputs=True
        )

    def prepare_decode_handoff(
        self,
        handoff: RequestHandoff,
    ) -> DecodeReservation:
        """Reserve D resources but keep the request invisible to scheduling."""
        if self.pd_config.role != PDRole.DECODE:
            raise RuntimeError("only a decode engine accepts handoffs")
        seq_group = handoff.rebuild_sequence_group(self.cache_config.block_size)
        if len(handoff.sequences) != 1:
            raise ValueError("PD handoff currently supports one sequence")
        progress = handoff.sequences[0].num_computed_tokens
        try:
            block_tables = self.scheduler.reserve_transferred_seq_group(
                seq_group, progress
            )
            seq_ids = [sequence.seq_id for sequence in handoff.sequences]
            rank_state_slots = self._run_workers(
                "reserve_decode_state_slots",
                get_all_outputs=True,
                seq_ids=seq_ids,
            )
            state_slots = rank_state_slots[0]
            if any(item != state_slots for item in rank_state_slots[1:]):
                raise RuntimeError("D workers disagree on hybrid state slots")
        except Exception:
            self.scheduler.abort_seq_group(handoff.request_id)
            self._flush_pending_state_operations()
            raise
        return DecodeReservation(
            block_tables={
                seq_id: tuple(block_ids)
                for seq_id, block_ids in block_tables.items()
            },
            state_slots=state_slots,
        )

    def activate_decode_handoff(self, request_id: str) -> None:
        if self.pd_config.role != PDRole.DECODE:
            raise RuntimeError("only a decode engine activates handoffs")
        self.scheduler.activate_transferred_seq_group(request_id)

    def abort_decode_handoff(self, request_id: str) -> None:
        self.scheduler.abort_seq_group(request_id)
        self._flush_pending_state_operations()

    def release_prefill_handoff(self, request_id: str) -> None:
        """Drop P cache only after all rank transfers have completed."""
        if self.pd_config.role != PDRole.PREFILL:
            raise RuntimeError("only a prefill engine owns sealed handoffs")
        self.scheduler.release_sealed_seq_group(request_id)
        self._flush_pending_state_operations()

    def execute_rank_cache_transfer(
        self,
        rank: int,
        plan: TransferPlan,
    ):
        if self.pd_config.role != PDRole.PREFILL:
            raise RuntimeError("only a prefill engine submits cache transfers")
        return self._run_worker_at(rank, "execute_cache_transfer", plan)

    def _flush_pending_state_operations(self) -> None:
        releases, copies = self.scheduler.pop_pending_state_operations()
        if not releases and not copies:
            return
        self._run_workers(
            "apply_hybrid_state_operations",
            seq_ids_to_release=releases,
            state_copies=copies,
        )

    def _decode_sequences(self, seq_groups: List[SequenceGroup]) -> None:
        """Decodes the sequence outputs."""
        for seq_group in seq_groups:
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                while len(seq.output_tokens) < seq.get_output_len():
                    token_index = len(seq.output_tokens)
                    token_id = seq.get_output_token_ids()[token_index]
                    new_token, new_output_text = detokenize_incrementally(
                        self.tokenizer,
                        seq.output_tokens,
                        token_id,
                        skip_special_tokens=True,
                    )
                    seq.output_tokens.append(new_token)
                    seq.output_text += new_output_text

    def _stop_sequences(self, seq_groups: List[SequenceGroup]) -> None:
        """Stop the finished sequences."""
        for seq_group in seq_groups:
            sampling_params = seq_group.sampling_params
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                # Check if the sequence has generated a stop string.
                stopped = False
                for stop_str in sampling_params.stop:
                    if seq.output_text.endswith(stop_str):
                        # Truncate the output text so that the stop string is
                        # not included in the output.
                        seq.output_text = seq.output_text[:-len(stop_str)]
                        self.scheduler.free_seq(
                            seq, SequenceStatus.FINISHED_STOPPED)
                        stopped = True
                        break
                if stopped:
                    continue
                    
                # Check if the sequence has reached max_tokens.
                if seq.get_output_len() >= sampling_params.max_tokens:
                    self.scheduler.free_seq(
                        seq, SequenceStatus.FINISHED_LENGTH_CAPPED
                    )
                    continue
                # Check if the sequence has generated the EOS token.
                if not sampling_params.ignore_eos:
                    if seq.get_last_token_id() == self.tokenizer.eos_token_id:
                        self.scheduler.free_seq(
                            seq, SequenceStatus.FINISHED_STOPPED
                        )
                        continue


if __name__ == "__main__":
    from pathlib import Path

    model_path = Path(__file__).parent.parent.parent / "models" / "Qwen3-0.6B"
    prompts = "Hello?"
    engine = LLMEngine(model_path, 64)
    outputs = engine.generate(prompts)
    print(outputs)
