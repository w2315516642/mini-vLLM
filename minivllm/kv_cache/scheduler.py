import time
from enum import Enum, auto
from typing import Dict, List, Set, Tuple, Optional

from loguru import logger

from minivllm.configs import CacheConfig, SchedulerConfig
from minivllm.kv_cache.policy import PolicyFactory
from minivllm.kv_cache.block_manager import BlockSpaceManager
from minivllm.sequence import (
    SequenceStatus, SequenceData, SequenceGroup,
    Sequence, SequenceGroupMetadata, SequenceOutputs
)

_LOGGING_INTERVAL_SEC = 5


class PreemptionMode(Enum):
    """Preemption modes.

    1. Swapping: Swap out the blocks of the preempted sequences to CPU memory
    and swap them back in when the sequences are resumed.
    2. Recomputation: Discard the blocks of the preempted sequences and
    recompute them when the sequences are resumed, treating the sequences as
    new prompts.
    """
    SWAP = auto()
    RECOMPUTE = auto()


class SchedulerOutputs:
    def __init__(
        self,
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        num_scheduled_tokens: Dict[int, int],
        sampled_seq_ids: Optional[Set[int]] = None,
        state_seq_ids_to_release: Optional[List[int]] = None,
        state_copies: Optional[Dict[int, int]] = None,
        speculative_seq_ids: Optional[Set[int]] = None,
    ) -> None:
        self.blocks_to_swap_in = blocks_to_swap_in
        self.blocks_to_swap_out = blocks_to_swap_out
        self.blocks_to_copy = blocks_to_copy
        self.num_scheduled_tokens = num_scheduled_tokens
        self.sampled_seq_ids = sampled_seq_ids or set()
        self.state_seq_ids_to_release = state_seq_ids_to_release or []
        # child_seq_id -> parent_seq_id
        self.state_copies = state_copies or {}
        self.speculative_seq_ids = speculative_seq_ids or set()
        # Swap in and swap out should never happen at the same time.
        assert not (blocks_to_swap_in and blocks_to_swap_out)

    def is_empty(self) -> bool:
        return (not self.blocks_to_swap_in
                and not self.blocks_to_swap_out
                and not self.blocks_to_copy
                and not self.num_scheduled_tokens
                and not self.state_seq_ids_to_release
                and not self.state_copies)


class Scheduler:
    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        log_stats: bool,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.log_stats = log_stats

        # Instantiate the scheduling policy.
        self.policy = PolicyFactory.get_policy(policy_name='fcfs')
        # Create the block space manager.
        self.block_manager = BlockSpaceManager(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=self.cache_config.num_gpu_blocks,
            num_cpu_blocks=self.cache_config.num_cpu_blocks,
            enable_prefix_caching=self.cache_config.enable_prefix_caching,
        )

        # Sequence groups in the WAITING state.
        self.waiting: List[SequenceGroup] = []
        # Sequence groups in the RUNNING state.
        self.running: List[SequenceGroup] = []
        # Sequence groups in the SWAPPED state.
        self.swapped: List[SequenceGroup] = []
        self._pending_state_releases: Set[int] = set()
        self._pending_state_copies: Dict[int, int] = {}

        self.last_logging_time: float = 0.0
        # List[timestamp, num_tokens]
        self.num_input_tokens: List[Tuple[float, int]] = []

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq_group)

    def abort_seq_group(self, request_id: str) -> None:
        for state_queue in [self.waiting, self.running, self.swapped]:
            for seq_group in state_queue:
                if seq_group.request_id == request_id:
                    # Remove the sequence group from the state queue.
                    state_queue.remove(seq_group)
                    for seq in seq_group.seqs:
                        if seq.is_finished():
                            continue
                        self.free_seq(seq, SequenceStatus.FINISHED_ABORTED)
                    return
    
    def has_unfinished_seqs(self) -> bool:
        return self.waiting or self.running or self.swapped
    
    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)

    def pop_pending_state_operations(self) -> Tuple[List[int], Dict[int, int]]:
        """Return worker-side recurrent-state operations accumulated so far."""
        releases = sorted(self._pending_state_releases)
        copies = self._pending_state_copies.copy()
        self._pending_state_releases.clear()
        self._pending_state_copies.clear()
        return releases, copies

    def _can_speculate(self, seq_group: SequenceGroup) -> bool:
        if getattr(self.scheduler_config, "num_speculative_tokens", 0) != 1:
            return False
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        if len(seqs) != 1 or seqs[0].speculative_token_id is None:
            return False
        params = seq_group.sampling_params
        return (
            params.temperature == 0.0
            and params.best_of == 1
            and not params.use_beam_search
            and params.presence_penalty == 0.0
            and params.frequency_penalty == 0.0
            and not params.stop
            and seqs[0].get_output_len() + 2 <= params.max_tokens
        )
    
    def _schedule(self) -> Tuple[SchedulerOutputs, List[str]]:
        # Blocks that need to be swaped or copied before model execution.
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}
        num_scheduled_tokens: Dict[int, int] = {}
        sampled_seq_ids: Set[int] = set()
        speculative_seq_ids: Set[int] = set()
        num_batched_tokens = 0
        prompt_group_ids: List[str] = []
        
        # Fix the current time.
        now = time.time()

        # NOTE(woosuk): We prioritize the sequence groups in the RUNNING state
        # in order to minimize the preemption overheads.
        # Preemption happens only when there is no available slot to keep all
        # the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        pending_running = self.policy.sort_by_priority(now, self.running)

        # Schedule prompt continuations and reserve decode slots. Prompt chunks
        # consume the budget once per group because all best-of sequences share
        # the same prompt computation.
        running: List[SequenceGroup] = []
        preempted: List[SequenceGroup] = []
        while pending_running:
            seq_group = pending_running.pop(0)
            running_seqs = seq_group.get_seqs(SequenceStatus.RUNNING)
            # Decode normally trails known tokens by exactly one: the token
            # sampled in the previous step. A larger gap means recomputation;
            # no output tokens means the original prompt is still in prefill.
            is_prefill = any(
                seq.get_output_len() == 0
                or seq.num_computed_tokens + 1 < seq.get_len()
                for seq in running_seqs
            )
            if is_prefill:
                remaining_tokens = {
                    seq.get_len() - seq.num_computed_tokens
                    for seq in running_seqs
                }
                if len(remaining_tokens) != 1:
                    raise ValueError(
                        "Sequences sharing a prompt must have equal prefill "
                        "progress"
                    )
                budget = (
                    self.scheduler_config.max_num_batched_tokens
                    - num_batched_tokens
                )
                chunk_size = min(remaining_tokens.pop(), budget)
                if chunk_size == 0:
                    running.append(seq_group)
                    running.extend(pending_running)
                    break
                prompt_group_ids.append(seq_group.request_id)
                for seq in running_seqs:
                    num_scheduled_tokens[seq.seq_id] = chunk_size
                    if seq.num_computed_tokens + chunk_size == seq.get_len():
                        sampled_seq_ids.add(seq.seq_id)
                num_batched_tokens += chunk_size
                running.append(seq_group)
                continue

            use_speculation = self._can_speculate(seq_group)
            tokens_per_seq = 2 if use_speculation else 1
            num_decode_tokens = len(running_seqs) * tokens_per_seq
            if (
                num_batched_tokens + num_decode_tokens
                > self.scheduler_config.max_num_batched_tokens
            ):
                running.append(seq_group)
                running.extend(pending_running)
                break
            while not self.block_manager.can_append_slots(
                seq_group, tokens_per_seq
            ):
                if pending_running:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq_group = pending_running.pop(-1)
                    self._preempt(victim_seq_group, blocks_to_swap_out)
                    preempted.append(victim_seq_group)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self._preempt(seq_group, blocks_to_swap_out)
                    preempted.append(seq_group)
                    break
            else:
                # Append new slots to the sequence group.
                self._append_slot(seq_group, blocks_to_copy)
                for seq in running_seqs:
                    num_scheduled_tokens[seq.seq_id] = tokens_per_seq
                    sampled_seq_ids.add(seq.seq_id)
                    if use_speculation:
                        self.block_manager.append_lookahead_slot(seq)
                        speculative_seq_ids.add(seq.seq_id)
                num_batched_tokens += num_decode_tokens
                running.append(seq_group)
        self.running = running

        # Swap in the sequence groups in the SWAPPED state if possible.
        self.swapped = self.policy.sort_by_priority(now, self.swapped)
        while self.swapped and not blocks_to_swap_out:
            seq_group = self.swapped[0]
            # If the sequence group has been preempted in this step, stop.
            if seq_group in preempted:
                break
            # If the sequence group cannot be swapped in, stop.
            if not self.block_manager.can_swap_in(seq_group):
                break
            
            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
            num_curr_seqs = sum(
                group.num_seqs(status=SequenceStatus.RUNNING)
                for group in self.running
            )
            if num_curr_seqs + num_new_seqs > self.scheduler_config.max_num_seqs:
                break
            if (
                num_batched_tokens + num_new_seqs
                > self.scheduler_config.max_num_batched_tokens
            ):
                break

            seq_group = self.swapped.pop(0)
            self._swap_in(seq_group, blocks_to_swap_in)
            self._append_slot(seq_group, blocks_to_copy)
            for seq in seq_group.get_seqs(SequenceStatus.RUNNING):
                num_scheduled_tokens[seq.seq_id] = 1
                sampled_seq_ids.add(seq.seq_id)
            num_batched_tokens += num_new_seqs
            self.running.append(seq_group)

        # Join waiting sequences if possible.
        # NOTE(woosuk): The sequence groups in the SWAPPED state are strictly
        # prioritized over the sequence groups in the WAITING state.
        # This is because we want to bound the amount of CPU memory taken by
        # the swapped sequence groups.
        if not self.swapped:
            # Optimization: We do not sort the waiting queue since the preempted
            # sequence groups are added to the front and the new sequence groups
            # are added to the back.
            while self.waiting:
                seq_group = self.waiting[0]
                # If the sequence group has been preempted in this step, stop.
                if seq_group in preempted:
                    break
                # If the sequence group cannot be allocated, stop.
                if not self.block_manager.can_allocate(seq_group):
                    break
                    
                # If the number of batched tokens exceeds the limit, stop.
                seq = seq_group.get_seqs()[0]
                num_cached_tokens = self.block_manager.get_num_cached_tokens(
                    seq_group)
                num_prompt_tokens = seq.get_len() - num_cached_tokens
                available_budget = (
                    self.scheduler_config.max_num_batched_tokens
                    - num_batched_tokens
                )
                if available_budget == 0:
                    break
                chunk_size = min(num_prompt_tokens, available_budget)
                    
                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.num_seqs(status=SequenceStatus.WAITING)
                num_curr_seqs = sum(
                    group.num_seqs(status=SequenceStatus.RUNNING)
                    for group in self.running
                )
                if num_curr_seqs + num_new_seqs > self.scheduler_config.max_num_seqs:
                    break

                seq_group = self.waiting.pop(0)
                self._allocate(seq_group)
                self.running.append(seq_group)
                num_batched_tokens += chunk_size
                prompt_group_ids.append(seq_group.request_id)
                for seq in seq_group.get_seqs(SequenceStatus.RUNNING):
                    num_scheduled_tokens[seq.seq_id] = chunk_size
                    if seq.num_computed_tokens + chunk_size == seq.get_len():
                        sampled_seq_ids.add(seq.seq_id)

        state_releases, state_copies = self.pop_pending_state_operations()
        scheduler_outputs = SchedulerOutputs(
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            num_scheduled_tokens=num_scheduled_tokens,
            sampled_seq_ids=sampled_seq_ids,
            state_seq_ids_to_release=state_releases,
            state_copies=state_copies,
            speculative_seq_ids=speculative_seq_ids,
        )
        if not self.log_stats:
            return scheduler_outputs, prompt_group_ids
        
        # TODO: Logging status.
        logger.warning("Not support logging yet.")
        return scheduler_outputs, prompt_group_ids

    def scheduler(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        scheduler_outputs, prompt_group_ids = self._schedule()

        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for seq_group in self.running:
            scheduled_seqs = [
                seq for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING)
                if seq.seq_id in scheduler_outputs.num_scheduled_tokens
            ]
            if not scheduled_seqs:
                continue
            is_speculative = any(
                seq.seq_id in scheduler_outputs.speculative_seq_ids
                for seq in scheduled_seqs
            )
            is_prompt = (
                seq_group.request_id in prompt_group_ids or is_speculative
            )

            seq_data: Dict[int, SequenceData] = {}
            block_tables: Dict[int, List[int]] = {}
            num_computed_tokens: Dict[int, int] = {}
            num_scheduled_tokens: Dict[int, int] = {}
            for seq in scheduled_seqs:
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                block_tables[seq_id] = self.block_manager.get_block_table(seq)
                num_computed_tokens[seq_id] = seq.num_computed_tokens
                num_scheduled_tokens[seq_id] = (
                    scheduler_outputs.num_scheduled_tokens[seq_id])
            
            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                is_prompt=is_prompt,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
                block_tables=block_tables,
                num_computed_tokens=num_computed_tokens,
                num_scheduled_tokens=num_scheduled_tokens,
                do_sample=(not is_speculative and all(
                    seq.seq_id in scheduler_outputs.sampled_seq_ids
                    for seq in scheduled_seqs
                )),
                is_speculative=is_speculative,
                speculative_token_ids={
                    seq.seq_id: seq.speculative_token_id
                    for seq in scheduled_seqs
                    if seq.speculative_token_id is not None
                },
            )
            seq_group_metadata_list.append(seq_group_metadata)
        return seq_group_metadata_list, scheduler_outputs

    def update(
        self,
        seq_outputs: Dict[int, SequenceOutputs],    # [seq_id, output]
        scheduler_outputs: SchedulerOutputs,
    ) -> List[SequenceGroup]:
        sampled_groups: List[SequenceGroup] = []
        # Update only sequences selected in this iteration. A long prompt may
        # remain RUNNING while another request uses the rest of the budget.
        for seq_group in self.running:
            scheduled_seqs = [
                seq for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING)
                if seq.seq_id in scheduler_outputs.num_scheduled_tokens
            ]
            if not scheduled_seqs:
                continue
            # The model has consumed these known tokens. Advance the KV
            # progress before beam forking or appending newly sampled tokens.
            for seq in scheduled_seqs:
                output = seq_outputs.get(seq.seq_id)
                num_scheduled_tokens = (
                    output.num_computed_tokens
                    if output is not None
                    and output.num_computed_tokens is not None
                    else scheduler_outputs.num_scheduled_tokens[seq.seq_id]
                )
                seq.num_computed_tokens += num_scheduled_tokens
                max_computed = seq.get_len() + (
                    seq.seq_id in scheduler_outputs.speculative_seq_ids
                )
                assert seq.num_computed_tokens <= max_computed, (
                    f"Sequence {seq.seq_id} computed "
                    f"{seq.num_computed_tokens} tokens, but has "
                    f"{seq.get_len()} known tokens."
                )
                self.block_manager.cache_blocks(seq)

            should_sample = all(
                seq.seq_id in scheduler_outputs.sampled_seq_ids
                for seq in scheduled_seqs
            )
            if not should_sample:
                continue

            # A best-of prompt is computed once, so only its representative
            # sequence owns the resulting recurrent state. Seed the remaining
            # candidates before they enter independent decode paths.
            if (
                len(scheduled_seqs) > 1
                and all(seq.get_output_len() == 0 for seq in scheduled_seqs)
            ):
                parent_seq_id = scheduled_seqs[0].seq_id
                for child_seq in scheduled_seqs[1:]:
                    self._pending_state_copies[
                        child_seq.seq_id
                    ] = parent_seq_id

            # Process beam search results before processing the new tokens.
            for seq in scheduled_seqs:
                # 这里默认seq_outputs一定有running队列里面所有序列的结果
                output = seq_outputs[seq.seq_id]
                if seq.seq_id != output.parent_seq_id:
                    # The sequence is a fork of the parent sequence (beam search).
                    # Free the current sequence.
                    self.block_manager.free(seq)
                    # Fork the parent sequence.
                    parent_seq = seq_group.find(output.parent_seq_id)
                    parent_seq.fork(seq)
                    self.block_manager.fork(parent_seq, seq)
                    self._pending_state_copies[seq.seq_id] = parent_seq.seq_id
            # Process the new tokens.
            for seq in scheduled_seqs:
                # Append a new token to the sequence.
                output = seq_outputs[seq.seq_id]
                for token_id, logprobs in zip(
                    output.output_token_ids, output.output_logprobs
                ):
                    seq.append_token_id(token_id, logprobs)
                seq.speculative_token_id = output.draft_token_id
                self.block_manager.cache_blocks(seq)
            sampled_groups.append(seq_group)
        # Return a shallow copy of the running queue to prevent the queue
        # from being modified by the caller.
        return sampled_groups

    def free_seq(self, seq: Sequence, finish_status: SequenceStatus) -> None:
        seq.status = finish_status
        # TODO: 解计数
        self.block_manager.free(seq)
        seq.speculative_token_id = None
        self._pending_state_releases.add(seq.seq_id)
        self._pending_state_copies.pop(seq.seq_id, None)

    def free_finished_seq_groups(self) -> None:
        self.running = [
            seq_group for seq_group in self.running
            if not seq_group.is_finished()
        ]

    def _allocate(self, seq_group: SequenceGroup) -> None:
        self.block_manager.allocate(seq_group)
        for seq in seq_group.get_seqs():
            seq.status = SequenceStatus.RUNNING
        
    def _append_slot(
        self,
        seq_group: SequenceGroup,
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        for seq in seq_group.get_seqs(SequenceStatus.RUNNING):
            ret = self.block_manager.append_slot(seq)
            if ret is not None:
                src_block, dst_block = ret
                if src_block in blocks_to_copy:
                    blocks_to_copy[src_block].append(dst_block)
                else:
                    blocks_to_copy[src_block] = [dst_block]
    
    def _preempt(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
        preemption_mode: Optional[PreemptionMode] = None
    ) -> None:
        # If preemption mode is not specified, we determine the mode as follows:
        # We use recomputation by default since it incurs lower overhead than
        # swapping. However, when the sequence group has multiple sequences
        # (e.g., beam search), recomputation is not supported. In such a case,
        # we use swapping instead.
        # As swapped sequences are prioritized over waiting sequences,
        # sequence groups with multiple sequences are implicitly prioritized
        # over sequence groups with a single sequence.
        # Support recomputation for sequence groups with multiple
        # sequences. This may require a more sophisticated CUDA kernel.
        if preemption_mode is None:
            seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
            if len(seqs) == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP
        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            self._preempt_by_swap(seq_group, blocks_to_swap_out)
        else:
            assert False, 'Invalid preemption mode.'

    def _preempt_by_recompute(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        # Only one sequence can be recomputed.
        assert len(seqs) == 1
        for seq in seqs:
            seq.status = SequenceStatus.WAITING
            self.block_manager.free(seq)
            # Recompute starts from a fresh block table. Prefix allocation may
            # immediately restore part of this progress from the LRU cache.
            seq.num_computed_tokens = 0
            seq.num_cached_blocks = 0
            seq.speculative_token_id = None
            self._pending_state_releases.add(seq.seq_id)
        # NOTE: For FCFS, we insert the preempted sequence group to the front
        # of the waiting queue.
        self.waiting.insert(0, seq_group)

    def _preempt_by_swap(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        for seq in seqs:
            seq.status = SequenceStatus.SWAPPED
        self._swap_out(seq_group, blocks_to_swap_out)
        self.swapped.append(seq_group)
    
    def _swap_in(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_in: Dict[int, int],
    ) -> None:
        mapping = self.block_manager.swap_in(seq_group)
        blocks_to_swap_in.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            seq.status = SequenceStatus.RUNNING
        
    def _swap_out(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        if not self.block_manager.can_swap_out(seq_group):
            # FIXME(woosuk): Abort the sequence group instead of aborting the
            # entire engine.
            raise RuntimeError(
                "Aborted due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error.")
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED
