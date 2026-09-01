from typing import Dict, Tuple, List, Optional

import torch

from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask

from minivllm.sampling_params import SamplingParams
from minivllm.multimodal import MultiModalInputs
from minivllm.sequence import SequenceData

class InputMetadata:
    """Describes the packed token layout consumed by every model layer.

    Prompt tokens are placed before decode tokens. Within the prompt region,
    prompts without a cache hit come first and cached prompt suffixes follow.
    This lets fresh prompts keep using xFormers while cached suffixes attend to
    their paged KV prefix with the varlen CUDA kernel.
    """

    def __init__(
        self,
        seq_groups: List[Tuple[List[int], SamplingParams]],     # List of (seq_ids, sampling_params).
        seq_data: Dict[int, SequenceData],                      # Seq id -> seq_data
        prompt_lens: List[int],             # Scheduled query length per prompt.
        slot_mapping: torch.Tensor,         # (num_valid_tokens,)
        context_lens: torch.Tensor,         # (num_generation_seqs,)
        max_context_len: int,
        block_tables: torch.Tensor,         # Decode block tables.
        fresh_prompt_lens: Optional[List[int]] = None,
        cached_prompt_query_lens: Optional[List[int]] = None,
        cached_prompt_cu_seqlens: Optional[torch.Tensor] = None,
        cached_prompt_context_lens: Optional[torch.Tensor] = None,
        cached_prompt_block_tables: Optional[torch.Tensor] = None,
        max_cached_prompt_context_len: int = 0,
        prompt_seq_ids: Optional[List[int]] = None,
        generation_seq_ids: Optional[List[int]] = None,
        state_slot_mapping: Optional[torch.Tensor] = None,
        prompt_sample_indices: Optional[List[int]] = None,
        speculative_seq_ids: Optional[List[int]] = None,
        speculative_token_ids: Optional[List[int]] = None,
        speculative_hidden_indices: Optional[List[Tuple[int, int]]] = None,
        enable_mtp: bool = False,
        speculative_sampling_params: Optional[List[SamplingParams]] = None,
        multimodal_inputs: Optional[Dict[int, MultiModalInputs]] = None,
        multimodal_token_maps: Optional[
            List[Tuple[int, int, int, int]]
        ] = None,
    ) -> None:
        self.seq_groups = seq_groups
        self.seq_data = seq_data
        self.prompt_lens = prompt_lens
        self.slot_mapping = slot_mapping
        self.context_lens = context_lens
        self.max_context_len = max_context_len
        self.block_tables = block_tables

        self.fresh_prompt_lens = (
            prompt_lens if fresh_prompt_lens is None else fresh_prompt_lens)
        self.cached_prompt_query_lens = cached_prompt_query_lens or []
        self.cached_prompt_cu_seqlens = cached_prompt_cu_seqlens
        self.cached_prompt_context_lens = cached_prompt_context_lens
        self.cached_prompt_block_tables = cached_prompt_block_tables
        self.max_cached_prompt_context_len = max_cached_prompt_context_len
        self.prompt_seq_ids = prompt_seq_ids or []
        self.generation_seq_ids = generation_seq_ids or []
        self.state_slot_mapping = state_slot_mapping

        # xFormers only sees prompts whose query contains the whole context.
        # Cached suffixes need the separate paged-attention metadata below.
        self.attn_bias = None
        if self.fresh_prompt_lens:
            self.attn_bias = BlockDiagonalCausalMask.from_seqlens(
                self.fresh_prompt_lens)
        self.num_prompts = len(prompt_lens)
        if prompt_sample_indices is None:
            prompt_sample_indices = []
            offset = 0
            for prompt_len in prompt_lens:
                prompt_sample_indices.append(offset + prompt_len - 1)
                offset += prompt_len
        self.prompt_sample_indices = prompt_sample_indices
        self.num_prompt_samples = len(prompt_sample_indices)
        self.speculative_seq_ids = speculative_seq_ids or []
        self.speculative_token_ids = speculative_token_ids or []
        self.speculative_hidden_indices = speculative_hidden_indices or []
        self.enable_mtp = enable_mtp
        self.speculative_sampling_params = speculative_sampling_params or []
        self.multimodal_inputs = multimodal_inputs or {}
        # packed token index, sequence id, modality (1=image/2=video),
        # modality-local feature index.
        self.multimodal_token_maps = multimodal_token_maps or []
        if not (
            len(self.speculative_seq_ids)
            == len(self.speculative_token_ids)
            == len(self.speculative_hidden_indices)
            == len(self.speculative_sampling_params)
        ):
            raise ValueError("Speculative metadata lists must have equal length")
        self.num_prompt_tokens = sum(prompt_lens)
        self.num_fresh_prompt_tokens = sum(self.fresh_prompt_lens)
        self.num_cached_prompt_tokens = sum(self.cached_prompt_query_lens)
        assert (self.num_fresh_prompt_tokens
                + self.num_cached_prompt_tokens == self.num_prompt_tokens)
        self.num_generation_tokens = context_lens.shape[0]
        self.num_valid_tokens = slot_mapping.shape[0]
        if self.prompt_seq_ids:
            assert len(self.prompt_seq_ids) == self.num_prompts
        if self.generation_seq_ids:
            assert len(self.generation_seq_ids) == self.num_generation_tokens
        assert self.num_prompt_samples <= self.num_prompts
        assert len(self.seq_groups) >= self.num_prompt_samples
        if block_tables.numel() > 0:
            # Every row is padded to the same block-table length.
            self.max_num_blocks_per_seq = block_tables.shape[1]
        else:
            self.max_num_blocks_per_seq = 0
        # Check number of decode seq
        assert block_tables.shape[0] == self.num_generation_tokens
        assert context_lens.shape[0] == self.num_generation_tokens

        num_cached_prompts = len(self.cached_prompt_query_lens)
        if num_cached_prompts:
            assert self.cached_prompt_cu_seqlens is not None
            assert self.cached_prompt_context_lens is not None
            assert self.cached_prompt_block_tables is not None
            assert self.cached_prompt_cu_seqlens.shape[0] == num_cached_prompts + 1
            assert self.cached_prompt_context_lens.shape[0] == num_cached_prompts
            assert self.cached_prompt_block_tables.shape[0] == num_cached_prompts
    
    def __repr__(self) -> str:
        # Print only useful metadata.
        return (f'InputMetadata('
                f'num_valid_tokens={self.num_valid_tokens}, '
                f'num_prompt_tokens={self.num_prompt_tokens}, '
                f'num_fresh_prompt_tokens={self.num_fresh_prompt_tokens}, '
                f'num_cached_prompt_tokens={self.num_cached_prompt_tokens}, '
                f'num_prompts={self.num_prompts}, '
                f'num_prompt_samples={self.num_prompt_samples}, '
                f'num_speculative_seqs={len(self.speculative_seq_ids)}, '
                f'prompt_lens={self.prompt_lens}, '
                f'num_generation_tokens={self.num_generation_tokens}, '
                f'prompt_seq_ids={self.prompt_seq_ids}, '
                f'generation_seq_ids={self.generation_seq_ids}, '
                f'context_lens={self.context_lens}, '
                f'max_context_len={self.max_context_len}, '
                f'max_num_blocks_per_seq={self.max_num_blocks_per_seq}, '
                f'block_tables={self.block_tables}, '
                f'slot_mapping={self.slot_mapping})')
