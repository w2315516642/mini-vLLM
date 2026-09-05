"""Rebuild accepted GDN prefixes without executing the target model again."""

from dataclasses import dataclass

import torch
from minivllm.profiling import nvtx_function

from minivllm.model_executor.layers.gated_delta_net_cuda import (
    causal_conv1d_varlen,
    gated_delta_rule_varlen,
)


@dataclass
class _LayerReplayInputs:
    projected_qkv: torch.Tensor
    conv_weight: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    log_decay: torch.Tensor
    beta: torch.Tensor


class GatedDeltaNetReplay:
    """One verification transaction; its snapshot is consumed by commit().

    The accepted prefix has exactly the same causal layer inputs as it did
    during verification. Save those inputs, not a full recurrent state per
    token. Replay updates only Conv/GDN states, with no projections or MLPs.
    """

    def __init__(self, metadata, snapshot):
        self.snapshot = snapshot
        self.seq_ids = snapshot.seq_ids
        spans = {}
        offset = 0
        for seq_id, length in zip(metadata.prompt_seq_ids, metadata.prompt_lens):
            spans[seq_id] = (offset, length)
            offset += length
        indices, offsets, self.query_lens = [], [0], []
        for seq_id in self.seq_ids:
            start, length = spans[seq_id]
            indices.extend(range(start, start + length))
            offsets.append(offsets[-1] + length)
            self.query_lens.append(length)
        device = metadata.slot_mapping.device
        self.token_indices = torch.tensor(indices, dtype=torch.long, device=device)
        self.cu_seqlens = torch.tensor(offsets, dtype=torch.int32, device=device)
        self.layers = {}

    @nvtx_function("replay_record")
    def record(self, layer_idx, qkv, weight, key, value, log_decay, beta):
        """Keep only speculative tokens; unrelated prefill/decode is excluded."""
        self.layers[layer_idx] = _LayerReplayInputs(
            qkv.index_select(0, self.token_indices), weight,
            key.index_select(0, self.token_indices),
            value.index_select(0, self.token_indices),
            log_decay.index_select(0, self.token_indices),
            beta.index_select(0, self.token_indices),
        )

    @nvtx_function("state_replay")
    def commit(self, cache, committed_tokens):
        """Consume the snapshot and install states at each accepted boundary.

        All speculative rows are reconstructed together, including fully
        accepted rows when another row rejected. Ordinary requests are never
        touched. The worker skips this entirely when every draft is accepted.
        """
        counts = [committed_tokens[seq_id] for seq_id in self.seq_ids]
        if any(not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= limit
               for n, limit in zip(counts, self.query_lens)):
            raise ValueError("Committed GDN lengths must include the anchor and fit the verified prefix")
        if self.layers.keys() != self.snapshot.layer_states.keys():
            raise RuntimeError("Verification did not record every GDN layer")
        lengths = torch.tensor(counts, dtype=torch.int32, device=self.cu_seqlens.device)
        slots = cache.acquire(self.seq_ids)
        for layer_idx, inputs in self.layers.items():
            state = self.snapshot.layer_states[layer_idx]
            causal_conv1d_varlen(
                inputs.projected_qkv, state.conv_state, inputs.conv_weight.contiguous(),
                self.cu_seqlens, lengths,
            )
            # Query affects only the unused output, never the state update.
            # Reuse K here rather than retaining another per-token Q buffer.
            gated_delta_rule_varlen(
                inputs.key, inputs.key, inputs.value, inputs.log_decay, inputs.beta,
                state.recurrent_state, self.cu_seqlens, max(counts), lengths,
            )
            cache._write_state(layer_idx, slots, state)


def replay_buffer_bytes(spec, num_layers, max_num_seqs, max_tokens, dtype):
    """Snapshot plus packed replay inputs, excluding shared model weights."""
    state_elements = (spec.conv_dim * spec.conv_kernel_size
                      + spec.num_value_heads * spec.key_head_dim * spec.value_head_dim)
    token_elements = (spec.conv_dim + spec.num_value_heads
                      * (spec.key_head_dim + spec.value_head_dim + 1))
    dtype_bytes = torch.empty((), dtype=dtype).element_size()
    return num_layers * (
        max_num_seqs * state_elements * 4
        + max_tokens * (token_elements * dtype_bytes + spec.num_value_heads * 4)
    )
