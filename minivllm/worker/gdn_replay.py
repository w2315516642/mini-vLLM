"""Rebuild accepted GDN prefixes without executing the target model again."""

from dataclasses import dataclass

import torch
from minivllm.profiling import nvtx_function
from minivllm.model_executor.layers.gated_delta_net import GatedDeltaNetState

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

        Fully accepted rows already hold the correct verification state. Only
        partially accepted rows are rebuilt, in snapshot order. Return the
        number of rows actually replayed for stage profiling.
        """
        counts = [committed_tokens[seq_id] for seq_id in self.seq_ids]
        if any(not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= limit
               for n, limit in zip(counts, self.query_lens)):
            raise ValueError("Committed GDN lengths must include the anchor and fit the verified prefix")
        if self.layers.keys() != self.snapshot.layer_states.keys():
            raise RuntimeError("Verification did not record every GDN layer")
        rows = [i for i, (count, limit) in enumerate(zip(counts, self.query_lens))
                if count < limit]
        if not rows:
            return 0
        device = self.cu_seqlens.device
        row_indices = token_indices = None
        cu_seqlens = self.cu_seqlens
        if len(rows) != len(self.seq_ids):
            # Offsets refer to record()'s snapshot-ordered packed inputs, not
            # the original mixed batch. Build metadata once for all layers.
            starts, offset = [], 0
            for length in self.query_lens:
                starts.append(offset)
                offset += length
            tokens, offsets = [], [0]
            for i in rows:
                tokens.extend(range(starts[i], starts[i] + counts[i]))
                offsets.append(offsets[-1] + counts[i])
            row_indices = torch.tensor(rows, dtype=torch.long, device=device)
            token_indices = torch.tensor(tokens, dtype=torch.long, device=device)
            cu_seqlens = torch.tensor(offsets, dtype=torch.int32, device=device)
        counts = [counts[i] for i in rows]
        lengths = torch.tensor(counts, dtype=torch.int32, device=device)
        slots = cache.acquire([self.seq_ids[i] for i in rows])
        for layer_idx, inputs in self.layers.items():
            state = self.snapshot.layer_states[layer_idx]
            if row_indices is not None:
                state = GatedDeltaNetState(
                    state.conv_state.index_select(0, row_indices),
                    state.recurrent_state.index_select(0, row_indices),
                )
                inputs = _LayerReplayInputs(
                    inputs.projected_qkv.index_select(0, token_indices), inputs.conv_weight,
                    inputs.key.index_select(0, token_indices),
                    inputs.value.index_select(0, token_indices),
                    inputs.log_decay.index_select(0, token_indices),
                    inputs.beta.index_select(0, token_indices),
                )
            causal_conv1d_varlen(
                inputs.projected_qkv, state.conv_state, inputs.conv_weight.contiguous(),
                cu_seqlens, lengths,
            )
            # Query affects only the unused output, never the state update.
            # Reuse K here rather than retaining another per-token Q buffer.
            gated_delta_rule_varlen(
                inputs.key, inputs.key, inputs.value, inputs.log_decay, inputs.beta,
                state.recurrent_state, cu_seqlens, max(counts), lengths,
            )
            cache._write_state(layer_idx, slots, state)
        return len(rows)


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
