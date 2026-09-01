"""Qwen3.5 vision tower shared by image and video inputs."""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class Qwen3_5VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10_000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, sequence_length: int) -> torch.Tensor:
        positions = torch.arange(
            sequence_length,
            dtype=self.inv_freq.dtype,
            device=self.inv_freq.device,
        )
        return torch.outer(positions, self.inv_freq)


class Qwen3_5VisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.in_channels = config.in_channels
        self.temporal_patch_size = config.temporal_patch_size
        self.patch_size = config.patch_size
        kernel_size = (
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        self.proj = nn.Conv3d(
            self.in_channels,
            config.hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        expected_width = (
            self.in_channels
            * self.temporal_patch_size
            * self.patch_size
            * self.patch_size
        )
        if pixel_values.ndim != 2 or pixel_values.shape[1] != expected_width:
            raise ValueError(
                "Qwen pixel_values must have shape "
                f"[num_patches, {expected_width}]"
            )
        patches = pixel_values.reshape(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        return self.proj(patches.to(self.proj.weight.dtype)).flatten(1)


class Qwen3_5VisionMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=True
        )
        self.linear_fc2 = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=True
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.gelu(self.linear_fc1(hidden_states), approximate="tanh")
        return self.linear_fc2(hidden_states)


class Qwen3_5VisionAttention(nn.Module):
    """Non-causal packed attention; each video frame is one attention item."""

    def __init__(self, config) -> None:
        super().__init__()
        if config.hidden_size % config.num_heads:
            raise ValueError("Vision hidden_size must be divisible by num_heads")
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = nn.Linear(
            config.hidden_size, 3 * config.hidden_size, bias=True
        )
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        sequence_length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states).reshape(
            sequence_length, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=1)
        cos = cos.unsqueeze(1).to(torch.float32)
        sin = sin.unsqueeze(1).to(torch.float32)
        query_float = query.float()
        key_float = key.float()
        query = (
            query_float * cos + _rotate_half(query_float) * sin
        ).to(query.dtype)
        key = (
            key_float * cos + _rotate_half(key_float) * sin
        ).to(key.dtype)

        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        outputs: List[torch.Tensor] = []
        offset = 0
        for length in lengths:
            end = offset + length
            q = query[offset:end].transpose(0, 1).unsqueeze(0)
            k = key[offset:end].transpose(0, 1).unsqueeze(0)
            v = value[offset:end].transpose(0, 1).unsqueeze(0)
            output = F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=False
            )
            outputs.append(output.squeeze(0).transpose(0, 1))
            offset = end
        if offset != sequence_length:
            raise ValueError("Vision cu_seqlens do not cover all patch tokens")
        return self.proj(torch.cat(outputs).reshape(sequence_length, -1))


class Qwen3_5VisionBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3_5VisionAttention(config)
        self.mlp = Qwen3_5VisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens, cos, sin
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class Qwen3_5VisionPatchMerger(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        merged_size = config.hidden_size * config.spatial_merge_size ** 2
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(merged_size, merged_size, bias=True)
        self.linear_fc2 = nn.Linear(
            merged_size, config.out_hidden_size, bias=True
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states).reshape(
            -1, self.linear_fc1.in_features
        )
        hidden_states = F.gelu(self.linear_fc1(hidden_states))
        return self.linear_fc2(hidden_states)


class Qwen3_5VisionModel(nn.Module):
    """Encode processor patches and merge each spatial 2x2 group."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = Qwen3_5VisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(
            config.num_position_embeddings, config.hidden_size
        )
        self.num_grid_per_side = int(config.num_position_embeddings ** 0.5)
        if self.num_grid_per_side ** 2 != config.num_position_embeddings:
            raise ValueError("Vision position embedding count must be square")
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(
            Qwen3_5VisionBlock(config) for _ in range(config.depth)
        )
        self.merger = Qwen3_5VisionPatchMerger(config)

    def _rotary_embeddings(
        self,
        grid_thw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        merge_size = self.spatial_merge_size
        grids = grid_thw.tolist()
        max_side = max(max(height, width) for _, height, width in grids)
        frequency_table = self.rotary_pos_emb(max_side)
        coordinates = []
        for frames, height, width in grids:
            merged_height = height // merge_size
            merged_width = width // merge_size
            block_rows = torch.arange(merged_height, device=grid_thw.device)
            block_cols = torch.arange(merged_width, device=grid_thw.device)
            inner_rows = torch.arange(merge_size, device=grid_thw.device)
            inner_cols = torch.arange(merge_size, device=grid_thw.device)
            rows = (
                block_rows[:, None, None, None] * merge_size
                + inner_rows[None, None, :, None]
            ).expand(
                merged_height, merged_width, merge_size, merge_size
            ).reshape(-1)
            cols = (
                block_cols[None, :, None, None] * merge_size
                + inner_cols[None, None, None, :]
            ).expand(
                merged_height, merged_width, merge_size, merge_size
            ).reshape(-1)
            item_coordinates = torch.stack((rows, cols), dim=-1)
            coordinates.append(item_coordinates.repeat(frames, 1))
        rotary = frequency_table[torch.cat(coordinates)].flatten(1)
        rotary = torch.cat((rotary, rotary), dim=-1)
        return rotary.cos(), rotary.sin()

    def _absolute_position_embeddings(
        self,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = []
        side = self.num_grid_per_side
        merge_size = self.spatial_merge_size
        for frames, height, width in grid_thw.tolist():
            row = torch.linspace(0, side - 1, height, device=grid_thw.device)
            col = torch.linspace(0, side - 1, width, device=grid_thw.device)
            row_floor, col_floor = row.long(), col.long()
            row_ceil = (row_floor + 1).clamp(max=side - 1)
            col_ceil = (col_floor + 1).clamp(max=side - 1)
            row_weight = row - row_floor
            col_weight = col - col_floor
            indices = torch.stack((
                (row_floor[:, None] * side + col_floor[None, :]).flatten(),
                (row_floor[:, None] * side + col_ceil[None, :]).flatten(),
                (row_ceil[:, None] * side + col_floor[None, :]).flatten(),
                (row_ceil[:, None] * side + col_ceil[None, :]).flatten(),
            ))
            weights = torch.stack((
                ((1 - row_weight)[:, None] * (1 - col_weight)[None, :]).flatten(),
                ((1 - row_weight)[:, None] * col_weight[None, :]).flatten(),
                (row_weight[:, None] * (1 - col_weight)[None, :]).flatten(),
                (row_weight[:, None] * col_weight[None, :]).flatten(),
            )).to(self.pos_embed.weight.dtype)
            item = (self.pos_embed(indices) * weights[..., None]).sum(0)
            item = item.repeat(frames, 1).reshape(
                frames,
                height // merge_size,
                merge_size,
                width // merge_size,
                merge_size,
                -1,
            ).permute(0, 1, 3, 2, 4, 5).flatten(0, 4)
            embeddings.append(item)
        return torch.cat(embeddings)

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        if grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
            raise ValueError("grid_thw must have shape [num_items, 3]")
        expected_patches = int(grid_thw.prod(dim=-1).sum().item())
        if pixel_values.shape[0] != expected_patches:
            raise ValueError(
                f"Vision grid requires {expected_patches} patches, got "
                f"{pixel_values.shape[0]}"
            )
        hidden_states = self.patch_embed(pixel_values)
        hidden_states = hidden_states + self._absolute_position_embeddings(
            grid_thw
        ).to(hidden_states.dtype)
        cos, sin = self._rotary_embeddings(grid_thw)
        frame_lengths = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        )
        cu_seqlens = F.pad(frame_lengths.cumsum(0), (1, 0), value=0)
        for block in self.blocks:
            hidden_states = block(
                hidden_states, cu_seqlens, cos, sin
            )
        return self.merger(hidden_states)


__all__ = ["Qwen3_5VisionModel"]
