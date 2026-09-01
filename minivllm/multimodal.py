"""Small multimodal request contract shared by the engine and workers."""

from dataclasses import dataclass, replace
from itertools import groupby
from typing import Any, Mapping, Optional, Sequence, Tuple

import torch


Grid = Tuple[Tuple[int, int, int], ...]
PositionIds = Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]


def _cpu_tensor(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.detach().cpu().contiguous()


def _single_row(value: Any, name: str) -> Tuple[int, ...]:
    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.ndim == 2:
        if tensor.shape[0] != 1:
            raise ValueError(f"{name} must describe exactly one request")
        tensor = tensor[0]
    if tensor.ndim != 1:
        raise ValueError(f"{name} must have shape [length] or [1, length]")
    return tuple(int(item) for item in tensor.tolist())


def _grid(value: Optional[torch.Tensor], name: str) -> Grid:
    if value is None:
        return ()
    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.ndim != 2 or tensor.shape[1] != 3:
        raise ValueError(f"{name} must have shape [num_items, 3]")
    grid = tuple(tuple(int(item) for item in row) for row in tensor.tolist())
    if any(min(row) <= 0 for row in grid):
        raise ValueError(f"{name} entries must be positive")
    return grid


@dataclass(frozen=True)
class MultiModalInputs:
    """Processor output for one prompt.

    Pixel tensors remain on CPU while requests wait in the scheduler. Workers
    move them to their local GPU only when the corresponding visual tokens are
    first scheduled.
    """

    token_type_ids: Tuple[int, ...]
    pixel_values: Optional[torch.Tensor] = None
    image_grid_thw: Grid = ()
    pixel_values_videos: Optional[torch.Tensor] = None
    video_grid_thw: Grid = ()
    position_ids: Optional[PositionIds] = None
    rope_delta: int = 0

    @classmethod
    def from_processor_output(
        cls,
        output: Mapping[str, Any],
    ) -> Tuple[Tuple[int, ...], "MultiModalInputs"]:
        """Convert one Hugging Face processor result without keeping padding."""
        if "input_ids" not in output:
            raise ValueError("Processor output is missing input_ids")
        if "mm_token_type_ids" not in output:
            raise ValueError("Processor output is missing mm_token_type_ids")
        prompt_token_ids = _single_row(output["input_ids"], "input_ids")
        token_type_ids = _single_row(
            output["mm_token_type_ids"], "mm_token_type_ids"
        )
        if len(prompt_token_ids) != len(token_type_ids):
            raise ValueError(
                "input_ids and mm_token_type_ids must have equal length"
            )

        pixel_values = _cpu_tensor(output.get("pixel_values"))
        image_grid = _grid(output.get("image_grid_thw"), "image_grid_thw")
        video_values = _cpu_tensor(output.get("pixel_values_videos"))
        video_grid = _grid(output.get("video_grid_thw"), "video_grid_thw")
        if (pixel_values is None) != (not image_grid):
            raise ValueError(
                "pixel_values and image_grid_thw must be provided together"
            )
        if (video_values is None) != (not video_grid):
            raise ValueError(
                "pixel_values_videos and video_grid_thw must be provided together"
            )
        if pixel_values is None and video_values is None:
            raise ValueError("Processor output contains no image or video data")

        return prompt_token_ids, cls(
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid,
            pixel_values_videos=video_values,
            video_grid_thw=video_grid,
        )

    def with_positions(
        self,
        prompt_token_ids: Sequence[int],
        spatial_merge_size: int,
    ) -> "MultiModalInputs":
        """Compute the prompt's temporal/height/width M-RoPE positions."""
        if spatial_merge_size <= 0:
            raise ValueError("spatial_merge_size must be positive")
        if len(prompt_token_ids) != len(self.token_type_ids):
            raise ValueError(
                "Prompt tokens and multimodal token types must have equal length"
            )
        if any(token_type not in (0, 1, 2) for token_type in self.token_type_ids):
            raise ValueError("Multimodal token types must be text=0/image=1/video=2")

        image_grids = iter(self.image_grid_thw)
        # Qwen inserts timestamps between video frames. Each contiguous video
        # token run therefore consumes one temporal slice of the source grid.
        video_frames = iter(
            (1, height, width)
            for frames, height, width in self.video_grid_thw
            for _ in range(frames)
        )
        grid_iters = {1: image_grids, 2: video_frames}
        rows = [[], [], []]
        current_position = 0

        for token_type, token_group in groupby(self.token_type_ids):
            group_length = sum(1 for _ in token_group)
            if token_type == 0:
                values = list(
                    range(current_position, current_position + group_length)
                )
                for row in rows:
                    row.extend(values)
                current_position += group_length
                continue

            try:
                frames, height, width = next(grid_iters[token_type])
            except StopIteration as exc:
                modality = "image" if token_type == 1 else "video frame"
                raise ValueError(
                    f"More {modality} token groups than processor grids"
                ) from exc
            if height % spatial_merge_size or width % spatial_merge_size:
                raise ValueError(
                    "Vision grid height and width must be divisible by "
                    "spatial_merge_size"
                )
            merged_height = height // spatial_merge_size
            merged_width = width // spatial_merge_size
            expected_length = frames * merged_height * merged_width
            if group_length != expected_length:
                raise ValueError(
                    f"Multimodal token run has {group_length} tokens, but "
                    f"grid {(frames, height, width)} requires {expected_length}"
                )

            temporal = torch.arange(frames).repeat_interleave(
                merged_height * merged_width
            )
            heights = torch.arange(merged_height).repeat_interleave(
                merged_width
            ).repeat(frames)
            widths = torch.arange(merged_width).repeat(
                frames * merged_height
            )
            for row, values in zip(rows, (temporal, heights, widths)):
                row.extend((values + current_position).tolist())
            current_position += max(merged_height, merged_width)

        try:
            next(image_grids)
            raise ValueError("Fewer image token groups than processor grids")
        except StopIteration:
            pass
        try:
            next(video_frames)
            raise ValueError("Fewer video token groups than processor grids")
        except StopIteration:
            pass

        position_ids: PositionIds = tuple(
            tuple(int(value) for value in row) for row in rows
        )  # type: ignore[assignment]
        max_position = max(max(row) for row in position_ids)
        return replace(
            self,
            position_ids=position_ids,
            rope_delta=max_position + 1 - len(prompt_token_ids),
        )

    def positions_only(self) -> "MultiModalInputs":
        """Drop large processor tensors after workers cache visual features."""
        if self.position_ids is None:
            raise ValueError("Multimodal positions have not been prepared")
        return replace(
            self,
            pixel_values=None,
            image_grid_thw=(),
            pixel_values_videos=None,
            video_grid_thw=(),
        )


__all__ = ["MultiModalInputs"]
