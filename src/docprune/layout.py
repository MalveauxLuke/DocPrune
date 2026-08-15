"""Spatial bookkeeping for Qwen2-VL merge-safe visual token pruning."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class VisualLayout:
    """Original fine-patch grids and their complete spatial-merge groups."""

    grid_thw: torch.Tensor
    spatial_merge_size: int = 2

    def __post_init__(self) -> None:
        grid = torch.as_tensor(self.grid_thw, dtype=torch.long)
        if grid.ndim != 2 or grid.shape[1] != 3:
            raise ValueError("grid_thw must have shape [pages, 3]")
        if grid.numel() == 0 or torch.any(grid <= 0):
            raise ValueError("grid_thw values must be positive")
        if self.spatial_merge_size <= 0:
            raise ValueError("spatial_merge_size must be positive")
        if torch.any(grid[:, 1:] % self.spatial_merge_size != 0):
            raise ValueError("grid height and width must be divisible by spatial_merge_size")
        object.__setattr__(self, "grid_thw", grid)

    @property
    def fine_token_counts(self) -> tuple[int, ...]:
        return tuple(int(t * h * w) for t, h, w in self.grid_thw.tolist())

    @property
    def group_counts(self) -> tuple[int, ...]:
        merge_area = self.spatial_merge_size**2
        return tuple(count // merge_area for count in self.fine_token_counts)

    @staticmethod
    def _offsets(counts: tuple[int, ...]) -> tuple[int, ...]:
        values = [0]
        for count in counts:
            values.append(values[-1] + count)
        return tuple(values)

    @property
    def page_fine_offsets(self) -> tuple[int, ...]:
        return self._offsets(self.fine_token_counts)

    @property
    def page_group_offsets(self) -> tuple[int, ...]:
        return self._offsets(self.group_counts)

    @property
    def total_fine_tokens(self) -> int:
        return self.page_fine_offsets[-1]

    @property
    def total_groups(self) -> int:
        return self.page_group_offsets[-1]

    def group_fine_indices(self) -> torch.Tensor:
        groups: list[list[int]] = []
        merge = self.spatial_merge_size
        page_offset = 0
        for temporal, height, width in self.grid_thw.tolist():
            for time_index in range(temporal):
                time_offset = page_offset + time_index * height * width
                for row in range(0, height, merge):
                    for column in range(0, width, merge):
                        groups.append(
                            [
                                time_offset + (row + dy) * width + column + dx
                                for dy in range(merge)
                                for dx in range(merge)
                            ]
                        )
            page_offset += temporal * height * width
        return torch.tensor(groups, dtype=torch.long, device=self.grid_thw.device)

    def expand_group_mask(self, group_mask: torch.Tensor) -> torch.Tensor:
        group_mask = torch.as_tensor(group_mask, dtype=torch.bool)
        if group_mask.ndim != 1 or group_mask.numel() != self.total_groups:
            raise ValueError(f"group_mask must contain {self.total_groups} values")
        indices = self.group_fine_indices().to(group_mask.device)
        fine_mask = torch.zeros(self.total_fine_tokens, dtype=torch.bool, device=group_mask.device)
        fine_mask[indices[group_mask].flatten()] = True
        return fine_mask

    def flatten_page_values(self, values: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(values)
        if values.ndim == 1:
            if values.numel() != self.total_fine_tokens:
                raise ValueError(f"values must contain {self.total_fine_tokens} fine tokens")
            return values
        if values.ndim != 2 or values.shape[0] != len(self.fine_token_counts):
            raise ValueError("values must be flat or have one row per page")
        if len(set(self.fine_token_counts)) != 1 or values.shape[1] != self.fine_token_counts[0]:
            raise ValueError("rectangular page values require equal fine-token counts")
        return values.flatten()
