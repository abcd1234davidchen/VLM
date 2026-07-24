"""Shared masking and temporal pooling primitives."""

from __future__ import annotations

import torch


def masked_mean_max_pool(
    output: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    time_steps = torch.arange(
        output.size(1),
        device=lengths.device,
    ).unsqueeze(0)
    mask = time_steps < lengths.unsqueeze(1)
    masked_output = output.masked_fill(
        ~mask.unsqueeze(-1),
        float("-inf"),
    )
    max_pool = masked_output.max(dim=1).values
    mean_pool = output.masked_fill(~mask.unsqueeze(-1), 0.0).sum(dim=1)
    mean_pool = mean_pool / lengths.unsqueeze(1)
    return torch.cat([mean_pool, max_pool], dim=-1)
