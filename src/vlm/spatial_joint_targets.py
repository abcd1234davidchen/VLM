"""Training-only NTU color-joint targets on the XCLIP patch grid."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from vlm.phase4_alignment import color_joints, transform_point


@dataclass(frozen=True)
class SpatialTargets:
    occupancy: Tensor
    frame_mask: Tensor
    joint_validity: Tensor
    body_ids: tuple[int, ...]
    tracked_in_crop: tuple[int, ...]


def build_spatial_targets(
    skeleton_path: Path,
    frame_indices: list[int],
    *,
    source_width: int,
    source_height: int,
    input_size: int = 224,
    patch_size: int = 32,
    min_tracked_joints: int = 10,
) -> SpatialTargets:
    if not frame_indices or len(set(frame_indices)) != len(frame_indices):
        raise ValueError("Expected distinct sampled frame indices")
    if source_width < 1 or source_height < 1 or input_size != 224 or input_size % patch_size:
        raise ValueError("Unsupported source or patch geometry")
    frame_count, selected = color_joints(skeleton_path, set(frame_indices))
    if max(frame_indices) >= frame_count:
        raise ValueError("Sampled RGB frame index is outside the skeleton sequence")
    tracked_by_body: Counter[int] = Counter()
    for bodies in selected.values():
        for body in bodies:
            tracked_by_body[body["body_id"]] += sum(joint["state"] == 2 for joint in body["joints"])
    body_ids = tuple(sorted(tracked_by_body, key=lambda body_id: (-tracked_by_body[body_id], body_id))[:2])
    grid = input_size // patch_size
    occupancy = torch.zeros(len(frame_indices), grid * grid, dtype=torch.float32)
    frame_mask = torch.zeros(len(frame_indices), dtype=torch.bool)
    joint_validity = torch.zeros(len(frame_indices), 2, 25, dtype=torch.bool)
    tracked_counts = []
    for frame_slot, frame_index in enumerate(frame_indices):
        bodies = {body["body_id"]: body for body in selected[frame_index]}
        tracked = 0
        for body_slot, body_id in enumerate(body_ids):
            body = bodies.get(body_id)
            if body is None:
                continue
            for joint in body["joints"]:
                if joint["state"] != 2:
                    continue
                x, y = transform_point(joint["x"], joint["y"], source_width, source_height)
                if not (0 <= x < input_size and 0 <= y < input_size):
                    continue
                joint_validity[frame_slot, body_slot, joint["joint"]] = True
                occupancy[frame_slot, int(y // patch_size) * grid + int(x // patch_size)] = 1
                tracked += 1
        tracked_counts.append(tracked)
        frame_mask[frame_slot] = tracked >= min_tracked_joints
    occupancy[~frame_mask] = 0
    joint_validity[~frame_mask] = False
    return SpatialTargets(occupancy, frame_mask, joint_validity, body_ids, tuple(tracked_counts))


def spatial_region_loss(logits: Tensor, occupancy: Tensor, frame_mask: Tensor) -> Tensor:
    """Balanced patch BCE averaged only over supervised frames."""
    if logits.ndim != 3 or occupancy.shape != logits.shape or frame_mask.shape != logits.shape[:2]:
        raise ValueError("Expected logits/targets [batch, frames, patches] and frame mask [batch, frames]")
    if not torch.isfinite(logits).all() or not torch.isfinite(occupancy).all():
        raise ValueError("Spatial logits and targets must be finite")
    if bool(((occupancy != 0) & (occupancy != 1)).any()):
        raise ValueError("Spatial occupancy must be binary")
    if bool((frame_mask & (occupancy.sum(-1) == 0)).any()):
        raise ValueError("Supervised frames must contain occupied patches")
    if not bool(frame_mask.any()):
        return logits.sum() * 0
    positives = occupancy.sum(-1).clamp_min(1)
    positive_weight = ((logits.shape[-1] - positives) / positives).clamp(1, 16)
    weights = torch.where(occupancy > 0, positive_weight.unsqueeze(-1), 1.0)
    per_patch = F.binary_cross_entropy_with_logits(logits.float(), occupancy.float(), reduction="none")
    return (per_patch * weights).mean(-1)[frame_mask].mean()
