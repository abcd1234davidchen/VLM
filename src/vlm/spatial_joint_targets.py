"""Training-only NTU color-joint targets on the XCLIP patch grid."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import math

import torch
from torch import Tensor
from torch.nn import functional as F

from vlm.phase4_alignment import color_joints, transform_point


MOTION_TARGET_VERSION = "root-scale-motion-v1"
# NTU zero-based joint IDs: head, torso, left/right arms, left/right legs.
BODY_PART_JOINTS = ((2, 3), (0, 1, 20), (4, 5, 6, 7, 21, 22),
                    (8, 9, 10, 11, 23, 24), (12, 13, 14, 15), (16, 17, 18, 19))


@dataclass(frozen=True)
class MotionSpatialTargets:
    distribution: Tensor
    frame_mask: Tensor
    joint_validity: Tensor
    occupancy: Tensor
    body_part_mass: Tensor
    body_ids: tuple[int, ...]
    version: str = MOTION_TARGET_VERSION


def build_motion_spatial_targets(
    skeleton_path: Path,
    frame_indices: list[int],
    *,
    source_width: int,
    source_height: int,
    input_size: int = 224,
    patch_size: int = 32,
    min_tracked_joints: int = 10,
) -> MotionSpatialTargets:
    """Map adjacent-frame root-relative, body-scaled XYZ motion to color patches.

    Static or missing-motion frames are unsupervised, never replaced by a person
    prior. Each neighbor must track the same body, joint, root and spine shoulder.
    Body-part mass retains two body slots and sums to the main distribution.
    """
    if (not frame_indices or len(set(frame_indices)) != len(frame_indices)
            or min(frame_indices) < 0):
        raise ValueError("Expected distinct nonnegative sampled frame indices")
    if (source_width < 1 or source_height < 1 or input_size != 224
            or patch_size < 1 or input_size % patch_size or min_tracked_joints < 1):
        raise ValueError("Unsupported source or patch geometry")
    wanted = {neighbor for frame in frame_indices for neighbor in (frame - 1, frame, frame + 1) if neighbor >= 0}
    count, selected = color_joints(skeleton_path, wanted)
    if max(frame_indices) >= count:
        raise ValueError("Sampled RGB frame index is outside the skeleton sequence")
    tracked: Counter[int] = Counter()
    for frame in frame_indices:
        for body in selected[frame]:
            tracked[body["body_id"]] += sum(joint["state"] == 2 for joint in body["joints"])
    body_ids = tuple(sorted(tracked, key=lambda body: (-tracked[body], body))[:2])
    grid = input_size // patch_size
    distribution = torch.zeros(len(frame_indices), grid * grid)
    occupancy = torch.zeros_like(distribution)
    validity = torch.zeros(len(frame_indices), 2, 25, dtype=torch.bool)
    parts = torch.zeros(len(frame_indices), 2, 6, grid * grid)
    mask = torch.zeros(len(frame_indices), dtype=torch.bool)
    lookup = {frame: {body["body_id"]: body for body in bodies} for frame, bodies in selected.items()}

    def normalized(body: dict | None) -> tuple[Tensor, Tensor] | None:
        if body is None:
            return None
        joints = body["joints"]
        xyz = torch.tensor([joint["xyz"] for joint in joints], dtype=torch.float64)
        valid = torch.tensor([joint["state"] == 2 for joint in joints]) & torch.isfinite(xyz).all(-1)
        if not bool(valid[0] & valid[20]):
            return None
        scale = (xyz[20] - xyz[0]).norm()
        if not torch.isfinite(scale) or scale <= 1e-6:
            return None
        return (xyz - xyz[0]) / scale, valid

    part_by_joint = {joint: part for part, joints in enumerate(BODY_PART_JOINTS) for joint in joints}
    for slot, frame in enumerate(frame_indices):
        for body_slot, body_id in enumerate(body_ids):
            body = lookup[frame].get(body_id)
            current = normalized(body)
            if current is None:
                continue
            xyz, valid = current
            speeds = torch.zeros(25, dtype=torch.float64)
            observations = torch.zeros(25, dtype=torch.float64)
            for neighbor in (frame - 1, frame + 1):
                adjacent = normalized(lookup.get(neighbor, {}).get(body_id))
                if adjacent is None:
                    continue
                adjacent_xyz, adjacent_valid = adjacent
                usable = valid & adjacent_valid
                displacement = (adjacent_xyz - xyz).norm(dim=-1)
                speeds[usable] += displacement[usable]
                observations[usable] += 1
            speeds /= observations.clamp_min(1)
            for joint in body["joints"]:
                index = joint["joint"]
                x, y = transform_point(joint["x"], joint["y"], source_width, source_height)
                if not (bool(valid[index]) and math.isfinite(x) and math.isfinite(y)
                        and 0 <= x < input_size and 0 <= y < input_size):
                    continue
                patch = int(y // patch_size) * grid + int(x // patch_size)
                occupancy[slot, patch] = 1
                if observations[index] == 0:
                    continue
                validity[slot, body_slot, index] = True
                mass = float(speeds[index])
                parts[slot, body_slot, part_by_joint[index], patch] += mass
        mass = parts[slot].sum()
        if int(validity[slot].sum()) >= min_tracked_joints and mass > 1e-6:
            mask[slot] = True
            parts[slot] /= mass
            distribution[slot] = parts[slot].sum((0, 1))
        else:
            parts[slot] = 0
            validity[slot] = False
    return MotionSpatialTargets(distribution, mask, validity, occupancy, parts, body_ids)


def spatial_attention_loss(logits: Tensor, distribution: Tensor, frame_mask: Tensor) -> Tensor:
    """Soft target cross-entropy over valid motion frames, including CE-only batches."""
    if logits.ndim != 3 or distribution.shape != logits.shape or frame_mask.shape != logits.shape[:2]:
        raise ValueError("Expected logits/targets [batch, frames, patches] and frame mask [batch, frames]")
    if frame_mask.dtype != torch.bool:
        raise ValueError("Frame mask must be boolean")
    if not torch.isfinite(logits).all() or not torch.isfinite(distribution).all() or (distribution < 0).any():
        raise ValueError("Attention logits and nonnegative targets must be finite")
    if not torch.allclose(distribution.sum(-1), frame_mask.to(distribution.dtype), atol=1e-5, rtol=0):
        raise ValueError("Targets must sum to one on valid frames and zero on invalid frames")
    if not bool(frame_mask.any()):
        return logits.sum() * 0
    return -(distribution.float() * logits.float().log_softmax(-1)).sum(-1)[frame_mask].mean()


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
