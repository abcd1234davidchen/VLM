"""Training-only pose targets and loss for native XCLIP CLS attention."""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

TARGET_VERSION = "native-occupancy-gaussian05-uniform20-v1"


def native_pose_distribution(occupancy: Tensor, frame_mask: Tensor) -> Tensor:
    if occupancy.ndim != 3 or occupancy.shape[-1] != 49 or frame_mask.shape != occupancy.shape[:2] or frame_mask.dtype != torch.bool:
        raise ValueError("Expected occupancy [batch, frames, 49] and boolean frame mask")
    if not torch.isfinite(occupancy).all() or ((occupancy != 0) & (occupancy != 1)).any():
        raise ValueError("Occupancy must be finite and binary")
    if (frame_mask & (occupancy.sum(-1) == 0)).any():
        raise ValueError("A valid frame must contain tracked joints")
    axis = torch.arange(-1, 2, device=occupancy.device, dtype=torch.float32)
    kernel = torch.exp(-axis.square() / (2 * .5**2))
    kernel = kernel[:, None] * kernel[None, :]
    kernel /= kernel.sum()
    smooth = F.conv2d(occupancy.float().reshape(-1, 1, 7, 7), kernel[None, None], padding=1).reshape_as(occupancy)
    target = .8 * smooth / smooth.sum(-1, keepdim=True).clamp_min(1e-8) + .2 / 49
    return target * frame_mask.unsqueeze(-1)


def native_patch_distribution(attention: Tensor) -> Tensor:
    if attention.ndim != 4 or attention.shape[-2:] != (2, 51):
        raise ValueError("Expected CLS attention [batch, frames, two layers, 51 tokens]")
    if not torch.isfinite(attention).all() or (attention < 0).any():
        raise ValueError("Native attention must be finite and nonnegative")
    patches = attention[..., 1:50].float()
    return patches / patches.sum(-1, keepdim=True).clamp_min(1e-8)


def native_attention_loss(attention: Tensor, target: Tensor, frame_mask: Tensor) -> Tensor:
    prediction = native_patch_distribution(attention)
    if target.shape != attention.shape[:2] + (49,) or frame_mask.shape != target.shape[:2] or frame_mask.dtype != torch.bool:
        raise ValueError("Attention/target/frame mask shapes differ")
    if not torch.isfinite(target).all() or (target < 0).any() or not torch.allclose(target.sum(-1), frame_mask.float(), atol=1e-5, rtol=0):
        raise ValueError("Targets must sum to one on valid frames and zero otherwise")
    if not bool(frame_mask.any()):
        return attention.sum() * 0
    desired = target.float().unsqueeze(-2).expand_as(prediction)
    kl = (desired * (desired.clamp_min(1e-8).log() - prediction.clamp_min(1e-8).log())).sum(-1)
    return kl[frame_mask].mean()


def native_attention_metrics(attention: Tensor, occupancy: Tensor, frame_mask: Tensor) -> dict[str, float | int]:
    target = native_pose_distribution(occupancy, frame_mask)
    prediction = native_patch_distribution(attention)
    count = int(frame_mask.sum())
    if not count:
        return {"valid_frames": 0}
    selected = attention.float()[frame_mask]
    patch = prediction[frame_mask]
    occupied = occupancy[frame_mask].unsqueeze(1)
    return {
        "valid_frames": count,
        "target_kl": float(native_attention_loss(attention, target, frame_mask)),
        "joint_region_mass": float((selected[..., 1:50] * occupied).sum(-1).mean()),
        "conditional_joint_region_mass": float((patch * occupied).sum(-1).mean()),
        "patch_mass": float(selected[..., 1:50].sum(-1).mean()),
        "cls_mass": float(selected[..., 0].mean()),
        "message_mass": float(selected[..., 50].mean()),
        "patch_entropy": float(-(patch * patch.clamp_min(1e-8).log()).sum(-1).mean()),
    }
