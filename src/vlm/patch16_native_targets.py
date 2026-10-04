"""Patch16 native attention distributions, diagnostics and derived cache."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from vlm.patch16_skeleton_targets import TargetCache, atomic_save
from vlm.skeleton_teacher_cache import sha256_file

TARGET_VERSION = 'native-p16-occupancy-gaussian1-uniform20-v1'


def native_pose_distribution(occupancy, frame_mask):
    if occupancy.ndim != 3 or occupancy.shape[-1] != 196 or frame_mask.shape != occupancy.shape[:2] or frame_mask.dtype != torch.bool:
        raise ValueError('Expected binary occupancy [batch, frames, 196] and boolean mask')
    if not torch.isfinite(occupancy).all() or ((occupancy != 0) & (occupancy != 1)).any():
        raise ValueError('Occupancy must be finite and binary')
    if (frame_mask & (occupancy.sum(-1) == 0)).any():
        raise ValueError('Valid frames must contain joints')
    axis = torch.arange(-2, 3, device=occupancy.device, dtype=torch.float32)
    kernel = torch.exp(-axis.square() / 2)
    kernel = kernel[:, None] * kernel[None, :]
    kernel /= kernel.sum()
    smooth = F.conv2d(occupancy.float().reshape(-1, 1, 14, 14), kernel[None, None], padding=2).reshape_as(occupancy)
    target = .8 * smooth / smooth.sum(-1, keepdim=True).clamp_min(1e-8) + .2 / 196
    return target * frame_mask.unsqueeze(-1)


def native_patch_distribution(attention):
    if attention.ndim != 4 or attention.shape[-2] != 2 or attention.shape[-1] < 3:
        raise ValueError('Expected CLS attention [batch, frames, two layers, patches+2]')
    if not torch.isfinite(attention).all() or (attention < 0).any():
        raise ValueError('Attention must be finite and nonnegative')
    patches = attention[..., 1:-1].float()
    mass = patches.sum(-1, keepdim=True)
    if (mass <= 0).any():
        raise ValueError('Conditional attention requires positive patch mass')
    return patches / mass


def native_attention_loss(attention, target, frame_mask):
    prediction = native_patch_distribution(attention)
    if target.shape != prediction.shape[:2] + (prediction.shape[-1],) or frame_mask.shape != target.shape[:2] or frame_mask.dtype != torch.bool:
        raise ValueError('Attention/target/mask shapes differ')
    if not torch.isfinite(target).all() or (target < 0).any() or not torch.allclose(target.sum(-1), frame_mask.float(), atol=1e-5, rtol=0):
        raise ValueError('Targets must sum to one on valid frames, zero otherwise')
    if not frame_mask.any():
        return attention.float().sum() * 0
    desired = target.float().unsqueeze(-2).expand_as(prediction)
    kl = (desired * (desired.clamp_min(1e-8).log() - prediction.clamp_min(1e-8).log())).sum(-1)
    return kl[frame_mask].mean()


def native_attention_metrics(attention, occupancy, frame_mask):
    target = native_pose_distribution(occupancy, frame_mask)
    prediction = native_patch_distribution(attention)
    if prediction.shape[-1] != 196:
        raise ValueError('Patch16 metrics require 196 patches')
    count = int(frame_mask.sum())
    if not count:
        return {'valid_frames': 0}
    selected = attention.float()[frame_mask]
    patch = prediction[frame_mask]
    occupied = occupancy[frame_mask].unsqueeze(1)
    return {'valid_frames': count, 'target_kl': float(native_attention_loss(attention, target, frame_mask).detach()),
            'joint_region_mass': float((selected[..., 1:-1] * occupied).sum(-1).mean().detach()),
            'conditional_joint_region_mass': float((patch * occupied).sum(-1).mean().detach()),
            'patch_mass': float(selected[..., 1:-1].sum(-1).mean().detach()),
            'cls_mass': float(selected[..., 0].mean().detach()), 'message_mass': float(selected[..., -1].mean().detach()),
            'patch_entropy': float(-(patch * patch.clamp_min(1e-8).log()).sum(-1).mean().detach())}


class NativeTargetCache:
    """Derived targets retain the validated Spatial source identity."""

    def __init__(self, spatial_root, derived_root, processor_config):
        self.spatial = TargetCache(spatial_root, 'spatial', processor_config)
        self.root = Path(derived_root) / TARGET_VERSION

    def load(self, row):
        source = self.spatial.load(row)
        occupancy, valid = source['target'], source['valid']
        if occupancy.shape != (8, 196) or valid.shape != (8,) or valid.dtype != torch.bool:
            raise ValueError('Spatial source shape/mask differs')
        fingerprint = {'version': TARGET_VERSION, 'occupancy_source': source['fingerprint'],
                       'derived_builder_sha256': sha256_file(Path(__file__)), 'grid': [14, 14],
                       'kernel': [5, 5], 'sigma_patches': 1, 'uniform_floor': .2,
                       'occupancy_sha256': hashlib.sha256(occupancy.contiguous().numpy().tobytes()).hexdigest(),
                       'mask_sha256': hashlib.sha256(valid.contiguous().numpy().tobytes()).hexdigest()}
        digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
        path = self.root / digest[:2] / f'{row["sample_id"]}_{digest}.pt'
        expected = native_pose_distribution(occupancy[None], valid[None])[0]
        if path.exists():
            payload = torch.load(path, map_location='cpu', weights_only=True)
            if payload['fingerprint'] != fingerprint or not torch.equal(payload['valid'], valid) or not torch.equal(payload['target'], expected):
                raise ValueError('Native derived cache identity/distribution differs')
            return payload
        payload = {**source, 'occupancy': occupancy, 'target': expected, 'valid': valid, 'fingerprint': fingerprint}
        atomic_save(payload, path)
        return payload


def native_budget_decision(completed_epoch, qualified):
    """Two epochs always end this pilot; extension needs a separate request."""
    if completed_epoch < 2:
        return 'continue'
    return 'two_epoch_review_promising' if any(qualified.get(view) is not None for view in ('whole', 'ensemble')) else 'two_epoch_review_no_candidate'
