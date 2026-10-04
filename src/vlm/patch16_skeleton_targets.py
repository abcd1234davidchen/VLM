"""Versioned raw-timing temporal and 196-patch occupancy targets."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import uuid

import av
import torch

from vlm.experiments.video_sampling import uniform_frame_indices
from vlm.phase4_alignment import color_joints, transform_point
from vlm.skeleton_teacher_cache import sha256_file

TARGET_VERSION = 'patch16-skeleton-v1'


def select_body_ids(selected, method, indices):
    """Temporal uses whole sequence; spatial preserves sampled-frame policy."""
    if method not in ('temporal', 'spatial'):
        raise ValueError('Unsupported target method')
    tracked = Counter()
    frames = selected.values() if method == 'temporal' else (selected.get(frame, []) for frame in indices)
    for bodies in frames:
        for body in bodies:
            tracked[body['body_id']] += sum(j['state'] == 2 for j in body['joints'])
    return sorted(tracked, key=lambda body: (-tracked[body], body))[:2]


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def temporal_target(selected, count, indices, body_ids):
    velocity = torch.zeros(count, dtype=torch.float64)
    acceleration = torch.zeros_like(velocity)
    nv, na = torch.zeros_like(velocity), torch.zeros_like(velocity)
    for body_id in body_ids:
        xyz = torch.zeros(count, 25, 3, dtype=torch.float64)
        valid = torch.zeros(count, 25, dtype=torch.bool)
        for frame, bodies in selected.items():
            body = next((b for b in bodies if b['body_id'] == body_id), None)
            if body is not None:
                points = torch.tensor([j['xyz'] for j in body['joints']], dtype=torch.float64)
                tracked = torch.tensor([j['state'] == 2 for j in body['joints']]) & torch.isfinite(points).all(-1)
                if tracked[0]:
                    xyz[frame] = points - points[0]
                    valid[frame] = tracked
        distances = xyz[:, 20].norm(dim=-1)
        good_scale = valid[:, 0] & valid[:, 20] & torch.isfinite(distances) & (distances > 1e-6)
        if not good_scale.any():
            continue
        scale = distances[good_scale].quantile(.5)
        xyz /= scale
        common = valid[1:] & valid[:-1]
        weights = common.sum(-1)
        usable = weights >= 10
        differences = torch.where(common, (xyz[1:] - xyz[:-1]).norm(dim=-1), 0).sum(-1)
        velocity[1:] += torch.where(usable, differences, 0)
        nv[1:] += torch.where(usable, weights, 0)
        common = valid[2:] & valid[1:-1] & valid[:-2]
        weights = common.sum(-1)
        usable = weights >= 10
        differences = torch.where(common, (xyz[2:] - 2 * xyz[1:-1] + xyz[:-2]).norm(dim=-1), 0).sum(-1)
        acceleration[2:] += torch.where(usable, differences, 0)
        na[2:] += torch.where(usable, weights, 0)
    velocity /= nv.clamp_min(1)
    acceleration /= na.clamp_min(1)
    for signal, observations in ((velocity, nv), (acceleration, na)):
        if (observations > 0).any():
            percentile = signal[observations > 0].quantile(.95)
            signal.copy_((signal / percentile).clamp(0, 1) if percentile > 0 else torch.zeros_like(signal))
    valid = nv > 0
    energy = .7 * velocity + .3 * acceleration
    smoothed = torch.zeros_like(energy)
    for frame in range(count):
        region = slice(max(0, frame - 1), min(count, frame + 2))
        mask = valid[region]
        if mask.any():
            smoothed[frame] = energy[region][mask].mean()
    # Nearest frame cells, ties go to the earlier sampled index.
    cells = (torch.arange(count)[:, None] - torch.tensor(indices)[None, :]).abs().argmin(-1)
    cell_energy = torch.zeros(8, dtype=torch.float64)
    cell_counts = []
    for cell in range(8):
        mask = (cells == cell) & valid
        cell_counts.append(int(mask.sum()))
        if mask.any():
            cell_energy[cell] = smoothed[mask].mean()
    reason = 'missing_velocity_cells' if min(cell_counts) == 0 else 'static' if cell_energy.sum() <= 1e-6 else None
    target = (.8 * cell_energy / cell_energy.sum() + .2 / 8).float() if reason is None else torch.zeros(8)
    return {'target': target, 'valid': reason is None, 'reason': reason, 'energy': energy.float(),
            'smoothed_energy': smoothed.float(), 'velocity_valid': valid, 'cell_counts': cell_counts,
            'cell_energy': cell_energy.float(), 'entropy': float(-(target * target.clamp_min(1e-8).log()).sum())}


def build_target(row, method, processor_config):
    video, skeleton = Path(row['rgb_path']), Path(row['skeleton_path'])
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        count, width, height = stream.frames, stream.width, stream.height
    if count < 1 or width < 1 or height < 1:
        raise ValueError('Missing video frame metadata')
    indices = uniform_frame_indices(8, count).tolist()
    raw_count, selected = color_joints(skeleton, set(range(int(row['skeleton']['frame_count']))))
    body_ids = select_body_ids(selected, method, indices)
    identity = {'version': TARGET_VERSION, 'method': method, 'sample_id': row['sample_id'],
                'video_path': str(video.resolve()), 'skeleton_sha256': sha256_file(skeleton),
                'video_fingerprint': [video.stat().st_size, video.stat().st_mtime_ns],
                'video_frames': count, 'skeleton_frames': raw_count, 'frame_indices': indices,
                'source_size': [width, height], 'processor': processor_config,
                'body_ids': body_ids, 'body_policy': f'{"whole-sequence" if method == "temporal" else "sampled-frames"}-top2-state2-tie-body-id',
                'patch_size': 16, 'patches': 196}
    mismatch = count != raw_count if method == 'temporal' else abs(count - raw_count) > 2 or max(indices) >= raw_count
    reason = 'frame_count_mismatch' if mismatch else 'short_video' if count < 8 else None
    if method == 'temporal':
        result = ({'target': torch.zeros(8), 'valid': False, 'reason': reason, 'entropy': 0.}
                  if reason else temporal_target(selected, count, indices, body_ids))
    elif method == 'spatial':
        occupancy = torch.zeros(8, 196)
        mask = torch.zeros(8, dtype=torch.bool)
        coordinates = []
        for slot, frame in enumerate(indices):
            points = []
            if reason is None:
                for body in selected[frame]:
                    if body['body_id'] not in body_ids:
                        continue
                    for joint in body['joints']:
                        x, y = transform_point(joint['x'], joint['y'], width, height)
                        if joint['state'] == 2 and math.isfinite(x) and math.isfinite(y) and 0 <= x < 224 and 0 <= y < 224:
                            points.append([x, y])
                            occupancy[slot, int(y // 16) * 14 + int(x // 16)] = 1
            mask[slot] = len(points) >= 10
            coordinates.append(points)
        occupancy[~mask] = 0
        result = {'target': occupancy, 'valid': mask, 'reason': reason or (None if mask.any() else 'insufficient_tracked_joints'),
                  'coordinates': coordinates}
    else:
        raise ValueError('Unsupported target method')
    return {'identity': identity, **result}


class TargetCache:
    def __init__(self, root, method, processor_config):
        self.root, self.method, self.processor_config = Path(root) / method, method, processor_config
        # Source hashes in namespace prevent silently reading targets from another implementation.
        self.builder_hash = sha256_file(Path(__file__)) + sha256_file(Path(__file__).with_name('phase4_alignment.py'))

    def load(self, row):
        video, raw = Path(row['rgb_path']), Path(row['skeleton_path'])
        fingerprint = {'version': TARGET_VERSION, 'method': self.method, 'source': self.builder_hash,
                       'video': [str(video.resolve()), video.stat().st_size, video.stat().st_mtime_ns],
                       'skeleton': [str(raw.resolve()), raw.stat().st_size, raw.stat().st_mtime_ns],
                       'processor': self.processor_config}
        digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
        path = self.root / digest[:2] / f'{row["sample_id"]}_{digest}.pt'
        if path.exists():
            payload = torch.load(path, map_location='cpu', weights_only=True)
            if payload['fingerprint'] != fingerprint:
                raise ValueError('Target cache identity differs')
            return payload
        payload = build_target(row, self.method, self.processor_config)
        if [raw.stat().st_size, raw.stat().st_mtime_ns] != fingerprint['skeleton'][1:]:
            raise RuntimeError('Raw skeleton changed during target extraction')
        if [video.stat().st_size, video.stat().st_mtime_ns] != fingerprint['video'][1:]:
            raise RuntimeError('Video changed during target extraction')
        payload['fingerprint'] = fingerprint
        atomic_save(payload, path)
        return payload
