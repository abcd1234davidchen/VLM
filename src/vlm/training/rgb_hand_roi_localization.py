"""Generate label-independent RGB hand/context crop boxes for a frozen audit selection."""
from __future__ import annotations
import argparse
import ast
import hashlib
import inspect
import json
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import av
import numpy as np
from vlm.experiments.video_sampling import read_video_pyav, uniform_frame_indices
from vlm.reporting import write_json
from vlm.skeleton_teacher_cache import sha256_file
from vlm.training.hand_roi_audit import square_boxes

_DETECTOR = None
_MP = None


def init_detector(model_path):
    global _DETECTOR, _MP
    import cv2
    import mediapipe as mp
    cv2.setNumThreads(1)
    _MP = mp
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_poses=1,
        min_pose_detection_confidence=.5, min_pose_presence_confidence=.5,
        min_tracking_confidence=.5)
    _DETECTOR = mp.tasks.vision.PoseLandmarker.create_from_options(options)


def localize_row(task):
    sid, video_path = task
    path = Path(video_path)
    with av.open(str(path)) as container:
        indices = uniform_frame_indices(8, container.streams.video[0].frames)
        frames = read_video_pyav(container, indices)
    height, width = frames.shape[1:3]
    hand, context, scales, poses = [], [], [], []
    valid_frames = 0
    for idx, frame in zip(indices, frames):
        image = _MP.Image(image_format=_MP.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
        result = _DETECTOR.detect(image)
        usable = {}
        if result.pose_landmarks:
            for j, point in enumerate(result.pose_landmarks[0]):
                if (min(point.visibility, point.presence) >= .5
                        and math.isfinite(point.x) and math.isfinite(point.y)
                        and 0 <= point.x < 1 and 0 <= point.y < 1):
                    usable[j] = np.array([point.x * width, point.y * height])
        poses.append({'frame_index': int(idx), 'points': {str(j): xy.tolist() for j, xy in usable.items()}})
        pts = [usable[j] for j in range(15, 23) if j in usable]
        if not pts:
            continue
        valid_frames += 1
        hand.extend(pts)
        context.extend(pts + [usable[j] for j in (0, 7, 8, 9, 10, 11, 12) if j in usable])
        if all(j in usable for j in (11, 12, 23, 24)):
            scale = float(np.linalg.norm((usable[11] + usable[12] - usable[23] - usable[24]) / 2))
            if scale > 1:
                scales.append(scale)
    fallback = None
    if valid_frames < 4 or not scales:
        fallback = 'Insufficient RGB hand/torso geometry'
        roi = {key: [0, 0, width, height] for key in ('hands', 'interaction')}
    else:
        roi = square_boxes(hand, context, scales, width, height)
    return {'sample_id': sid, 'video_path': str(path), 'rgb_sha256': sha256_file(path),
            'frame_indices': indices.tolist(), 'boxes': roi,
            'quality': {'valid_hand_frames': valid_frames, 'torso_pixels': float(np.median(scales)) if scales else None},
            'fallback': fallback, 'poses': poses}


def validated_cache(path, pose_hash, geometry_source):
    payload = json.loads(path.read_text())
    lock = json.loads((path.parent / 'experiment_lock.json').read_text())
    if payload['pose_model_sha256'] != pose_hash or lock['mediapipe_version'] != '0.10.32':
        raise ValueError('Localization cache model/version mismatch')
    if any(lock[key] != .5 for key in ('detection_confidence', 'presence_confidence', 'landmark_visibility', 'landmark_presence')):
        raise ValueError('Localization cache thresholds mismatch')
    if 'crop_geometry_sha256' in lock:
        if lock['crop_geometry_sha256'] != hashlib.sha256(geometry_source.encode()).hexdigest():
            raise ValueError('Localization cache crop geometry mismatch')
    else:
        snapshot = (path.parent / 'crop_geometry_snapshot.py').read_text()
        if hashlib.sha256(snapshot.encode()).hexdigest() != lock['source_sha256']['crop_geometry']:
            raise ValueError('Legacy cache geometry snapshot mismatch')
        node = next(n for n in ast.parse(snapshot).body if isinstance(n, ast.FunctionDef) and n.name == 'square_boxes')
        if ast.get_source_segment(snapshot, node).strip() != geometry_source.strip():
            raise ValueError('Legacy cache crop function mismatch')
    return {row['sample_id']: row for row in payload['samples']}


def run(args):
    import mediapipe as mp
    root = args.artifacts_dir
    if (root / 'experiment_lock.json').exists():
        raise ValueError('Use a fresh artifacts directory')
    selection = json.loads(args.selection_lock.read_text())
    split_path = Path('artifacts/xclip/selected_split.json')
    split = json.loads(split_path.read_text())
    split_name = selection.get('split', 'val')
    if split_name not in ('train', 'val'):
        raise ValueError('Only train/val localization is allowed')
    rows = {Path(r['video_path']).stem.removesuffix('_rgb'): r for r in split[split_name]}
    ids = selection['sample_ids']
    if len(set(ids)) != len(ids) or not set(ids) <= set(rows):
        raise ValueError('Selection is not unique canonical train/validation')
    pose_hash = sha256_file(args.pose_model)
    geometry_source = inspect.getsource(square_boxes)
    cache = {} if args.reuse_manifest is None else validated_cache(args.reuse_manifest, pose_hash, geometry_source)
    reused = {}
    for sid in ids:
        if sid in cache:
            if cache[sid]['rgb_sha256'] != sha256_file(Path(rows[sid]['video_path'])):
                raise ValueError('Cached RGB source changed')
            reused[sid] = cache[sid]
    pending = [(sid, rows[sid]['video_path']) for sid in ids if sid not in reused]
    root.mkdir(parents=True, exist_ok=True)
    lock = {'localization': 'RGB-only MediaPipe PoseLandmarker lite', 'mediapipe_version': mp.__version__,
            'pose_model_sha256': pose_hash,
            'source_sha256': {'selection': sha256_file(args.selection_lock), 'split': sha256_file(split_path),
                              'runner': sha256_file(Path(__file__))},
            'crop_geometry_sha256': hashlib.sha256(geometry_source.encode()).hexdigest(),
            'sample_ids': ids, 'split': split_name, 'training': False, 'test_evaluated': False,
            'running_mode': 'IMAGE', 'num_poses': 1, 'workers': args.workers,
            'detection_confidence': .5, 'presence_confidence': .5, 'landmark_visibility': .5, 'landmark_presence': .5,
            'hand_landmarks': list(range(15, 23)), 'context_landmarks': [0, 7, 8, 9, 10, 11, 12],
            'crop_policy': 'Same clip-static union square and torso margin as privileged diagnosis; shoulder-midpoint to hip-midpoint torso scale',
            'fallback': 'Use whole frames when fewer than4 valid hand frames or no torso scale; retain all samples',
            'input_policy': 'Only RGB frames used by localizer; labels and skeleton coordinates not consumed',
            'reused_samples': len(reused)}
    if args.reuse_manifest is not None:
        lock['source_sha256']['reuse_manifest'] = sha256_file(args.reuse_manifest)
    write_json(root / 'experiment_lock.json', lock)
    (root / 'runner_snapshot.py').write_text(Path(__file__).read_text())
    (root / 'crop_geometry_snapshot.py').write_text(geometry_source)
    started = time.monotonic()
    metadata = dict(reused)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=init_detector, initargs=(args.pose_model,)) as executor:
        for count, record in enumerate(executor.map(localize_row, pending, chunksize=1), 1):
            metadata[record['sample_id']] = record
            if count % 32 == 0 or count == len(pending):
                progress = {'state': 'RGB localization', 'samples': len(metadata), 'new_samples': count,
                            'total': len(ids), 'seconds': round(time.monotonic() - started, 1)}
                write_json(root / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
                write_json(root / 'partial_crop_manifest.json', {'samples': list(metadata.values())})
    ordered = [metadata[sid] for sid in ids]
    write_json(root / 'crop_manifest.json', {'localization': lock['localization'], 'pose_model_sha256': pose_hash,
                                          'source_lock_sha256': sha256_file(root / 'experiment_lock.json'), 'samples': ordered})
    report = {'samples': len(ids), 'fallbacks': sum(row['fallback'] is not None for row in ordered),
              'reused_samples': len(reused), 'seconds': time.monotonic() - started, 'training': False, 'test_evaluated': False}
    write_json(root / 'report.json', report)
    print(json.dumps({'state': 'complete', **report}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-lock', type=Path, default=Path('artifacts/hand_roi_audit_v1/experiment_lock.json'))
    parser.add_argument('--pose-model', type=Path, default=Path('artifacts/hand_roi_models/pose_landmarker_lite.task'))
    parser.add_argument('--artifacts-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--reuse-manifest', type=Path)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error('workers must be1–4')
    run(args)
