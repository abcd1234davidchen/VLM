"""Sequential, bounded RGB fusion/data/input experiments; never consume test rows."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import XCLIPModel, XCLIPProcessor

from vlm.experiments.video_sampling import read_video_pyav, uniform_frame_indices
from vlm.models.lora import lora_state_dict, merge_lora, merged_base_state_dict
from vlm.models.xclip_vision_lora import XCLIPVisionClassifier
from vlm.reporting import write_json
from vlm.skeleton_teacher_cache import sha256_file
from vlm.training.hand_roi_followup import (
    ADAPTER, FOCUS, MODEL, MODEL_REVISION, REFERENCE, SPLIT, TARGETS, load_model, sample_id,
)
from vlm.training.rgb_input_audit import PRODUCT_EXCLUDED_IDS, ViewDataset

UPPER4 = Path('artifacts/hand_roi_upper4_v1')
FOLLOWUP = Path('artifacts/hand_roi_followup_v1')
FULL = Path('artifacts/hand_roi_full86_v1')
VIEWS = ('whole', 'center_zoom70', 'interaction')
EQUAL = {'weights': [1 / 3] * 3, 'temperatures': [1.] * 3}
SOURCE_PATHS = [Path('src/vlm/training/hand_roi_improvement_series.py'), Path('src/vlm/training/hand_roi_followup.py'),
                Path('src/vlm/training/rgb_hand_roi_localization.py'),
                Path('src/vlm/training/hand_roi_audit.py'),
                Path('src/vlm/experiments/video_sampling.py'),
                Path('src/vlm/models/lora.py'), Path('src/vlm/models/xclip_vision_lora.py'),
                Path('src/vlm/training/rgb_input_audit.py')]
INPUT_PATHS = [ADAPTER, SPLIT, REFERENCE, UPPER4 / 'pilot/local_adapter.pt',
               UPPER4 / 'evaluation/view_predictions.pt',
               FOLLOWUP / 'train_selection.json', FOLLOWUP / 'train_localization/crop_manifest.json',
               FOLLOWUP / 'val_selection.json', FULL / 'val_selection.json',
               FULL / 'localization/crop_manifest.json',
               Path('artifacts/hand_roi_models/pose_landmarker_lite.task')]


def read_json(path):
    return json.loads(Path(path).read_text())


def subject(sid):
    match = re.search(r'P(\d{3})', sid)
    if match is None:
        raise ValueError(f'Missing participant: {sid}')
    return int(match.group(1))


def calibration_mask(ids):
    participants = sorted({subject(sid) for sid in ids},
                          key=lambda p: hashlib.sha256(f'roi-series42:P{p:03d}'.encode()).hexdigest())
    if len(participants) != 18:
        raise ValueError('Expected 18 canonical validation participants')
    selected = set(participants[:9])
    return torch.tensor([subject(sid) in selected for sid in ids])


def expanded_selection(split, old_ids):
    """Preserve every old sample; fill quotas without predictions or validation labels."""
    lookup = {sample_id(row): row for row in split['train']}
    if len(set(old_ids)) != 2280 or not set(old_ids) <= set(lookup):
        raise ValueError('Invalid original training selection')
    ids = []
    for cls in range(1, 121):
        previous = [sid for sid in old_ids if int(lookup[sid]['class_id']) == cls]
        if len(previous) != (128 if cls in FOCUS else 8):
            raise ValueError('Original training distribution changed')
        candidates = [sid for sid, row in lookup.items()
                      if int(row['class_id']) == cls and sid not in set(previous)]
        candidates.sort(key=lambda sid: hashlib.sha256(f'roi-series-train42:{sid}'.encode()).hexdigest())
        quota = 512 if cls in FOCUS else 32
        if len(candidates) + len(previous) < quota:
            raise ValueError(f'Insufficient train videos for A{cls:03d}')
        ids.extend(previous + candidates[:quota - len(previous)])
    if len(ids) != 9120 or len(set(ids)) != 9120:
        raise ValueError('Expected exactly 9,120 unique train videos')
    return ids


def expanded_box(box, width, height, scale=1.25):
    left, top, right, bottom = box
    if not (0 <= left < right <= width and 0 <= top < bottom <= height and right-left == bottom-top):
        raise ValueError('Invalid source interaction square')
    if not math.isfinite(scale) or scale < 1:
        raise ValueError('Crop scale must be finite and at least one')
    side = min(width, height, math.ceil((right - left) * scale))
    x = max(0, min(width - side, round((left + right - side) / 2)))
    y = max(0, min(height - side, round((top + bottom - side) / 2)))
    return [x, y, x + side, y + side]


def temporal_indices(frames, temporal=False):
    indices = uniform_frame_indices(16 if temporal else 8, frames)
    return [indices[::2], indices[1::2]] if temporal else [indices]


class SeriesDataset(Dataset):
    def __init__(self, rows, crops, processor, *, temporal=False, larger_crop=False):
        self.rows, self.crops, self.processor = rows, crops, processor
        self.temporal, self.larger_crop = temporal, larger_crop

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, position):
        row = self.rows[position]
        crop = self.crops[sample_id(row)]
        path = Path(row['video_path'])
        if sha256_file(path) != crop['rgb_sha256']:
            raise RuntimeError('RGB video changed after localization')
        with av.open(str(path)) as container:
            container.streams.video[0].codec_context.thread_count = 2
            n = container.streams.video[0].frames
            if uniform_frame_indices(8, n).tolist() != crop['frame_indices']:
                raise RuntimeError('Original localization frames changed')
            groups = temporal_indices(n, self.temporal)
            frames = read_video_pyav(container, np.concatenate(groups))
        height, width = frames.shape[1:3]
        box = [0, 0, width, height]
        if crop['fallback'] is None:
            box = expanded_box(crop['boxes']['interaction'], width, height,
                               1.25 if self.larger_crop else 1.)
            left, top, right, bottom = box
            frames = frames[:, top:bottom, left:right]
        pixels = torch.stack([self.processor(images=list(frames[i*8:(i+1)*8]),
                              return_tensors='pt')['pixel_values'].squeeze(0) for i in range(len(groups))])
        return pixels, int(row['class_id']) - 1, {
            'sample_id': sample_id(row), 'rgb_sha256': crop['rgb_sha256'],
            'frame_indices': [group.tolist() for group in groups], 'box': box,
            'fallback': crop['fallback'], 'dimensions': [width, height]}


def collate(items):
    pixels, labels, metadata = zip(*items)
    return torch.stack(pixels), torch.tensor(labels), list(metadata)


def loader(rows, crops, processor, *, shuffle=False, temporal=False, larger_crop=False):
    return DataLoader(SeriesDataset(rows, crops, processor, temporal=temporal, larger_crop=larger_crop),
                      batch_size=4, shuffle=shuffle, num_workers=4, pin_memory=True,
                      persistent_workers=True, generator=torch.Generator().manual_seed(42), collate_fn=collate)


def validate_logits(logits, n):
    if set(logits) != set(VIEWS):
        raise ValueError('Expected exactly the three RGB views')
    for values in logits.values():
        if values.ndim not in (2, 3) or values.shape[0] != n or values.shape[-1] != 120 or not torch.isfinite(values).all():
            raise ValueError('Expected finite canonical 120-output logits')
        if values.ndim == 3 and values.shape[1] != 2:
            raise ValueError('Expected exactly two eight-frame local forwards')


def load_baseline(ids, labels):
    baseline = torch.load(UPPER4 / 'evaluation/view_predictions.pt', map_location='cpu', weights_only=True)
    if baseline['sample_ids'] != ids or not torch.equal(baseline['labels'], labels) or baseline['test_evaluated'] is not False:
        raise ValueError('Upper4 cache IDs/labels mismatch')
    reference = torch.load(REFERENCE, map_location='cpu', weights_only=True)
    lookup = {sid: index for index, sid in enumerate(reference['sample_ids'])}
    if len(lookup) != len(reference['sample_ids']) or not set(ids) <= set(lookup):
        raise ValueError('Reference ID coverage mismatch')
    selected = torch.tensor([lookup[sid] for sid in ids])
    if not torch.equal(labels, reference['labels'][selected]) or reference['test_evaluated'] is not False:
        raise ValueError('Reference label/provenance mismatch')
    for name in VIEWS[:2]:
        values = reference['logits'][name][selected]
        if not torch.equal(values.softmax(-1), baseline['probabilities'][name]):
            raise ValueError('Upper4 whole/zoom cache differs from frozen reference')
        baseline['logits'][name] = values
    validate_logits(baseline['logits'], len(ids))
    if not torch.allclose(fuse(baseline['logits'], EQUAL), baseline['probabilities']['whole_zoom_interaction_mean'], atol=1e-7, rtol=0):
        raise ValueError('Upper4 equal fusion cache mismatch')
    return baseline


def view_probabilities(logits, temperatures):
    if len(temperatures) != 3 or any(not math.isfinite(t) or t <= 0 for t in temperatures):
        raise ValueError('Invalid temperatures')
    result = []
    for name, temperature in zip(VIEWS, temperatures):
        values = (logits[name] / temperature).softmax(-1)
        result.append(values.mean(1) if values.ndim == 3 else values)
    return result


def fuse(logits, config):
    weights = config['weights']
    if len(weights) != 3 or any(not math.isfinite(w) or w < 0 for w in weights) or abs(sum(weights) - 1) > 1e-8:
        raise ValueError('Invalid fusion simplex')
    probabilities = view_probabilities(logits, config['temperatures'])
    # Match the established sum-then-divide cache convention for the control.
    if weights == [1 / 3] * 3:
        return sum(probabilities) / 3
    return sum(w*p for w, p in zip(weights, probabilities))


def nll(probabilities, labels):
    return float(-probabilities[torch.arange(len(labels)), labels].clamp_min(1e-12).log().double().mean())


def weight_search(logits, labels, mask, temperatures):
    probabilities = [p[mask] for p in view_probabilities(logits, temperatures)]
    y = labels[mask]
    candidates = [[i / 10, j / 10, (10-i-j) / 10] for i in range(11) for j in range(11-i)] + [[1 / 3] * 3]
    def key(weights):
        p = sum(w*v for w, v in zip(weights, probabilities))
        return (-int(p.argmax(-1).eq(y).sum()), nll(p, y), sum(w > 0 for w in weights),
                sum((w-1/3)**2 for w in weights), tuple(weights))
    return {'weights': min(candidates, key=key), 'temperatures': temperatures}


def calibrate(logits, labels, mask):
    temperatures = []
    for name in VIEWS:
        values = logits[name][mask]
        def key(t):
            p = (values/t).softmax(-1)
            if p.ndim == 3:
                p = p.mean(1)
            return nll(p, labels[mask]), abs(t-1), t
        temperatures.append(min((.5, .75, 1., 1.5, 2.), key=key))
    return {'equal': EQUAL, 'weights_only': weight_search(logits, labels, mask, [1.] * 3),
            'temperature_weights': weight_search(logits, labels, mask, temperatures)}


def calibration_key(logits, labels, mask, config):
    p = fuse(logits, config)[mask]
    forwards = sum(w > 0 for w in config['weights']) + int(logits['interaction'].ndim == 3 and config['weights'][2] > 0)
    return (-int(p.argmax(-1).eq(labels[mask]).sum()), nll(p, labels[mask]), forwards)


def choose_config(logits, labels, mask):
    candidates = calibrate(logits, labels, mask)
    best = min(candidates, key=lambda k: calibration_key(logits, labels, mask, candidates[k]))
    baseline = -calibration_key(logits, labels, mask, EQUAL)[0]
    if -calibration_key(logits, labels, mask, candidates[best])[0] <= baseline:
        best = 'equal'
    return candidates, best


def comparisons(probabilities, before, labels, ids, mask, diagnostic_ids):
    after_ok, before_ok = probabilities.argmax(-1).eq(labels), before.argmax(-1).eq(labels)
    groups = {'full': torch.ones(len(ids), dtype=torch.bool), 'calibration': mask, 'confirmation': ~mask,
              'targets': torch.isin(labels, torch.tensor(TARGETS)-1), 'neck_A047': labels.eq(46),
              'attention_A042_A048': (labels >= 41) & (labels <= 47),
              'not_in_prior_diagnostic': torch.tensor([sid not in diagnostic_ids for sid in ids])}
    groups.update({f'A{c:03d}': labels.eq(c-1) for c in range(1, 121) if c not in PRODUCT_EXCLUDED_IDS})
    result = {}
    for name, group in groups.items():
        count = int(group.sum())
        result[name] = {'samples': count, 'correct': int(after_ok[group].sum()),
                        'accuracy': float(after_ok[group].double().mean()) if count else None,
                        'before_correct': int(before_ok[group].sum()),
                        'fixed': int((group & after_ok & ~before_ok).sum()),
                        'regressed': int((group & ~after_ok & before_ok).sum()),
                        'net': int(after_ok[group].sum()-before_ok[group].sum()),
                        'nll': nll(probabilities[group], labels[group]) if count else None}
    return result


def stage_complete(root, stage):
    path = root / stage / 'complete.json'
    if not path.exists():
        return False
    record = read_json(path)
    if record['lock_sha256'] != sha256_file(root / 'experiment_lock.json'):
        raise ValueError(f'Experiment lock changed for {stage}')
    for name, digest in record['outputs'].items():
        if sha256_file(root / stage / name) != digest:
            raise ValueError(f'Completed artifact changed: {stage}/{name}')
    return True


def finish_stage(root, stage):
    directory = root / stage
    outputs = {str(p.relative_to(directory)): sha256_file(p) for p in directory.rglob('*')
               if p.is_file() and p.name not in ('complete.json', 'progress.json')}
    write_json(directory / 'complete.json', {'lock_sha256': sha256_file(root / 'experiment_lock.json'),
                                            'outputs': outputs, 'test_evaluated': False})


def train_expert(root, stage, rows, crops, validation_rows, validation_crops, baseline, *, smoke=False):
    directory = root / stage
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    model = load_model(torch.load(ADAPTER, map_location='cpu', weights_only=False), upper_blocks=4, expand=True)
    processor = XCLIPProcessor.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=True)
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    trainable_names = [name for name, _ in parameters]
    targets = list(model.target_modules)
    if sum(p.numel() for _, p in parameters) != 291960:
        raise ValueError('Upper4 trainable parameter budget changed')
    pixels, _ = next(iter(DataLoader(ViewDataset(validation_rows[:4], processor, ('whole',)), batch_size=4)))
    with torch.inference_mode():
        initial = model(pixels[:, 0].cuda()).cpu()
    error = float((initial-baseline['logits']['whole'][:4]).abs().max())
    if error > 1e-3 or not torch.equal(initial.argmax(-1), baseline['logits']['whole'][:4].argmax(-1)):
        raise RuntimeError('Baseline warm-start parity failed')
    actual_rows = rows[:16] if smoke else rows
    batches = loader(actual_rows, crops, processor, shuffle=True)
    optimizer = torch.optim.AdamW([p for _, p in parameters], lr=1e-5, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(batches))
    model.train()
    gradients, families, steps, seen = {}, {}, 0, 0
    losses = []
    started = time.monotonic()
    for steps, (pixels, y, _) in enumerate(batches, 1):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(pixels[:, 0].cuda()), y.cuda(), label_smoothing=.1)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        loss.backward()
        if steps == 1:
            gradients = {target: float(dict(parameters)[target+'.lora_b'].grad.abs().sum())
                         for target in model.target_modules if target.startswith('vision_model.')}
            families = {family: sum(float(p.grad.abs().sum()) for name, p in parameters
                                    if name.startswith(prefix) and p.grad is not None)
                        for family, prefix in [('vision', 'vision_model.'), ('mit', 'mit.'), ('classifier', 'classifier.')]}
            if len(gradients) != 8 or any(not math.isfinite(v) or v <= 0 for v in [*gradients.values(), *families.values()]):
                raise RuntimeError('Missing/nonfinite vision, MIT or classifier gradients')
        torch.nn.utils.clip_grad_norm_([p for _, p in parameters], 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach()))
        seen += len(y)
        if steps % 25 == 0 or steps == len(batches):
            write_json(directory / 'progress.json', {'state': 'training', 'steps': steps, 'total_steps': len(batches),
                'samples': seen, 'recent_loss': sum(losses[-25:])/len(losses[-25:]),
                'lr': optimizer.param_groups[0]['lr'], 'seconds': time.monotonic()-started})
    if (steps, seen) != ((4, 16) if smoke else (2280, 9120)):
        raise RuntimeError('Training budget not fulfilled')
    identity = {'model_name': MODEL, 'model_revision': MODEL_REVISION, 'upper_blocks': 4, 'vision_layout': 'upper',
                'train_samples': seen, 'optimizer_steps': steps, 'epochs': 1, 'seed': 42,
                'test_evaluated': False, 'init_adapter_sha256': sha256_file(ADAPTER),
                'split_sha256': sha256_file(SPLIT), 'series_lock_sha256': sha256_file(root / 'experiment_lock.json')}
    torch.save({'identity': identity, 'model': {'target_modules': model.target_modules,
               'lora_state_dict': lora_state_dict(model), 'classifier_state_dict': model.classifier.state_dict()}},
               directory / 'local_adapter.pt')
    model.eval()
    check_rows = validation_rows[:16]
    check = loader(check_rows, validation_crops, processor)
    smoke_logits, examples = [], None
    with torch.inference_mode():
        for pixels, _, _ in check:
            values = model(pixels[:, 0].cuda()).cpu()
            if not torch.isfinite(values).all():
                raise RuntimeError('Nonfinite smoke predictions')
            smoke_logits.append(values)
            if examples is None:
                examples = pixels[:, 0].cuda()
        original = model(examples).cpu()
    merge_lora(model)
    merged = {k: v.cpu() for k, v in merged_base_state_dict(model).items()}
    del model, optimizer, parameters, batches, check
    gc.collect()
    torch.cuda.empty_cache()
    restored = XCLIPVisionClassifier(XCLIPModel.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=True),
                                    num_classes=120, head_dropout=.25, feature_noise_std=.01).cuda().eval()
    restored.load_state_dict(merged, strict=True)
    with torch.inference_mode():
        reload_error = float((restored(examples).cpu()-original).abs().max())
    if reload_error > 1e-4:
        raise RuntimeError('Merged RGB-only strict reload failed')
    torch.save(merged, directory / 'merged_local_rgb_state.pt')
    torch.save(torch.cat(smoke_logits), directory / 'smoke_logits.pt')
    write_json(directory / 'report.json', {'status': 'smoke_pass' if smoke else 'training_complete',
               'identity': identity, 'warm_start_parity_max_error': error, 'trainable_parameters': 291960,
               'trainable_names': trainable_names,
               'vision_train_blocks': [8, 9, 10, 11], 'target_modules': targets,
               'vision_target_gradients': gradients, 'gradient_families': families,
               'merged_reload_max_abs_error': reload_error, 'seconds': time.monotonic()-started})
    del restored, merged, examples
    gc.collect()
    torch.cuda.empty_cache()


def predict(root, stage, adapter, rows, crops, baseline, *, temporal=False, larger_crop=False):
    directory = root / stage
    directory.mkdir(parents=True, exist_ok=True)
    source = torch.load(adapter, map_location='cpu', weights_only=False)
    identity = source['identity']
    if (identity['model_revision'] != MODEL_REVISION or identity['init_adapter_sha256'] != sha256_file(ADAPTER)
            or identity['split_sha256'] != sha256_file(SPLIT) or identity['test_evaluated'] is not False):
        raise ValueError('Prediction adapter provenance mismatch')
    model = load_model(source, upper_blocks=4).eval()
    processor = XCLIPProcessor.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=True)
    batches = loader(rows, crops, processor, temporal=temporal, larger_crop=larger_crop)
    output, metadata, seen = [], [], 0
    started = time.monotonic()
    with torch.inference_mode():
        for pixels, y, info in batches:
            count, views = pixels.shape[:2]
            # Sequential forwards keep peak memory at the original batch4 budget.
            values = torch.stack([model(pixels[:, i].cuda()).cpu() for i in range(views)], 1)
            if not torch.equal(y, baseline['labels'][seen:seen+count]) or not torch.isfinite(values).all():
                raise RuntimeError('Invalid prediction labels/logits')
            output.append(values if temporal else values[:, 0])
            metadata.extend(info)
            seen += count
            if seen % 100 == 0 or seen == len(rows):
                write_json(directory / 'progress.json', {'state': 'evaluation', 'samples': seen,
                           'total': len(rows), 'seconds': time.monotonic()-started})
    logits = {name: baseline['logits'][name] for name in VIEWS[:2]}
    logits['interaction'] = torch.cat(output)
    validate_logits(logits, len(rows))
    torch.save({'sample_ids': baseline['sample_ids'], 'labels': baseline['labels'], 'logits': logits,
                'test_evaluated': False, 'training': False}, directory / 'view_predictions.pt')
    write_json(directory / 'input_manifest.json', {'temporal': temporal, 'larger_crop': larger_crop,
                'adapter': str(adapter), 'adapter_sha256': sha256_file(adapter),
                'seconds': time.monotonic()-started, 'fallbacks': sum(i['fallback'] is not None for i in metadata),
                'samples': metadata})
    del model, batches
    gc.collect()
    torch.cuda.empty_cache()


def summarize(root, stage, payload, config, baseline, mask, diagnostic_ids):
    probabilities = fuse(payload['logits'], config)
    before = baseline['probabilities']['whole_zoom_interaction_mean']
    report = {'config': config, 'equal_mean': comparisons(fuse(payload['logits'], EQUAL), before,
                 payload['labels'], payload['sample_ids'], mask, diagnostic_ids),
              'selected_fusion': comparisons(probabilities, before, payload['labels'], payload['sample_ids'], mask, diagnostic_ids),
              'training': False, 'test_evaluated': False}
    write_json(root / stage / 'report.json', report)
    torch.save(probabilities, root / stage / 'ensemble_probabilities.pt')
    return report


def verify_inputs(root):
    lock = read_json(root / 'experiment_lock.json')
    for name, digest in lock['sha256'].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError(f'Locked source/input changed: {name}')


def audit(root):
    """Recompute saved metrics and verify provenance/geometry without GPU inference."""
    checks = {}
    def check(name, condition):
        if not condition:
            raise RuntimeError(f'Artifact audit failed: {name}')
        checks[name] = 'verified'
    verify_inputs(root)
    check('original_inputs_and_sources_unchanged', True)
    for path in SOURCE_PATHS:
        check(f'snapshot_{path.name}', sha256_file(root/'source_snapshots'/path) == sha256_file(path))
    lock = read_json(root/'experiment_lock.json')
    split = read_json(SPLIT)
    rows = [r for r in split['val'] if int(r['class_id']) not in PRODUCT_EXCLUDED_IDS]
    ids = [sample_id(r) for r in rows]
    y = torch.tensor([int(r['class_id'])-1 for r in rows])
    baseline = load_baseline(ids, y)
    mask = calibration_mask(ids)
    cal_ids = [sid for sid, keep in zip(ids, mask.tolist()) if keep]
    confirm_ids = [sid for sid, keep in zip(ids, mask.tolist()) if not keep]
    check('locked_participant_split', lock['calibration_ids'] == cal_ids and lock['confirmation_ids'] == confirm_ids)
    check('participant_disjoint', not {subject(sid) for sid in cal_ids} & {subject(sid) for sid in confirm_ids})
    check('full_validation_coverage', len(ids) == 8640 and len(set(ids)) == 8640)
    diagnostic = set(read_json(FOLLOWUP/'val_selection.json')['sample_ids'])
    check('outside_diagnostic_coverage', len(set(ids)-diagnostic) == 7548)
    fusion = read_json(root/'fusion/report.json')
    configs, winner = choose_config(baseline['logits'], y, mask)
    check('fusion_selected_using_calibration_only', fusion['candidates'] == configs and fusion['winner'] == winner)
    config = fusion['config']
    old_train_ids = read_json(FOLLOWUP/'train_selection.json')['sample_ids']
    train_ids = read_json(root/'selection/train_selection.json')['sample_ids']
    check('deterministic_expanded_selection', train_ids == expanded_selection(split, old_train_ids))
    check('train_validation_disjoint', not set(train_ids) & set(ids))
    training_subjects = {subject(sid) for sid in train_ids}
    check('train_validation_participant_disjoint', not training_subjects & {subject(sid) for sid in ids})
    train_payload = read_json(root/'localization/crop_manifest.json')
    check('train_crop_coverage', [r['sample_id'] for r in train_payload['samples']] == train_ids)
    check('train_rgb_only_localization', train_payload['localization'] == 'RGB-only MediaPipe PoseLandmarker lite')
    check('original_2280_crops_reused', read_json(root/'localization/report.json')['reused_samples'] == 2280)
    trained = torch.load(root/'training/local_adapter.pt', map_location='cpu', weights_only=True)
    old = torch.load(ADAPTER, map_location='cpu', weights_only=False)
    for stage, steps, seen in [('smoke', 4, 16), ('training', 2280, 9120)]:
        report = read_json(root/stage/'report.json')
        check(f'{stage}_budget', report['identity']['optimizer_steps'] == steps and report['identity']['train_samples'] == seen)
        check(f'{stage}_exact_parameter_count', report['trainable_parameters'] == 291960)
        check(f'{stage}_warm_start', report['warm_start_parity_max_error'] <= 1e-3)
        check(f'{stage}_merged_reload', report['merged_reload_max_abs_error'] <= 1e-4)
        check(f'{stage}_gradients', len(report['vision_target_gradients']) == 8 and
              all(math.isfinite(v) and v > 0 for v in [*report['vision_target_gradients'].values(), *report['gradient_families'].values()]))
        vision = [name for name in report['trainable_names'] if name.startswith('vision_model.')]
        check(f'{stage}_trainable_blocks', {int(name.split('.')[3]) for name in vision} == {8, 9, 10, 11}
              and all(name.endswith(('lora_a', 'lora_b')) for name in vision))
        check(f'{stage}_no_base_weights_trainable', all(name.startswith('classifier.') or name.endswith(('lora_a', 'lora_b'))
                                                        for name in report['trainable_names']))
    check('trained_checkpoint_lock', trained['identity']['series_lock_sha256'] == sha256_file(root/'experiment_lock.json'))
    for name, value in trained['model']['lora_state_dict'].items():
        check(f'adapter_finite_{name}', bool(torch.isfinite(value).all()))
        if name.startswith(('vision_model.encoder.layers.10.', 'vision_model.encoder.layers.11.')):
            check(f'tail_updated_{name}', not torch.equal(value, old['model']['lora_state_dict'][name]))
    val_crops = {r['sample_id']: r for r in read_json(FULL/'localization/crop_manifest.json')['samples']}
    stages = ['fusion', 'selection', 'localization', 'smoke', 'training', 'expanded_data', 'temporal', 'context', 'final']
    final = read_json(root/'final/report.json')
    if final['combined_run']:
        stages.append('combined')
    for name in stages:
        check(f'{name}_complete_and_hashes', stage_complete(root, name))
    candidates = {'fusion': baseline}
    for name in ['expanded_data', 'temporal', 'context'] + (['combined'] if final['combined_run'] else []):
        payload = torch.load(root/name/'view_predictions.pt', map_location='cpu', weights_only=True)
        check(f'{name}_ids_labels', payload['sample_ids'] == ids and torch.equal(payload['labels'], y))
        check(f'{name}_no_test', payload['test_evaluated'] is False)
        validate_logits(payload['logits'], len(ids))
        check(f'{name}_finite_120_outputs', True)
        check(f'{name}_frozen_whole_zoom', all(torch.equal(payload['logits'][view], baseline['logits'][view]) for view in VIEWS[:2]))
        expected = comparisons(fuse(payload['logits'], config), baseline['probabilities']['whole_zoom_interaction_mean'], y, ids, mask, diagnostic)
        report = read_json(root/name/'report.json')
        check(f'{name}_metrics_recomputed', report['selected_fusion'] == expected)
        check(f'{name}_probability_cache', torch.equal(torch.load(root/name/'ensemble_probabilities.pt', map_location='cpu', weights_only=True), fuse(payload['logits'], config)))
        metadata = read_json(root/name/'input_manifest.json')
        check(f'{name}_all_fallbacks_retained', metadata['fallbacks'] == 197 and len(metadata['samples']) == 8640)
        check(f'{name}_adapter_hash', sha256_file(Path(metadata['adapter'])) == metadata['adapter_sha256'])
        geometry_valid = True
        for sid, info in zip(ids, metadata['samples']):
            crop = val_crops[sid]
            w, h = info['dimensions']
            expected_box = [0, 0, w, h] if crop['fallback'] is not None else expanded_box(
                crop['boxes']['interaction'], w, h, 1.25 if metadata['larger_crop'] else 1.)
            expected_indices = [g.tolist() for g in temporal_indices(crop['frame_indices'][-1]+1, metadata['temporal'])]
            geometry_valid &= (info['sample_id'] == sid and info['box'] == expected_box and info['frame_indices'] == expected_indices
                               and info['fallback'] == crop['fallback'] and info['rgb_sha256'] == crop['rgb_sha256'])
        check(f'{name}_geometry_sampling_provenance', geometry_valid)
        candidates[name] = payload
    source_count = lambda payload: -calibration_key(payload['logits'], y, mask, config)[0]
    input_base = 'expanded_data' if source_count(candidates['expanded_data']) > source_count(baseline) else 'fusion'
    check('input_base_calibration_only', final['input_baseline'] == input_base)
    input_count = source_count(candidates[input_base])
    check('combined_gate_calibration_only', final['combined_run'] == all(source_count(candidates[name]) > input_count for name in ('temporal', 'context')))
    eligible = {name: payload for name, payload in candidates.items() if name != ('fusion' if input_base == 'expanded_data' else 'expanded_data')}
    winning = min(eligible, key=lambda name: calibration_key(eligible[name]['logits'], y, mask, config))
    if source_count(eligible[winning]) <= input_count:
        winning = input_base
    check('winning_input_calibration_only', final['winning_stage'] == winning)
    payload = torch.load(root/'final/view_predictions.pt', map_location='cpu', weights_only=True)
    check('final_payload_is_selected_candidate', payload['sample_ids'] == ids and torch.equal(payload['labels'], y)
          and all(torch.equal(payload['logits'][view], candidates[winning]['logits'][view]) for view in VIEWS))
    final_configs, _ = choose_config(payload['logits'], y, mask)
    final_configs['locked'] = config
    selected = min(final_configs, key=lambda name: calibration_key(payload['logits'], y, mask, final_configs[name]))
    if source_count({'logits': payload['logits']}) >= -calibration_key(payload['logits'], y, mask, final_configs[selected])[0]:
        selected = 'locked'
    check('final_refit_calibration_only', final['selected_config'] == selected and final['config'] == final_configs[selected])
    expected = comparisons(fuse(payload['logits'], final['config']), baseline['probabilities']['whole_zoom_interaction_mean'], y, ids, mask, diagnostic)
    check('final_metrics_recomputed', final['selected_fusion'] == expected)
    check('confirmation_gain_claim', final['confirmed_gain'] == (expected['calibration']['net'] > 0 and expected['confirmation']['net'] > 0))
    check('no_test_inference', lock['test_evaluated'] is False and final['test_evaluated'] is False)
    write_json(root/'requirement_audit.json', checks)
    return checks


def write_report(root):
    final = read_json(root/'final/report.json')
    fusion = read_json(root/'fusion/report.json')
    lines = ['# RGB accuracy improvement series', '',
             '依序比較融合校準、9,120筆upper4短訓練、時間採樣及裁切上下文。所有選擇只使用固定calibration participants；confirmation與全量結果用於報告。這兩組皆曾用於開發，不是獨立test。', '',
             '| 試驗 | Calibration答對／總數 | Confirmation答對／總數 | Full答對／8,640 | Full accuracy |',
             '|---|---:|---:|---:|---:|']
    def row(name, groups):
        values = [f"{groups[g]['correct']}/{groups[g]['samples']}" for g in ('calibration', 'confirmation', 'full')]
        lines.append(f"| {name} | {' | '.join(values)} | {groups['full']['accuracy']*100:.4f}% |")
    row('upper4等權基準', fusion['results']['equal'])
    for name in ('weights_only', 'temperature_weights'):
        row(name, fusion['results'][name])
    for name in ('expanded_data', 'temporal', 'context') + (('combined',) if final['combined_run'] else ()):
        report = read_json(root/name/'report.json')
        row(name+'（等權）', report['equal_mean'])
        row(name+'（固定融合）', report['selected_fusion'])
    row('final', final['selected_fusion'])
    lines += ['', f"Calibration選出的input：`{final['winning_stage']}`；融合：`{final['config']}`；confirmation gain：`{final['confirmed_gain']}`。", '',
              '| 群組 | 原upper4答對 | Final答對／總數 | 修正 | 退步 | Net |', '|---|---:|---:|---:|---:|---:|']
    for name in ['full', 'calibration', 'confirmation', 'targets', 'neck_A047', 'attention_A042_A048', 'not_in_prior_diagnostic'] + [f'A{c:03d}' for c in range(1, 121) if c not in PRODUCT_EXCLUDED_IDS]:
        result = final['selected_fusion'][name]
        lines.append(f"| {name} | {result['before_correct']} | {result['correct']}/{result['samples']} | {result['fixed']} | {result['regressed']} | {result['net']:+d} |")
    lines += ['', 'Whole與zoom保持原權重／採樣；local expert使用rank16、blocks8–11、291,960可訓練參數。更多資料從Phase3 baseline重新初始化，單pass2,280steps，未接續舊local expert；smoke權重未接續正式訓練。', '',
              '時間採樣為16個均勻位置分成兩組八幀，先平均local softmax機率，再作三視角融合；context為原interaction crop邊長1.25倍。所有197個fallback均保留，全120 outputs仍可被預測，沒有test inference或自動部署。', '',
              f"Artifact audit：{len(read_json(root/'requirement_audit.json'))}項全部通過。設定、source snapshots、SHA256、逐輪predictions、輸入metadata與完整report保存在 `{root}`。", '',
              '```bash', f'OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/vlm-series-matplotlib .venv/bin/python -m vlm.training.hand_roi_improvement_series --artifacts-dir {root}',
              '```', '', '重跑使用fresh root；中斷可加 `--resume`，完成階段與來源hash不符會拒絕重用。`--audit-only --resume`可在CPU重新驗證完整產物。', '']
    (root/'results.md').write_text('\n'.join(lines))


def run(root, *, resume=False, fusion_only=False):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    split = read_json(SPLIT)
    rows = [r for r in split['val'] if int(r['class_id']) not in PRODUCT_EXCLUDED_IDS]
    ids = [sample_id(r) for r in rows]
    labels = torch.tensor([int(r['class_id'])-1 for r in rows])
    baseline = load_baseline(ids, labels)
    if (len(ids) != 8640 or ids != read_json(FULL / 'val_selection.json')['sample_ids']):
        raise ValueError('Canonical retained86 baseline mismatch')
    validate_logits(baseline['logits'], len(ids))
    if int(fuse(baseline['logits'], EQUAL).argmax(-1).eq(labels).sum()) != 7305:
        raise ValueError('Upper4 baseline does not match 7,305 correct')
    mask = calibration_mask(ids)
    diagnostic_ids = set(read_json(FOLLOWUP / 'val_selection.json')['sample_ids'])
    if root.exists():
        if not resume:
            raise ValueError('Use a fresh root or --resume')
        verify_inputs(root)
    else:
        root.mkdir(parents=True)
        hashes = {str(p): sha256_file(p) for p in SOURCE_PATHS + INPUT_PATHS}
        write_json(root / 'experiment_lock.json', {'sha256': hashes, 'seed': 42, 'train_samples': 9120,
                   'steps': 2280, 'epochs': 1, 'batch_size': 4, 'workers': 4, 'model_revision': MODEL_REVISION,
                   'test_evaluated': False, 'calibration_ids': [sid for sid, keep in zip(ids, mask.tolist()) if keep],
                   'confirmation_ids': [sid for sid, keep in zip(ids, mask.tolist()) if not keep],
                   'selection_policy': 'Calibration correct count; NLL; fewer forwards. Strict accuracy gain to advance.',
                   'recipe': {'lr': 1e-5, 'weight_decay': 5e-4, 'schedule': 'cosine one pass',
                              'label_smoothing': .1, 'gradient_clip': 1., 'rank': 16, 'alpha': 32,
                              'lora_dropout': .05, 'head_dropout': .25, 'feature_noise_std': .01},
                   'development_validation_warning': 'Both groups were previously used in development; neither is an independent test.'})
        snapshots = root / 'source_snapshots'
        for path in SOURCE_PATHS:
            target = snapshots / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(path.read_text())
    def stage(name):
        verify_inputs(root)
        write_json(root / 'progress.json', {'state': name, 'test_evaluated': False})
        (root / name).mkdir(parents=True, exist_ok=True)
        return stage_complete(root, name)
    if not stage('fusion'):
        candidates, winner = choose_config(baseline['logits'], labels, mask)
        summaries = {}
        for name, config in candidates.items():
            summaries[name] = comparisons(fuse(baseline['logits'], config), fuse(baseline['logits'], EQUAL),
                                          labels, ids, mask, diagnostic_ids)
        write_json(root / 'fusion/report.json', {'candidates': candidates, 'winner': winner,
                    'config': candidates[winner], 'results': summaries, 'test_evaluated': False})
        finish_stage(root, 'fusion')
    config = read_json(root / 'fusion/report.json')['config']
    if fusion_only:
        write_json(root / 'progress.json', {'state': 'fusion_complete', 'test_evaluated': False})
        return
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for remaining stages; use GPU host and --resume')
    if not (root / 'gpu_before.txt').exists():
        (root / 'gpu_before.txt').write_text(subprocess.run(['nvidia-smi'], check=True, capture_output=True, text=True).stdout)
    if not stage('selection'):
        old_ids = read_json(FOLLOWUP / 'train_selection.json')['sample_ids']
        train_ids = expanded_selection(split, old_ids)
        if set(train_ids) & set(ids):
            raise ValueError('Train/validation ID overlap')
        write_json(root / 'selection/train_selection.json', {'split': 'train', 'sample_ids': train_ids,
                   'selection': 'Preserve old selection; supplement by SHA256 roi-series-train42 ID hashes, no predictions',
                   'classes': list(range(1, 121)), 'focus_classes': list(FOCUS)})
        finish_stage(root, 'selection')
    if not stage('localization'):
        command = [sys.executable, '-m', 'vlm.training.rgb_hand_roi_localization', '--selection-lock',
                   str(root / 'selection/train_selection.json'), '--reuse-manifest',
                   str(FOLLOWUP / 'train_localization/crop_manifest.json'), '--artifacts-dir', str(root / 'localization')]
        # An incomplete localization stage restarts deterministically under the same root lock.
        localization_lock = root / 'localization/experiment_lock.json'
        if localization_lock.exists():
            localization_lock.unlink()
        with (root / 'localization.log').open('w') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        finish_stage(root, 'localization')
    train_ids = read_json(root / 'selection/train_selection.json')['sample_ids']
    train_lookup = {sample_id(r): r for r in split['train']}
    train_rows = [train_lookup[sid] for sid in train_ids]
    train_crops = {r['sample_id']: r for r in read_json(root / 'localization/crop_manifest.json')['samples']}
    val_crops = {r['sample_id']: r for r in read_json(FULL / 'localization/crop_manifest.json')['samples']}
    if set(train_crops) != set(train_ids) or set(val_crops) != set(ids):
        raise ValueError('Localization ID coverage mismatch')
    for name, smoke in [('smoke', True), ('training', False)]:
        if not stage(name):
            train_expert(root, name, train_rows, train_crops, rows, val_crops, baseline, smoke=smoke)
            finish_stage(root, name)
    if not stage('expanded_data'):
        predict(root, 'expanded_data', root / 'training/local_adapter.pt', rows, val_crops, baseline)
        payload = torch.load(root / 'expanded_data/view_predictions.pt', map_location='cpu', weights_only=True)
        summarize(root, 'expanded_data', payload, config, baseline, mask, diagnostic_ids)
        finish_stage(root, 'expanded_data')
    incumbent = 'fusion'
    incumbent_payload = baseline
    adapter = UPPER4 / 'pilot/local_adapter.pt'
    expanded = torch.load(root / 'expanded_data/view_predictions.pt', map_location='cpu', weights_only=True)
    current_count = -calibration_key(baseline['logits'], labels, mask, config)[0]
    expanded_count = -calibration_key(expanded['logits'], labels, mask, config)[0]
    if expanded_count > current_count:
        incumbent, incumbent_payload = 'expanded_data', expanded
        adapter = root / 'training/local_adapter.pt'
        current_count = expanded_count
    input_base = incumbent
    input_gains = []
    candidates = {incumbent: incumbent_payload}
    for name, temporal, larger_crop in [('temporal', True, False), ('context', False, True)]:
        if not stage(name):
            predict(root, name, adapter, rows, val_crops, baseline, temporal=temporal, larger_crop=larger_crop)
            payload = torch.load(root / name / 'view_predictions.pt', map_location='cpu', weights_only=True)
            report = summarize(root, name, payload, config, baseline, mask, diagnostic_ids)
            report['input_baseline'] = input_base
            report['vs_input_baseline'] = comparisons(fuse(payload['logits'], config), fuse(incumbent_payload['logits'], config),
                                                       labels, ids, mask, diagnostic_ids)
            write_json(root / name / 'report.json', report)
            finish_stage(root, name)
        payload = torch.load(root / name / 'view_predictions.pt', map_location='cpu', weights_only=True)
        candidates[name] = payload
        if -calibration_key(payload['logits'], labels, mask, config)[0] > current_count:
            input_gains.append(name)
    if len(input_gains) == 2:
        if not stage('combined'):
            predict(root, 'combined', adapter, rows, val_crops, baseline, temporal=True, larger_crop=True)
            payload = torch.load(root / 'combined/view_predictions.pt', map_location='cpu', weights_only=True)
            summarize(root, 'combined', payload, config, baseline, mask, diagnostic_ids)
            finish_stage(root, 'combined')
        candidates['combined'] = torch.load(root / 'combined/view_predictions.pt', map_location='cpu', weights_only=True)
    winning = min(candidates, key=lambda name: calibration_key(candidates[name]['logits'], labels, mask, config))
    if -calibration_key(candidates[winning]['logits'], labels, mask, config)[0] <= current_count:
        winning = incumbent
    if not stage('final'):
        payload = candidates[winning]
        configs, selected = choose_config(payload['logits'], labels, mask)
        # Also retain the previously locked fusion: refitting must not reduce calibration accuracy.
        configs['locked'] = config
        selected = min(configs, key=lambda name: calibration_key(payload['logits'], labels, mask, configs[name]))
        selected_config = configs[selected]
        if -calibration_key(payload['logits'], labels, mask, selected_config)[0] <= -calibration_key(payload['logits'], labels, mask, config)[0]:
            selected_config, selected = config, 'locked'
        report = summarize(root, 'final', payload, selected_config, baseline, mask, diagnostic_ids)
        report.update({'winning_stage': winning, 'adapter': str(adapter), 'input_baseline': input_base,
                       'fusion_candidates': configs, 'selected_config': selected,
                       'combined_run': len(input_gains) == 2,
                       'confirmed_gain': report['selected_fusion']['calibration']['net'] > 0
                                         and report['selected_fusion']['confirmation']['net'] > 0})
        # Fusion baseline winner can use the old adapter even if training was evaluated.
        if winning == 'fusion':
            report['adapter'] = str(UPPER4 / 'pilot/local_adapter.pt')
        write_json(root / 'final/report.json', report)
        torch.save(payload, root / 'final/view_predictions.pt')
        finish_stage(root, 'final')
    verify_inputs(root)
    for name in ['fusion', 'selection', 'localization', 'smoke', 'training', 'expanded_data', 'temporal', 'context', 'final']:
        if not stage_complete(root, name):
            raise RuntimeError(f'Incomplete stage: {name}')
    audit(root)
    write_report(root)
    write_json(root / 'progress.json', {'state': 'complete', 'test_evaluated': False,
               'final': read_json(root / 'final/report.json')['selected_fusion']['full']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--fusion-only', action='store_true', help='CPU-only first stage; resume on GPU host afterwards')
    parser.add_argument('--audit-only', action='store_true', help='Recompute completed artifact checks on CPU')
    args = parser.parse_args()
    try:
        if args.audit_only:
            torch.set_num_threads(2)
            audit(args.artifacts_dir)
        else:
            run(args.artifacts_dir, resume=args.resume, fusion_only=args.fusion_only)
    except Exception as exc:
        if (args.artifacts_dir / 'experiment_lock.json').exists():
            progress = read_json(args.artifacts_dir / 'progress.json') if (args.artifacts_dir / 'progress.json').exists() else {}
            write_json(args.artifacts_dir / 'failure.json', {'stage': progress.get('state'), 'error': str(exc), 'test_evaluated': False})
        raise


if __name__ == '__main__':
    main()
