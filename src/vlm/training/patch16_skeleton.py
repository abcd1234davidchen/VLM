"""Fixed patch16 skeleton experiments: audit, smoke, then guarded full pilot.

All targets are training-only; test data is never loaded. See the plan in docs.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import random
import shutil
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import XCLIPModel, XCLIPProcessor

from vlm.data.skeleton import SKELETON_TENSOR_PREPROCESSING_ID, load_skeleton_tensor
from vlm.models.lora import load_lora_state_dict, lora_state_dict, merge_lora, merged_base_state_dict
from vlm.models.patch16_temporal import XCLIPTemporalClassifier, XCLIPTemporalLoRAClassifier
from vlm.models.skeletonclip import SkeletonPrototypeTeacher
from vlm.models.ski_semantic import SKISemanticClassifier
from vlm.models.xclip_vision_lora import XCLIPVisionClassifier, XCLIPVisionLoRAClassifier, XCLIPSpatialLoRAClassifier
from vlm.patch16_skeleton_contracts import early_kill, promising, temporal_loss, terminal_status, view_probabilities
from vlm.patch16_skeleton_targets import TARGET_VERSION, TargetCache, atomic_save
from vlm.processed_uint8_cache import ProcessedUInt8Cache
from vlm.reporting import write_json
from vlm.skeleton_teacher_cache import sha256_file
from vlm.spatial_joint_targets import spatial_region_loss
from vlm.training.decision_distillation import score_kl
from vlm.training.engine import set_seed
from vlm.training.rgb_input_audit import ViewDataset
from vlm.training.ski_semantic import balanced
from vlm.training.skeletonclip_phase4_gate2 import _sample_id, _worker_seed
from vlm.training.spatial_attention import classification_report
from vlm.xclip_features import FRAME_SAMPLING_STRATEGY

STUDENT = ('microsoft/xclip-base-patch16', 'd6184e3fd8780d04c85d0f1eabe5f94bf44d98f6')
TEACHER_TEXT = ('microsoft/xclip-base-patch32', 'a2e27a78a2b5d802e894b8a1ef14f3a8ce490963')
CLASSES = [f'A{i:03d}' for i in range(1, 60)]
VIEWS = ('whole', 'center_zoom70')
BASE = Path('artifacts/patch16_resolution/pilot_seed42/vision_lora_ce_seed42')


def utc():
    return datetime.now(timezone.utc).isoformat()


def emit(root, state, **fields):
    record = {'state': state, 'timestamp': utc(), 'test_evaluated': False, **fields}
    write_json(root / 'progress.json', record)
    print(json.dumps(record), flush=True)


def source_hashes(args):
    files = [Path(__file__), Path(__file__).parents[1] / 'patch16_skeleton_contracts.py',
             Path(__file__).parents[1] / 'patch16_skeleton_targets.py',
             Path(__file__).parents[1] / 'models/patch16_temporal.py',
             Path(__file__).parents[1] / 'models/ski_semantic.py',
             Path(__file__).parents[1] / 'models/xclip_vision_lora.py',
             Path(__file__).parents[1] / 'models/lora.py',
             Path(__file__).parents[1] / 'models/skeletonclip.py',
             Path(__file__).parents[1] / 'phase4_alignment.py',
             Path(__file__).parents[1] / 'spatial_joint_targets.py',
             Path(__file__).parents[1] / 'processed_uint8_cache.py',
             Path(__file__).with_name('rgb_input_audit.py'), Path(__file__).with_name('ski_semantic.py'),
             Path(__file__).with_name('decision_distillation.py'), Path(__file__).with_name('spatial_attention.py'),
             args.adapter, args.teacher, args.split_manifest, args.paired_manifest,
             args.baseline_views, args.baseline_logits, args.baseline_lock]
    if getattr(args, 'anchor_cache', None):
        files.append(Path(__file__).with_name('patch16_anchor_cache.py'))
    if args.method in ('native', 'ce_retention'):
        files.remove(args.teacher)
        files.extend([Path(__file__).parents[1] / 'models/patch16_native.py', Path(__file__).parents[1] / 'patch16_native_targets.py', Path(__file__).with_name('patch16_native_runtime.py'), Path(__file__).with_name('patch16_acceleration.py')])
    if args.method == 'ce_retention':
        files.extend([Path(__file__).with_name('patch16_control.py'), Path(__file__).with_name('patch16_control_compare.py'), args.native_reference / 'experiment_lock.json', *[args.native_reference / f'epoch{epoch:02d}' / name for epoch in (1, 2) for name in ('checkpoint.pt', 'view_predictions.pt')]])
    if args.method in ('spatial', 'native'):
        files.extend([Path('docs/skeletonclip-phase4-gate0-visual-review.json'),
                      Path('docs/skeletonclip-phase4-gate0-landmarks.json'),
                      Path('docs/skeletonclip-phase4-gate0-lock.json')])
    return {str(path): sha256_file(path) for path in files}


def snapshot_sources(root, hashes):
    """Keep absolute source paths inside the artifact namespace, fail closed."""
    for source, digest in hashes.items():
        path = Path(source)
        if path.suffix != '.py':
            continue
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise RuntimeError(f'Source changed before snapshot: {path}')
        relative = path.resolve().relative_to(Path.cwd().resolve())
        destination = root / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(payload)


def audit_rows(paired, rgb):
    lookup = {_sample_id(row['video_path']): row for row in rgb}
    selected = []
    for cls in range(59):
        bucket = [row for row in paired if row['class_index'] == cls]
        bucket.sort(key=lambda r: hashlib.sha256(f'42:{r["sample_id"]}'.encode()).hexdigest())
        if len(bucket) < 8:
            raise ValueError('Missing held-out audit class coverage')
        selected.extend(lookup[row['sample_id']] for row in bucket[:8])
    return selected


def validate_sources(args):
    split = json.loads(args.split_manifest.read_text())
    paired = json.loads(args.paired_manifest.read_text())
    if ((len(split['train']), len(split['val'])) != (80640, 12960) or split['max_seq_len'] != 8
            or split['frame_sampling_strategy'] != FRAME_SAMPLING_STRATEGY
            or (len(paired['train']), len(paired['val'])) != (45236, 4180)):
        raise ValueError('Canonical data/sampling changed')
    source = torch.load(args.adapter, map_location='cpu', weights_only=False)
    identity = source['identity']
    if (identity['model_name'], identity['model_revision']) != STUDENT or identity['split_sha256'] != sha256_file(args.split_manifest) or identity['test_evaluated'] is not False or identity['arm'] != 'vision' or source['epoch'] != 20:
        raise ValueError('Patch16 epoch20 adapter provenance differs')
    teacher_source = None
    if args.method not in ('native', 'ce_retention'):
        teacher_source = torch.load(args.teacher, map_location='cpu', weights_only=False)
        identity = teacher_source['identity']
        if (identity['head'] != 'xclip' or identity['backbone'] != 'blockgcn' or identity['frames'] != 64
                or identity['ordered_class_ids'] != CLASSES or identity['manifest_sha256'] != sha256_file(args.paired_manifest)
                or identity['preprocessing_id'] != SKELETON_TENSOR_PREPROCESSING_ID
                or (identity['text']['model_name'], identity['text']['model_revision']) != TEACHER_TEXT
                or identity['text']['ordered_class_ids'] != CLASSES):
            raise ValueError('Frozen teacher text identity differs')
    skeleton = {}
    all_ids = set()
    for part in ('train', 'val'):
        rgb = {_sample_id(r['video_path']): r for r in split[part]}
        if len(rgb) != len(split[part]) or set(rgb) & all_ids:
            raise ValueError('Duplicate IDs or split overlap')
        all_ids.update(rgb)
        rows = paired[part]
        ids = [r['sample_id'] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate paired IDs')
        for r in rows:
            match = rgb.get(r['sample_id'])
            if (match is None or int(match['class_id']) - 1 != r['class_index'] or r['class_id'] != CLASSES[r['class_index']]
                    or Path(match['video_path']).resolve() != Path(r['rgb_path']).resolve()):
                raise ValueError('Partial59 pairing or label mismatch')
        if args.method in ('native', 'ce_retention'):
            skeleton[part] = {'sample_ids': []}
            continue
        payload = torch.load(args.skeleton_cache / f'{part}_45236_4180_64.pt', map_location='cpu', weights_only=True, mmap=True)
        if (payload['sample_ids'] != ids or payload['preprocessing_id'] != SKELETON_TENSOR_PREPROCESSING_ID
                or not torch.equal(payload['labels'], torch.tensor([r['class_index'] for r in rows]))
                or payload['inputs'].shape != (len(rows), 3, 64, 25, 2)):
            raise ValueError('Skeleton cache contract differs')
        for index in (0, len(rows) // 2, len(rows) - 1):
            if not torch.equal(load_skeleton_tensor(Path(rows[index]['skeleton_path']), frames=64), payload['inputs'][index]):
                raise ValueError('Raw skeleton/cache mismatch')
        # Chunked scan keeps mmap validation inexpensive in memory.
        for chunk in payload['inputs'].split(512):
            if not torch.isfinite(chunk).all():
                raise ValueError('Nonfinite skeleton cache')
        skeleton[part] = payload
    reference = torch.load(args.baseline_views, map_location='cpu', weights_only=True)
    ids = reference['sample_ids']
    lookup = {s: i for i, s in enumerate(ids)}
    if len(lookup) != 12960 or set(lookup) != {_sample_id(r['video_path']) for r in split['val']} or reference['test_evaluated'] is not False:
        raise ValueError('Baseline view coverage differs')
    for r in split['val']:
        if int(reference['labels'][lookup[_sample_id(r['video_path'])]]) != int(r['class_id']) - 1:
            raise ValueError('Baseline labels differ')
    if any(v.shape != (12960, 120) for v in reference['logits'].values()):
        raise ValueError('Baseline raw120 shape differs')
    view_probabilities(reference['logits'])
    whole = torch.load(args.baseline_logits, map_location='cpu', weights_only=True)
    ordered = torch.tensor([lookup[s] for s in whole['sample_ids']])
    if (len(set(whole['sample_ids'])) != 12960 or not torch.equal(whole['labels'], reference['labels'][ordered])
            or not torch.allclose(whole['logits'], reference['logits']['whole'][ordered], atol=2e-4, rtol=1e-5)):
        raise ValueError('Baseline whole logits disagree')
    processor = XCLIPProcessor.from_pretrained(STUDENT[0], revision=STUDENT[1], local_files_only=True)
    old_processor = XCLIPProcessor.from_pretrained(TEACHER_TEXT[0], revision=TEACHER_TEXT[1], local_files_only=True)
    if processor.image_processor.to_dict() != old_processor.image_processor.to_dict():
        raise ValueError('Patch16/patch32 processor differs; cannot reuse pixels')
    # Pixels keep patch32 identity; student model identity stays patch16.
    cache = ProcessedUInt8Cache(args.rgb_cache, old_processor, model_name=TEACHER_TEXT[0], model_revision=TEACHER_TEXT[1])
    return split, paired, skeleton, source, teacher_source, reference, processor, cache


class TrainingDataset(Dataset):
    def __init__(self, rows, cache, paired, skeleton, method, targets=None):
        self.rows, self.cache, self.method, self.targets = rows, cache, method, targets
        self.paired = {r['sample_id']: r for r in paired}
        self.skeleton = skeleton
        self.index = {s: i for i, s in enumerate(skeleton['sample_ids'])}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        sample = _sample_id(row['video_path'])
        label = int(row['class_id']) - 1
        pixels = self.cache.load(Path(row['video_path']))
        if self.method == 'ce_retention':
            return pixels, label, torch.zeros(0), torch.tensor(False)
        if self.method == 'semantic':
            paired = sample in self.index
            target = self.skeleton['inputs'][self.index[sample]] if paired else torch.zeros(3, 64, 25, 2)
            valid = torch.tensor(paired)
        elif sample in self.paired:
            payload = self.targets.load(self.paired[sample])
            target, valid = payload['target'], torch.as_tensor(payload['valid'])
        else:
            target = torch.zeros(8) if self.method == 'temporal' else torch.zeros(8, 196)
            valid = torch.tensor(False) if self.method == 'temporal' else torch.zeros(8, dtype=torch.bool)
        return pixels, label, target, valid


class CachedAnchorDataset(Dataset):
    """Keep anchor rows aligned with shuffled training samples."""
    def __init__(self, dataset, logits):
        self.dataset, self.logits = dataset, logits
        if len(dataset) != len(logits):
            raise ValueError('Training/cache row count differs')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return (*self.dataset[index], self.logits[index])


def loader(dataset, args, shuffle=False, batch_size=4):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=args.workers,
                      worker_init_fn=_worker_seed, pin_memory=True, persistent_workers=args.workers > 0,
                      generator=torch.Generator().manual_seed(42))


def build_model(method, source, teacher_source=None, clean=False):
    xclip = XCLIPModel.from_pretrained(STUDENT[0], revision=STUDENT[1], local_files_only=True)
    kwargs = {'num_classes': 120, 'head_dropout': .25, 'feature_noise_std': .01}
    if clean:
        cls = XCLIPTemporalClassifier if method == 'temporal' else XCLIPVisionClassifier
    else:
        kwargs.update(upper_blocks=2, lora_rank=16, lora_alpha=32, lora_dropout=.05)
        cls = {'semantic': SKISemanticClassifier, 'temporal': XCLIPTemporalLoRAClassifier,
               'spatial': XCLIPSpatialLoRAClassifier, 'anchor': XCLIPVisionLoRAClassifier}[method]
        if method == 'semantic':
            kwargs.update(text_prototypes=teacher_source['state_dict']['text_prototypes'],
                          semantic_scale=float(teacher_source['state_dict']['log_logit_scale'].exp().clamp(max=100)))
    model = cls(xclip, **kwargs).eval()
    if not clean:
        if model.target_modules != source['model']['target_modules'] or len(model.target_modules) != 6:
            raise ValueError('LoRA target whitelist differs')
        load_lora_state_dict(model, source['model']['lora_state_dict'])
        model.classifier.load_state_dict(source['model']['classifier_state_dict'], strict=True)
    return model


def build_candidate(method, source, teacher_source=None, clean=False):
    if method not in ('native', 'ce_retention'):
        return build_model(method, source, teacher_source, clean=clean)
    if clean:
        return build_model('anchor', source, clean=True)
    from vlm.models.patch16_native import XCLIPNativeLoRAClassifier
    xclip = XCLIPModel.from_pretrained(STUDENT[0], revision=STUDENT[1], local_files_only=True, attn_implementation='eager')
    if (xclip.config.vision_config.num_hidden_layers, xclip.config.vision_config.num_attention_heads, xclip.config.vision_config.patch_size) != (12, 12, 16):
        raise ValueError('Native backbone layers/heads/patch geometry differs')
    model = XCLIPNativeLoRAClassifier(xclip, num_classes=120, head_dropout=.25, feature_noise_std=.01,
                                    upper_blocks=2, lora_rank=16, lora_alpha=32, lora_dropout=.05).eval()
    if model.target_modules != source['model']['target_modules'] or len(model.target_modules) != 6:
        raise ValueError('Native LoRA whitelist differs')
    load_lora_state_dict(model, source['model']['lora_state_dict'])
    model.classifier.load_state_dict(source['model']['classifier_state_dict'], strict=True)
    return model


def forward_aux(model, pixels, method):
    key = {'semantic': 'return_semantic', 'temporal': 'return_temporal', 'spatial': 'return_spatial', 'native': 'return_native_attentions', 'ce_retention': 'return_native_attentions'}[method]
    return model(pixels, **{key: True})


def auxiliary(aux, labels, target, valid, method, teacher):
    if method == 'ce_retention':
        return torch.zeros((), device=aux.device, dtype=torch.float32)
    if method == 'native':
        from vlm.patch16_native_targets import native_attention_loss
        return .1 * native_attention_loss(aux.float(), target.float(), valid)
    if method == 'temporal':
        return .1 * temporal_loss(aux, target, valid)
    if method == 'spatial':
        return .1 * spatial_region_loss(aux, target, valid)
    if not valid.any():
        return aux.sum() * 0
    if not (labels[valid] < 59).all():
        raise ValueError('Semantic label outside partial59')
    with torch.no_grad():
        scores = teacher(target[valid])
    return .1 * F.cross_entropy(aux[valid], labels[valid]) + .01 * F.mse_loss(aux[valid].float(), scores.float())


def reports(logits, labels, ids, paired_ids, baseline=None):
    result = {}
    for view, probability in view_probabilities(logits).items():
        report = classification_report(probability, labels, ids, paired_ids)
        ok = probability.argmax(-1).eq(labels)
        attention = (labels >= 41) & (labels <= 47)
        report.update(samples=len(labels), correct=int(ok.sum()), attention_samples=int(attention.sum()),
                      attention_correct=int(ok[attention].sum()),
                      nll=float(-probability[torch.arange(len(labels)), labels].clamp_min(1e-38).log().mean()))
        if view != 'ensemble':
            report['ce'] = float(F.cross_entropy(logits[view].float(), labels))
        else:
            report['ce'] = report['nll']
        if baseline is not None:
            old = view_probabilities(baseline)[view].argmax(-1).eq(labels)
            report['fixed_ids'] = [ids[i] for i in (~old & ok).nonzero().flatten().tolist()]
            report['regressed_ids'] = [ids[i] for i in (old & ~ok).nonzero().flatten().tolist()]
        result[view] = report
    return result


def evaluate(model, rows, processor, args, root, state):
    outputs = {view: [] for view in VIEWS}
    labels = []
    started = time.monotonic()
    with torch.inference_mode():
        for pixels, y in loader(ViewDataset(rows, processor, VIEWS), args):
            for index, view in enumerate(VIEWS):
                outputs[view].append(model(pixels[:, index].cuda()).cpu())
            labels.append(y)
            n = sum(len(v) for v in labels)
            if n % 400 == 0 or n == len(rows):
                emit(root, state, samples=n, total=len(rows), elapsed_seconds=time.monotonic() - started)
    return {v: torch.cat(parts) for v, parts in outputs.items()}, torch.cat(labels), time.monotonic() - started


def selected_reference(reference, rows):
    lookup = {s: i for i, s in enumerate(reference['sample_ids'])}
    indices = torch.tensor([lookup[_sample_id(r['video_path'])] for r in rows])
    return {view: values[indices] for view, values in reference['logits'].items()}


def parity(actual, expected, atol=2e-4, rtol=1e-5):
    error = float((actual - expected).abs().max())
    if not torch.equal(actual.argmax(-1), expected.argmax(-1)) or not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        raise RuntimeError(f'Logit parity failed: max error {error}')
    return error


def snapshot_frozen(model):
    return {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}


def verify_frozen(model, before):
    if any(p.grad is not None for p in model.parameters()) or any(not torch.equal(t.detach().cpu(), before[n]) for n, t in model.state_dict().items()):
        raise RuntimeError('Frozen teacher/anchor state changed')


def prealign(model, dataset, audit_dataset, args, root):
    started = time.monotonic()
    features, labels = [], []
    # Only paired train samples are supplied here; no validation target is used.
    with torch.inference_mode():
        for pixels, y, _, valid in loader(dataset, args):
            if not valid.all():
                raise ValueError('Prealignment requires paired train')
            _, pooled = model(pixels.cuda(), return_features=True)
            features.append(pooled.cpu()); labels.append(y)
            n = sum(len(v) for v in labels)
            if n % 400 == 0 or n == len(dataset):
                emit(root, 'prealignment_feature_extraction', samples=n, total=len(dataset))
    audit_features, audit_labels = [], []
    with torch.inference_mode():
        for pixels, y, _, _ in loader(audit_dataset, args):
            _, pooled = model(pixels.cuda(), return_features=True)
            audit_features.append(pooled.cpu()); audit_labels.append(y)
    features, labels = torch.cat(features), torch.cat(labels)
    heldout, heldout_labels = torch.cat(audit_features).cuda(), torch.cat(audit_labels).cuda()
    atomic_save({'features': features, 'labels': labels, 'train_ids': [_sample_id(r['video_path']) for r in dataset.rows],
                 'audit_features': heldout.cpu(), 'audit_labels': heldout_labels.cpu(), 'test_evaluated': False}, root / 'prealignment_features.pt')
    def metric():
        with torch.no_grad():
            scores = model.semantic_logits(heldout)
            return {'ce': float(F.cross_entropy(scores, heldout_labels)),
                    'accuracy': float(scores.argmax(-1).eq(heldout_labels).float().mean())}
    before = metric()
    optimizer = torch.optim.AdamW(model.semantic_projection.parameters(), lr=1e-3, weight_decay=5e-4)
    training = DataLoader(TensorDataset(features, labels), batch_size=128, shuffle=True,
                          generator=torch.Generator().manual_seed(42))
    history = []
    for epoch in range(1, 13):
        for pooled, y in training:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model.semantic_logits(pooled.cuda()), y.cuda())
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite prealignment loss')
            loss.backward(); optimizer.step()
        history.append(metric())
        emit(root, 'prealignment', epoch=epoch, train_samples=len(features), audit=history[-1])
    after = history[-1]
    passed = after['ce'] < before['ce'] and after['accuracy'] > 1 / 59
    result = {'before': before, 'after': after, 'epochs': 12, 'history': history,
              'seconds': time.monotonic() - started, 'status': 'pass' if passed else 'failed'}
    write_json(root / 'prealignment_report.json', result)
    if not passed:
        raise RuntimeError('Held-out prealignment gate failed')
    return result


def export_rgb(model, method, source, pixels, root, filename):
    # A disposable copy avoids merging weights into the continuing training model.
    with torch.inference_mode():
        expected = model(pixels).cpu()
    merged = copy.deepcopy(model).eval()
    merge_lora(merged)
    state = merged_base_state_dict(merged)
    if method == 'semantic':
        state = {n: t for n, t in state.items() if not n.startswith(('semantic_projection.', 'text_prototypes', 'semantic_scale'))}
    if method == 'spatial':
        state = {n: t for n, t in state.items() if not n.startswith('spatial_head.')}
    del merged
    clean = build_candidate(method, source, clean=True).cuda().eval()
    clean.load_state_dict(state, strict=True)
    with torch.inference_mode():
        actual = clean(pixels).cpu()
    error = parity(actual, expected, atol=1e-5, rtol=1e-5)
    timings = {}
    for name, candidate in (('adapted', model), ('merged_rgb', clean)):
        with torch.inference_mode():
            candidate(pixels)
            torch.cuda.synchronize()
            start = time.monotonic()
            for _ in range(3):
                candidate(pixels)
            torch.cuda.synchronize()
            timings[name] = (time.monotonic() - start) / 3
    if method == 'temporal':
        # Same merged backbone and classifier, bypassing only the RGB gate.
        # This measures the branch's deployment cost without allocating a
        # second backbone or changing the exported architecture.
        with torch.inference_mode():
            XCLIPVisionClassifier.forward(clean, pixels)
            torch.cuda.synchronize()
            start = time.monotonic()
            for _ in range(3):
                XCLIPVisionClassifier.forward(clean, pixels)
            torch.cuda.synchronize()
            timings['merged_rgb_without_temporal_gate'] = (time.monotonic() - start) / 3
    atomic_save({'state_dict': state, 'architecture': 'temporal_rgb' if method == 'temporal' else 'plain_rgb',
                 'student_model': STUDENT, 'test_evaluated': False, 'parity_max_error': error}, root / filename)
    result = {'max_error': error, 'rgb_parameters': sum(p.numel() for p in clean.parameters()),
              'extra_deployment_parameters': sum(p.numel() for p in clean.temporal_gate.parameters()) if method == 'temporal' else 0,
              'seconds_per_batch': timings, 'batch_size': len(pixels)}
    result['temporal_gate_seconds_per_batch_delta'] = (timings['merged_rgb'] - timings['merged_rgb_without_temporal_gate']) if method == 'temporal' else 0.
    del clean
    torch.cuda.empty_cache()
    return result


def auxiliary_gradients(model, dataset, args, method, teacher, root):
    batch = next((b for b in loader(dataset, args) if b[3].any()), None)
    if batch is None:
        raise RuntimeError('No valid auxiliary samples in smoke')
    pixels, y, target, valid = [value.cuda() for value in batch]
    model.zero_grad(set_to_none=True)
    _, aux = forward_aux(model, pixels, method)
    auxiliary(aux, y, target, valid, method, teacher).backward()
    prefixes = ['vision_model.'] if method == 'native' else ['vision_model.', 'semantic_projection.' if method == 'semantic' else 'temporal_gate.' if method == 'temporal' else 'spatial_head.']
    if method == 'semantic':
        prefixes.append('mit.')
    gradient = {prefix: sum(float(p.grad.abs().sum()) for n, p in model.named_parameters() if n.startswith(prefix) and p.grad is not None)
                for prefix in prefixes}
    # Zero final temporal scorer blocks vision gradients at initialization. Take
    # one AUX-only step on a disposable model, then audit upstream gradients.
    if method == 'temporal' and gradient['vision_model.'] == 0:
        probe = copy.deepcopy(model)
        opt = torch.optim.AdamW([p for p in probe.parameters() if p.requires_grad], lr=2.25e-5, weight_decay=5e-4)
        opt.zero_grad(set_to_none=True)
        _, probe_aux = forward_aux(probe, pixels, method)
        auxiliary(probe_aux, y, target, valid, method, teacher).backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        _, aux = forward_aux(probe, pixels, method)
        auxiliary(aux, y, target, valid, method, teacher).backward()
        gradient['vision_model.'] = sum(float(p.grad.abs().sum()) for n, p in probe.named_parameters() if n.startswith('vision_model.') and p.grad is not None)
        del probe
    if not all(value > 0 and math_isfinite(value) for value in gradient.values()):
        raise RuntimeError(f'Auxiliary gradient failure: {gradient}')
    if method == 'native':
        from vlm.training.patch16_native_runtime import audit_native_q_gradient
        gradient.update(audit_native_q_gradient(model))
    model.zero_grad(set_to_none=True)
    # Empty supervision must remain differentiable and permit task updates.
    _, aux = forward_aux(model, pixels, method)
    empty = torch.zeros_like(valid)
    zeros = torch.zeros_like(target)
    loss = auxiliary(aux, y, zeros, empty, method, teacher)
    if float(loss.detach()) != 0 or not loss.requires_grad:
        raise RuntimeError('Empty auxiliary contract failed')
    loss.backward(); model.zero_grad(set_to_none=True)
    logits = model(pixels)
    F.cross_entropy(logits, y, label_smoothing=.05).backward()
    task_gradient = {prefix: sum(float(p.grad.abs().sum()) for n, p in model.named_parameters()
                                 if n.startswith(prefix) and p.grad is not None)
                     for prefix in ('vision_model.', 'mit.', 'classifier.')}
    if not all(value > 0 and math_isfinite(value) for value in task_gradient.values()):
        raise RuntimeError(f'Task gradient failure: {task_gradient}')
    # A deliberately perturbed target checks that retention updates the student,
    # while detach prevents the target from receiving any gradient.
    model.zero_grad(set_to_none=True)
    logits = model(pixels)
    target_scores = logits.detach().clone().requires_grad_(True)
    with torch.no_grad():
        target_scores[:, 0].add_(1)
    score_kl(logits, target_scores).backward()
    retention_gradient = {prefix: sum(float(p.grad.abs().sum()) for n, p in model.named_parameters()
                                      if n.startswith(prefix) and p.grad is not None)
                          for prefix in ('vision_model.', 'mit.', 'classifier.')}
    if target_scores.grad is not None or not all(value > 0 and math_isfinite(value) for value in retention_gradient.values()):
        raise RuntimeError(f'Retention gradient failure: {retention_gradient}')
    model.zero_grad(set_to_none=True)
    write_json(root / 'auxiliary_gradients.json', gradient)
    write_json(root / 'task_retention_gradients.json', {'task': task_gradient, 'retention': retention_gradient})
    return pixels


def math_isfinite(value):
    return bool(np.isfinite(value))


def target_audit(rows, paired, cache, pixels_cache, root, method):
    from PIL import Image, ImageDraw
    lookup = {r['sample_id']: r for r in paired}
    records = []
    for index, row in enumerate(rows):
        sample = _sample_id(row['video_path'])
        payload = cache.load(lookup[sample])
        destination = root / 'target_audit' / sample
        destination.mkdir(parents=True, exist_ok=True)
        atomic_save(payload, destination / 'target.pt')
        if method in ('spatial', 'native'):
            pixels = pixels_cache.load(Path(row['video_path']))
            mean = torch.tensor(pixels_cache.processor.image_processor.image_mean).view(1, 3, 1, 1)
            std = torch.tensor(pixels_cache.processor.image_processor.image_std).view(1, 3, 1, 1)
            for slot, frame in enumerate(((pixels * std + mean).clamp(0, 1) * 255).byte()):
                image = Image.fromarray(frame.permute(1, 2, 0).numpy())
                draw = ImageDraw.Draw(image)
                for x, y in payload['coordinates'][slot]:
                    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill='red')
                image.save(destination / f'frame_{slot}.png')
        else:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 1, figsize=(10, 4))
            if 'energy' in payload:
                axes[0].plot(payload['energy'].numpy())
                axes[0].plot(payload['smoothed_energy'].numpy())
                for frame in payload['identity']['frame_indices']:
                    axes[0].axvline(frame, color='gray', alpha=.3)
            axes[1].bar(range(8), payload['target'].numpy())
            fig.suptitle(f'{sample}: {payload["reason"] or "valid"}')
            fig.tight_layout(); fig.savefig(destination / 'motion.png'); plt.close(fig)
            pixels = pixels_cache.load(Path(row['video_path']))
            mean = torch.tensor(pixels_cache.processor.image_processor.image_mean).view(1, 3, 1, 1)
            std = torch.tensor(pixels_cache.processor.image_processor.image_std).view(1, 3, 1, 1)
            strip = torch.cat(list(((pixels * std + mean).clamp(0, 1) * 255).byte()), dim=-1)
            Image.fromarray(strip.permute(1, 2, 0).numpy()).save(destination / 'rgb_strip.png')
        records.append({'sample_id': sample, 'valid': torch.as_tensor(payload['valid']).tolist(),
                        'reason': payload['reason'], 'identity': payload['identity'],
                        'entropy': payload.get('entropy')})
        if (index + 1) % 32 == 0:
            emit(root, 'target_audit', samples=index + 1, total=len(rows))
    summary = {'samples': len(rows), 'valid_samples': sum(bool(torch.as_tensor(r['valid']).any()) for r in records),
               'invalid_reasons': dict(__import__('collections').Counter(r['reason'] for r in records if r['reason'])),
               'rows': records}
    write_json(root / 'target_audit.json', summary)
    return summary


def alignment_audit(paired, processor, root):
    """Reproject the existing manual landmarks against current raw sources."""
    import av
    import math
    from vlm.phase4_alignment import color_joints, transform_point
    review = json.loads(Path('docs/skeletonclip-phase4-gate0-visual-review.json').read_text())
    findings = review['findings']
    if findings['systematic_signed_offset_observed'] or findings['middle_frame_overlays_on_visible_people'] != findings['middle_frame_overlays_reviewed']:
        raise RuntimeError('Prior raw alignment review failed')
    annotations = json.loads(Path('docs/skeletonclip-phase4-gate0-landmarks.json').read_text())['annotations']
    lookup = {r['sample_id']: r for part in ('train', 'val') for r in paired[part]}
    errors, moving = [], {shift: [] for shift in range(-3, 4)}
    records = []
    for annotation in annotations:
        row = lookup[annotation['sample_id']]
        with av.open(row['rgb_path']) as container:
            stream = container.streams.video[0]
            width, height = stream.width, stream.height
        frame = annotation['frame_index']
        _, raw = color_joints(Path(row['skeleton_path']), {max(0, frame + shift) for shift in moving})
        for body_id, joint_id, x, y in annotation['landmarks']:
            shifted = {}
            for shift in moving:
                body = next((b for b in raw.get(frame + shift, []) if b['body_id'] == body_id), None)
                if body and body['joints'][joint_id]['state'] == 2:
                    joint = body['joints'][joint_id]
                    projected = transform_point(joint['x'], joint['y'], width, height)
                    shifted[shift] = math.hypot(projected[0] - x, projected[1] - y)
            if 0 not in shifted or not math.isfinite(shifted[0]):
                raise RuntimeError('Manual landmark missing in current raw skeleton')
            errors.append(shifted[0])
            records.append({'sample_id': row['sample_id'], 'frame': frame, 'body_id': body_id,
                            'joint': joint_id, 'error_px': shifted[0]})
            # Large shift sensitivity is evidence that this landmark tests timing.
            if len(shifted) == 7 and max(shifted.values()) - min(shifted.values()) >= 4:
                for shift, error in shifted.items():
                    moving[shift].append(error)
    median = float(np.median(errors))
    shift_medians = {shift: float(np.median(values)) for shift, values in moving.items() if values}
    passed = median <= 16 and len(shift_medians) == 7 and shift_medians[0] <= min(shift_medians.values())
    result = {'median_error_px': median, 'shifted_medians': shift_medians, 'rows': records,
              'processor': processor.image_processor.to_dict(), 'status': 'pass' if passed else 'failed'}
    write_json(root / 'raw_alignment_report.json', result)
    if not passed:
        raise RuntimeError('Current raw alignment did not satisfy the landmark/timing audit')


def mechanism_screen(model, training, heldout, args, method, teacher, root):
    def select(dataset):
        chosen = []
        sample_ids = []
        for index in range(len(dataset)):
            record = dataset[index]
            if record[3].any():
                chosen.append(record)
                sample_ids.append(_sample_id(dataset.rows[index]['video_path']))
            if len(chosen) == 32:
                return tuple(torch.stack([r[col] if isinstance(r[col], torch.Tensor) else torch.tensor(r[col]) for r in chosen]) for col in range(4)), sample_ids
        raise RuntimeError('Fewer than 32 valid mechanism screen samples')
    train, train_ids = select(training)
    audit, audit_ids = select(heldout)
    # Fixed cyclic derangement after seeded permutation: no self-targets.
    order = torch.randperm(32, generator=torch.Generator().manual_seed(42))
    permutation = torch.empty(32, dtype=torch.long)
    permutation[order] = order.roll(1)
    results = {}
    for arm in ('correct', 'shuffled'):
        candidate = copy.deepcopy(model)
        opt = torch.optim.AdamW([p for p in candidate.parameters() if p.requires_grad], lr=2.25e-5, weight_decay=5e-4)
        for step in range(8):
            indices = slice(step * 4, (step + 1) * 4)
            pixels, labels = train[0][indices].cuda(), train[1][indices].cuda()
            sources = torch.arange(32)[indices] if arm == 'correct' else permutation[indices]
            target, valid = train[2][sources].cuda(), train[3][sources].cuda()
            opt.zero_grad(set_to_none=True)
            _, aux = forward_aux(candidate, pixels, method)
            loss = auxiliary(aux, labels, target, valid, method, teacher)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1, error_if_nonfinite=True)
            opt.step()
        losses = []
        with torch.inference_mode():
            for start in range(0, 32, 4):
                batch = [v[start:start + 4].cuda() for v in audit]
                _, aux = forward_aux(candidate, batch[0], method)
                losses.append(float(auxiliary(aux, batch[1], batch[2], batch[3], method, teacher)))
        results[arm] = sum(losses) / len(losses)
        del candidate
        torch.cuda.empty_cache()
    result = {'heldout_loss': results, 'steps': 8, 'train_samples': 32, 'audit_samples': 32,
              'train_ids': train_ids, 'audit_ids': audit_ids,
              'derangement': permutation.tolist(), 'status': 'pass' if results['correct'] < results['shuffled'] else 'failed'}
    write_json(root / 'mechanism_screen.json', result)
    if result['status'] != 'pass':
        raise RuntimeError(f'Correct target did not beat shuffled target: {results}')
    return result


def diagnostics(model, dataset, args, method, root):
    if method == 'native':
        from vlm.training.patch16_native_runtime import native_diagnostics
        return native_diagnostics(model, dataset, args, root)
    if method in ('semantic', 'ce_retention'):
        return {}
    records, causal = [], []
    offset = 0
    with torch.inference_mode():
        for pixels, labels, target, valid in loader(dataset, args):
            logits, aux = forward_aux(model, pixels.cuda(), method)
            for index in range(len(labels)):
                row = {'sample_id': _sample_id(dataset.rows[offset + index]['video_path']),
                       'valid': valid[index].tolist(), 'target': target[index].tolist(), 'prediction': aux[index].cpu().tolist()}
                if method == 'temporal' and valid[index]:
                    weights = aux[index].cpu()
                    row.update(entropy=float(-(weights * weights.clamp_min(1e-8).log()).sum()),
                               target_kl=float(temporal_loss(weights[None], target[index:index + 1], valid[index:index + 1])))
                    if len(causal) < 32:
                        sample = pixels[index:index + 1].cuda()
                        baseline = float(logits[index, labels[index]])
                        selections = {'highest': int(weights.argmax()), 'lowest': int(weights.argmin()),
                                      'random': int(torch.randint(8, (), generator=torch.Generator().manual_seed(42 + len(causal))))}
                        effects = {}
                        for name, frame in selections.items():
                            changed = sample.clone(); changed[:, frame] = 0
                            effects[name] = baseline - float(model(changed)[0, labels[index]])
                        effects['reverse_order'] = baseline - float(model(sample.flip(1))[0, labels[index]])
                        causal.append({'sample_id': row['sample_id'], 'logit_drop': effects})
                elif method == 'spatial' and valid[index].any():
                    scores, truth = aux[index].cpu()[valid[index]], target[index][valid[index]]
                    ranked = truth.flatten()[scores.flatten().argsort(descending=True)]
                    precision = ranked.cumsum(0) / torch.arange(1, len(ranked) + 1)
                    row['patch_ap'] = float((precision * ranked).sum() / ranked.sum().clamp_min(1))
                    recalls, center_recalls = [], []
                    center = -torch.tensor([(r - 6.5) ** 2 + (c - 6.5) ** 2 for r in range(14) for c in range(14)])
                    for score, occupied in zip(scores, truth):
                        k = int(occupied.sum())
                        recalls.append(float(occupied[score.topk(k).indices].sum() / k))
                        center_recalls.append(float(occupied[center.topk(k).indices].sum() / k))
                    row['occupancy_topk_recall'] = sum(recalls) / len(recalls)
                    row['center_prior_recall'] = sum(center_recalls) / len(center_recalls)
                records.append(row)
            offset += len(labels)
    result = {'samples': len(records), 'rows': records, 'causal': causal}
    write_json(root / 'mechanism_diagnostics.json', result)
    return result


def checkpoint(model, optimizer, scheduler, epoch, training, history, lock, scaler=None):
    parameters = dict(model.named_parameters())
    return {'state_dict': {n: t.detach().cpu() for n, t in model.state_dict().items() if n in parameters and parameters[n].requires_grad},
            'lora_state_dict': lora_state_dict(model), 'classifier_state_dict': model.classifier.state_dict(),
            'grad_scaler': scaler.state_dict() if scaler is not None else None,
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch,
            'history': history, 'identity': lock, 'data_generator_state': training.generator.get_state(),
            'torch_rng_state': torch.get_rng_state(), 'cuda_rng_states': torch.cuda.get_rng_state_all(),
            'python_rng_state': random.getstate(), 'numpy_rng_state': np.random.get_state()}


def resume_selection(history, baseline, mode):
    best, qualified = {}, {'whole': None, 'ensemble': None}
    for epoch, metrics in enumerate(history, 1):
        for view in ('whole', 'ensemble'):
            candidate = {**metrics[view], 'epoch': epoch}
            if view not in best or candidate['correct'] > best[view]['correct']:
                best[view] = candidate
            if mode == 'pilot' and promising(candidate, baseline[view]) and (qualified[view] is None or candidate['correct'] > qualified[view]['correct']):
                qualified[view] = candidate
    return best, qualified


def validate_resume(state, lock):
    """Only completed epochs with the same data and training recipe may resume."""
    previous = state['identity']
    ignored = {'source_sha256', 'hostname', 'gpu', 'started_at', 'resume_from', 'resume_sha256'}
    for key in set(previous) | set(lock):
        if key not in ignored and previous.get(key) != lock.get(key):
            raise ValueError(f'Resume identity differs: {key}')
    def hashes(identity):
        return {str(Path(path).resolve()): digest for path, digest in identity['source_sha256'].items()}
    old, current = hashes(previous), hashes(lock)
    runner = str(Path(__file__).resolve())
    if {k: v for k, v in old.items() if k != runner} != {k: v for k, v in current.items() if k != runner}:
        raise ValueError('Resume data/model/dependency sources differ; only the audited runner may change')
    if lock.get('precision', 'fp32') == 'amp_fp16' and not state.get('grad_scaler'):
        raise ValueError('AMP resume requires GradScaler state')
    epoch = state['epoch']
    if not isinstance(epoch, int) or not 1 <= epoch <= lock['epochs_cap'] or len(state['history']) != epoch:
        raise ValueError('Resume checkpoint is not a completed epoch')
    steps_per_epoch = (len(lock['train_ids']) + lock['batch_size'] - 1) // lock['batch_size'] if lock['mode'] == 'pilot' else 8
    if state['scheduler']['last_epoch'] != epoch * steps_per_epoch:
        raise ValueError('Resume scheduler position differs from completed epochs')
    for key in ('optimizer', 'data_generator_state', 'torch_rng_state', 'cuda_rng_states', 'python_rng_state', 'numpy_rng_state'):
        if key not in state:
            raise ValueError(f'Resume state missing: {key}')
    return epoch * steps_per_epoch


def restore_training(state, model, optimizer, scheduler, training, scaler=None):
    expected = {n: p for n, p in model.named_parameters() if p.requires_grad}
    weights = state['state_dict']
    if set(weights) != set(expected) or any(weights[n].shape != p.shape for n, p in expected.items()):
        raise ValueError('Resume trainable parameter names/shapes differ')
    with torch.no_grad():
        for name, parameter in expected.items():
            parameter.copy_(weights[name])
    if scaler is not None and scaler.is_enabled():
        scaler.load_state_dict(state['grad_scaler'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    if training.persistent_workers:
        # A live persistent iterator does not draw another worker base seed at
        # epoch reset. Prime the fresh loader before restoring the shared RNG.
        iter(training)
    training.generator.set_state(state['data_generator_state'])
    random.setstate(state['python_rng_state'])
    np.random.set_state(state['numpy_rng_state'])
    torch.set_rng_state(state['torch_rng_state'])
    if state['cuda_rng_states']:
        if len(state['cuda_rng_states']) != torch.cuda.device_count():
            raise ValueError('Resume CUDA device count differs')
        torch.cuda.set_rng_state_all(state['cuda_rng_states'])


def inherit_selection(source_root, root, best, qualified, model, method, source, pixels):
    """Carry earlier winners into a fresh continuation without overwriting history."""
    for group, choices in (('best', best), ('best_promising', qualified)):
        for view, metrics in choices.items():
            if metrics is None:
                continue
            name = f'{group}_{view}'
            path = source_root / f'{name}.pt'
            saved = torch.load(path, map_location='cpu', weights_only=False) if path.exists() else {}
            fallback = saved.get('epoch') != metrics['epoch']
            if fallback:
                path = source_root / f'epoch{metrics["epoch"]:02d}/checkpoint.pt'
                saved = torch.load(path, map_location='cpu', weights_only=False)
                if saved['epoch'] != metrics['epoch']:
                    raise ValueError(f'Resume winner checkpoint epoch differs: {path}')
            atomic_save(saved, root / f'{name}.pt')
            epoch_root = root / f'epoch{metrics["epoch"]:02d}'
            atomic_save(saved, epoch_root / 'checkpoint.pt')
            export = source_root / f'{name}_rgb.pt'
            if fallback or not export.exists():
                winner = copy.deepcopy(model)
                with torch.no_grad():
                    for n, parameter in winner.named_parameters():
                        if parameter.requires_grad:
                            parameter.copy_(saved['state_dict'][n])
                export_rgb(winner, method, source, pixels, root, f'{name}_rgb.pt')
                del winner
            else:
                shutil.copy2(export, root / export.name)
    write_json(root / 'selection.json', {'best': best, 'best_promising': qualified})


def require_gate(root, name, hashes, method, precision='fp32', anchor_cache=None):
    report = json.loads((root / 'run_report.json').read_text())
    lock = json.loads((root / 'experiment_lock.json').read_text())
    if report['status'] != 'pass' or report['mode'] != name or report['test_evaluated'] is not False or lock['source_sha256'] != hashes or lock['method'] != method:
        raise ValueError(f'Missing passing {name} gate for current sources')
    if lock['precision'] != precision or lock.get('anchor_cache_path') != (str(anchor_cache.resolve()) if anchor_cache else None):
        raise ValueError(f'{name} gate acceleration recipe differs')
    if anchor_cache and lock['anchor_cache_sha256'] != sha256_file(anchor_cache / 'logits.pt'):
        raise ValueError(f'{name} gate anchor cache changed')


def run(args):
    torch.set_num_threads(8)
    set_seed(42)
    root = args.artifacts_dir or Path('artifacts/patch16_skeleton_v1') / args.method / args.mode / 'seed42'
    control_path = root.parents[2] / 'workflow_control.json'
    if args.method != 'semantic' and control_path.exists():
        control = json.loads(control_path.read_text())
        if control.get('method2_auto_start') is False:
            root.mkdir(parents=True, exist_ok=True)
            report = {'status': 'held', 'method': args.method, 'mode': args.mode,
                      'reason': control['reason'], 'test_evaluated': False, 'raw_accuracy_gain': None}
            write_json(root / 'run_report.json', report)
            emit(root, 'held', reason=control['reason'])
            return
    if (root / 'experiment_lock.json').exists():
        raise ValueError('Existing immutable run namespace; choose a fresh --artifacts-dir')
    resume_path = getattr(args, 'resume', None)
    if resume_path and resume_path.name != 'last_checkpoint.pt':
        raise ValueError('Resume requires last_checkpoint.pt with matching prior winners')
    if resume_path and args.mode == 'audit':
        raise ValueError('Audit cannot resume training')
    resume_state = torch.load(resume_path, map_location='cpu', weights_only=False) if resume_path else None
    hashes = source_hashes(args)
    parent = root.parent.parent
    if args.mode in ('smoke', 'pilot'):
        require_gate(args.audit_dir or parent / 'audit/seed42', 'audit', hashes, args.method, args.precision, args.anchor_cache)
    if args.method == 'native' and args.mode in ('smoke', 'pilot'):
        from vlm.training.patch16_native_runtime import require_native_precision
        require_native_precision(args.precision_dir or parent / 'precision/seed42', args)
    if args.mode == 'pilot':
        require_gate(args.smoke_dir or parent / 'smoke/seed42', 'smoke', hashes, args.method, args.precision, args.anchor_cache)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required before creating run lock')
    emit(root, 'source_audit')
    split, paired, skeleton, source, teacher_source, reference, processor, cache = validate_sources(args)
    audit = audit_rows(paired['val'], split['val'])
    train_rows = balanced(split['train'], 128, 42) if args.mode != 'pilot' else split['train']
    val_rows = balanced(split['val'], 120, 42) if args.mode != 'pilot' else split['val']
    if args.method == 'native':
        from vlm.patch16_native_targets import NativeTargetCache
        targets = NativeTargetCache(args.target_cache, args.native_target_cache, processor.image_processor.to_dict())
    elif args.method == 'ce_retention':
        targets = None
    else:
        targets = TargetCache(args.target_cache, args.method, processor.image_processor.to_dict()) if args.method != 'semantic' else None
    dataset = TrainingDataset(train_rows, cache, paired['train'], skeleton['train'], args.method, targets)
    audit_dataset = TrainingDataset(audit, cache, paired['val'], skeleton['val'], args.method, targets)
    cached_logits = None
    if args.anchor_cache:
        from vlm.training.patch16_anchor_cache import cache_identity, load_anchor_cache
        identity, ids, anchor_labels, _ = cache_identity(split['train'], cache, args.adapter, args.split_manifest)
        all_logits = load_anchor_cache(args.anchor_cache, identity, ids, anchor_labels)
        positions = {sample: index for index, sample in enumerate(ids)}
        cached_logits = all_logits[[positions[_sample_id(row['video_path'])] for row in train_rows]]
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid', '--format=csv,noheader'], text=True).strip()
    lock = {'method': args.method, 'mode': args.mode, 'seed': 42, 'source_sha256': hashes, 'student': STUDENT,
            'teacher_text': TEACHER_TEXT, 'cache_identity': TEACHER_TEXT, 'processor': processor.image_processor.to_dict(),
            'target_version': TARGET_VERSION, 'train_ids': [_sample_id(r['video_path']) for r in train_rows],
            'validation_ids': [_sample_id(r['video_path']) for r in val_rows], 'audit_ids': [_sample_id(r['video_path']) for r in audit],
            'batch_size': 4, 'precision': args.precision, 'gradient_accumulation': 1, 'epochs_cap': 10,
            'scheduler_steps': 201600, 'learning_rate': 2.25e-5, 'weight_decay': 5e-4,
            'betas': [.9, .999], 'eps': 1e-8, 'label_smoothing': .05, 'clip': 1,
            'retention_weight': 1, 'retention_temperature': 2, 'workers': args.workers, 'cpu_threads': 8,
            'student_policy': 'eval mode with autograd; upper2 vision/MIT q/v LoRA, classifier and auxiliary head',
            'auxiliary': {'semantic': '0.1 CE59 + 0.01 score-element MSE', 'temporal': '0.1 target-to-prediction KL', 'spatial': '0.1 balanced BCE', 'native': '0.1 native target-to-CLS-patch KL', 'ce_retention': 'disabled; CE120 + original RGB retention only'}[args.method],
            'hostname': socket.gethostname(), 'gpu': gpu, 'started_at': utc(), 'test_evaluated': False}
    lock['cache_paths'] = {'rgb': str(args.rgb_cache), 'skeleton': str(args.skeleton_cache), 'targets': str(args.target_cache)}
    if args.method in ('native', 'ce_retention'):
        lock.update(native_layers=[10, 11], native_heads=[0, 1, 2, 3], native_tokens=198, native_target_version='native-p16-occupancy-gaussian1-uniform20-v1', attention_backend='eager', native_target_cache=str(args.native_target_cache), native_pilot_review_epochs=2)
    if args.method == 'ce_retention':
        lock.update(auxiliary_weight=0, control_epochs=2, scaler_init_scale=256, scaler_growth_interval=1000000)
    if args.anchor_cache:
        lock.update(anchor_cache_path=str(args.anchor_cache.resolve()), anchor_cache_sha256=sha256_file(args.anchor_cache / 'logits.pt'), anchor_cache_identity=identity)
    if args.method == 'ce_retention':
        from vlm.training.patch16_control import audit_native_recipe
        audit_native_recipe(args.native_reference, lock, root)
    if resume_state is not None:
        validate_resume(resume_state, lock)
        lock.update(resume_from=str(resume_path.resolve()), resume_sha256=sha256_file(resume_path))
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'experiment_lock.json').open('x') as stream:
        json.dump(lock, stream, indent=2)
    snapshot_sources(root, hashes)
    started = time.monotonic()
    try:
        model = build_candidate(args.method, source, teacher_source).cuda().eval()
        anchor = build_model('anchor', source).cuda().requires_grad_(False).eval()
        anchor_before = snapshot_frozen(anchor)
        teacher = None
        if args.method == 'semantic':
            teacher = SkeletonPrototypeTeacher('blockgcn', teacher_source['state_dict']['text_prototypes'])
            teacher.load_state_dict(teacher_source['state_dict'], strict=True)
            teacher.cuda().requires_grad_(False).eval()
            teacher_before = snapshot_frozen(teacher)
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        allowed = ('classifier.', 'semantic_projection.', 'temporal_gate.', 'spatial_head.')
        if any(not (n.startswith(allowed) or n.endswith(('lora_a', 'lora_b'))) for n, _ in named):
            raise RuntimeError('Unexpected trainable base parameters')
        write_json(root / 'trainable_parameters.json', {'names': [n for n, _ in named], 'count': sum(p.numel() for _, p in named)})
        initial, labels, seconds = evaluate(model, balanced(split['val'], 120, 42), processor, args, root, 'initial_parity')
        expected = selected_reference(reference, balanced(split['val'], 120, 42))
        errors = {view: parity(initial[view], expected[view]) for view in VIEWS}
        # Complete 472 held-out RGB parity, not only the first batch.
        heldout_logits = []
        with torch.inference_mode():
            for pixels, _, _, _ in loader(audit_dataset, args):
                current = model(pixels.cuda())
                if args.method == 'ce_retention':
                    enabled, maps = forward_aux(model, pixels.cuda(), args.method)
                    parity(enabled.cpu(), current.cpu(), atol=1e-5, rtol=1e-5)
                    if maps.shape[1:] != (8, 2, 198):
                        raise RuntimeError('Control native attention shape differs')
                parity(current.cpu(), anchor(pixels.cuda()).cpu(), atol=1e-5, rtol=1e-5)
                heldout_logits.append(current.cpu())
        errors['audit'] = parity(torch.cat(heldout_logits), selected_reference(reference, audit)['whole'])
        verify_frozen(anchor, anchor_before)
        if teacher:
            verify_frozen(teacher, teacher_before)
        # Verify compact pixels against a fresh canonical processor path.
        for row in [train_rows[0], audit[0]]:
            fresh, _ = ViewDataset([row], processor, ('whole',))[0]
            if not torch.equal(cache.load(Path(row['video_path'])), fresh[0]):
                raise RuntimeError('Processed pixel cache parity failed')
        if cached_logits is not None:
            with torch.inference_mode():
                online = anchor(torch.stack([dataset[i][0] for i in range(4)]).cuda()).cpu()
            errors['cached_anchor'] = parity(online, cached_logits[:4], atol=1e-5, rtol=1e-5)
        write_json(root / 'initial_parity.json', errors)
        if targets:
            target_audit(audit, paired['val'], targets, cache, root, args.method)
            # The spatial baseline's raw color/timing audit is reusable only with
            # identical canonical sources and processor geometry; no index-only pass.
            if args.method in ('spatial', 'native'):
                alignment_audit(paired, processor, root)
        if args.method == 'native':
            from vlm.training.patch16_native_runtime import native_parity_audit
            native_parity_audit(model, anchor, audit_dataset, args, root)
        if args.mode == 'audit':
            if args.method == 'native':
                diagnostics(model, audit_dataset, args, args.method, root)
            report = {'status': 'pass', 'mode': 'audit', 'method': args.method, 'initial_parity': errors,
                      'elapsed_seconds': time.monotonic() - started, 'test_evaluated': False, 'raw_accuracy_gain': 0}
            write_json(root / 'run_report.json', report)
            emit(root, 'complete', status='pass')
            return
        prealignment = None
        if resume_state is not None:
            previous_report = resume_path.parent / 'prealignment_report.json'
            prealignment = json.loads(previous_report.read_text()) if previous_report.exists() else None
            if prealignment is not None:
                write_json(root / 'prealignment_report.json', prealignment)
        elif args.method == 'semantic':
            rgb = {_sample_id(r['video_path']): r for r in train_rows}
            paired_rows = [rgb[r['sample_id']] for r in paired['train'] if r['sample_id'] in rgb]
            alignment = TrainingDataset(paired_rows, cache, paired['train'], skeleton['train'], args.method)
            prealignment = prealign(model, alignment, audit_dataset, args, root)
            with torch.inference_mode():
                unchanged = []
                for pixels, _, _, _ in loader(audit_dataset, args):
                    unchanged.append(model(pixels.cuda()).cpu())
            parity(torch.cat(unchanged), torch.cat(heldout_logits), atol=1e-5, rtol=1e-5)
        elif args.method != 'ce_retention':
            # Balanced smoke dataset need not contain 32 valid samples; screen
            # searches the same deterministic full train order until it has 32.
            screen_rows = balanced(split['train'], len(split['train']), 42)
            screen_dataset = TrainingDataset(screen_rows, cache, paired['train'], skeleton['train'], args.method, targets)
            mechanism_screen(model, screen_dataset, audit_dataset, args, args.method, teacher, root)
        if args.method == 'ce_retention':
            from vlm.training.patch16_control import control_gradients
            pixels = control_gradients(model, dataset, args, root)
        else:
            pixels = auxiliary_gradients(model, dataset, args, args.method, teacher, root)
        parameters = [p for _, p in named]
        optimizer = torch.optim.AdamW(parameters, lr=2.25e-5, weight_decay=5e-4, betas=(.9, .999), eps=1e-8)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=201600)
        training_dataset = CachedAnchorDataset(dataset, cached_logits) if cached_logits is not None else dataset
        if args.method == 'ce_retention':
            from vlm.training.patch16_control import IndexedControlDataset
            training_dataset = IndexedControlDataset(training_dataset)
        training = loader(training_dataset, args, shuffle=True)
        scaler = torch.amp.GradScaler('cuda', enabled=args.precision == 'amp_fp16', init_scale=256, growth_interval=1000000)
        paired_ids = {r['sample_id'] for r in paired['val']}
        val_ids = [_sample_id(r['video_path']) for r in val_rows]
        labels = torch.tensor([int(r['class_id']) - 1 for r in val_rows])
        baseline_logits = selected_reference(reference, val_rows)
        baseline = reports(baseline_logits, labels, val_ids, paired_ids)
        write_json(root / 'baseline_report.json', baseline)
        best, qualified = {}, {'whole': None, 'ensemble': None}
        history, killed, total_steps = [], False, 0
        start_epoch = 1
        native_review_reason = None
        if resume_state is not None:
            total_steps = validate_resume(resume_state, lock)
            history = copy.deepcopy(resume_state['history'])
            best, qualified = resume_selection(history, baseline, args.mode)
            if args.mode == 'pilot' and early_kill(history, baseline):
                raise ValueError('Cannot extend an early-killed experiment')
            inherit_selection(resume_path.parent, root, best, qualified, model, args.method, source, pixels)
            # All preflight work precedes RNG restoration and the first iterator.
            restore_training(resume_state, model, optimizer, scheduler, training, scaler)
            atomic_save(checkpoint(model, optimizer, scheduler, resume_state['epoch'], training, history, lock, scaler), root / 'last_checkpoint.pt')
            start_epoch = resume_state['epoch'] + 1
            write_json(root / 'resume.json', {'checkpoint': str(resume_path.resolve()),
                       'checkpoint_sha256': lock['resume_sha256'], 'completed_epoch': start_epoch - 1,
                       'next_epoch': start_epoch, 'steps': total_steps,
                       'history_artifacts': str(resume_path.parent.resolve()), 'test_evaluated': False})
            emit(root, 'resumed', epoch=start_epoch, steps=total_steps)
        if args.method == 'ce_retention':
            from vlm.training.patch16_control import save_training_boundary, sample_order_loader
            save_training_boundary(model, training, scaler, root)
            if args.mode == 'pilot':
                training = sample_order_loader(training, args.native_reference, root, lock['train_ids'], start_epoch)
        torch.cuda.reset_peak_memory_stats()
        with (root / 'step_losses.jsonl').open('x', buffering=1) as losses:
            for epoch in range(start_epoch, 2 if args.mode == 'smoke' else 3 if args.method in ('native', 'ce_retention') else 11):
                train_start = time.monotonic()
                seen = 0
                skipped_updates = 0
                for step, batch in enumerate(training, 1):
                    if args.mode == 'smoke' and step > 8:
                        break
                    x, y, target, valid = [v.cuda(non_blocking=True) for v in batch[:4]]
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda', dtype=torch.float16, enabled=args.precision == 'amp_fp16'):
                        logits, aux = forward_aux(model, x, args.method)
                    logits, aux = logits.float(), aux.float()
                    with torch.no_grad():
                        anchor_logits = batch[4].cuda(non_blocking=True) if cached_logits is not None else anchor(x)
                    ce = F.cross_entropy(logits, y, label_smoothing=.05)
                    retention = score_kl(logits, anchor_logits)
                    extra = auxiliary(aux, y, target, valid, args.method, teacher)
                    loss = ce + retention + extra
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite training loss')
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    skipped = scaler.get_scale() < scale_before
                    skipped_updates += int(skipped)
                    if skipped:
                        raise RuntimeError('AMP optimizer update skipped; exposure/schedule contract would differ')
                    scheduler.step()
                    total_steps += 1; seen += len(y)
                    losses.write(json.dumps({'epoch': epoch, 'step': total_steps, 'ce120': float(ce.detach()),
                                             'retention': float(retention.detach()), 'auxiliary': float(extra.detach()),
                                             'valid_count': int(valid.sum()), 'scaler_scale': scaler.get_scale(), 'optimizer_update_skipped': skipped, 'lr': scheduler.get_last_lr()[0]}) + '\n')
                    if step % 25 == 0 or args.mode == 'smoke' and step == 8:
                        emit(root, 'training', epoch=epoch, samples=seen, total=len(dataset), steps=total_steps,
                             elapsed_seconds=time.monotonic() - train_start)
                if args.mode == 'pilot' and (seen != 80640 or step != 20160):
                    raise RuntimeError('Full epoch exposure contract failed')
                train_seconds = time.monotonic() - train_start
                verify_frozen(anchor, anchor_before)
                if teacher:
                    verify_frozen(teacher, teacher_before)
                logits, actual_labels, validation_seconds = evaluate(model, val_rows, processor, args, root, f'validation_epoch{epoch}')
                if not torch.equal(labels, actual_labels):
                    raise ValueError('Validation order changed')
                metrics = reports(logits, labels, val_ids, paired_ids, baseline_logits)
                history.append(metrics)
                epoch_root = root / f'epoch{epoch:02d}'
                payload = {'logits': logits, 'probabilities': view_probabilities(logits), 'labels': labels, 'sample_ids': val_ids,
                           'predictions': {k: p.argmax(-1) for k, p in view_probabilities(logits).items()}, 'test_evaluated': False}
                atomic_save(payload, epoch_root / 'view_predictions.pt')
                write_json(epoch_root / 'metrics.json', metrics)
                write_json(epoch_root / 'compute.json', {'train_seconds': train_seconds, 'validation_seconds': validation_seconds,
                                                       'samples': seen, 'examples_per_second': seen / train_seconds,
                                                       'steps': total_steps, 'optimizer_skipped_updates': skipped_updates, 'optimizer_updates': total_steps, 'peak_allocated_vram': torch.cuda.max_memory_allocated()})
                state = checkpoint(model, optimizer, scheduler, epoch, training, history, lock, scaler)
                atomic_save(state, epoch_root / 'checkpoint.pt')
                for view in ('whole', 'ensemble'):
                    if view not in best or metrics[view]['correct'] > best[view]['correct']:
                        best[view] = {**metrics[view], 'epoch': epoch}
                        atomic_save(state, root / f'best_{view}.pt')
                        export_rgb(model, args.method, source, pixels, root, f'best_{view}_rgb.pt')
                    if args.mode == 'pilot' and promising(metrics[view], baseline[view]) and (qualified[view] is None or metrics[view]['correct'] > qualified[view]['correct']):
                        qualified[view] = {**metrics[view], 'epoch': epoch}
                        atomic_save(state, root / f'best_promising_{view}.pt')
                        export_rgb(model, args.method, source, pixels, root, f'best_promising_{view}_rgb.pt')
                write_json(root / 'selection.json', {'best': best, 'best_promising': qualified})
                # Export uses loaders/RNG too: capture the true next-epoch state last.
                state = checkpoint(model, optimizer, scheduler, epoch, training, history, lock, scaler)
                atomic_save(state, root / 'last_checkpoint.pt')
                emit(root, 'epoch_complete', epoch=epoch, metrics={k: v['raw_accuracy'] for k, v in metrics.items()})
                if args.mode == 'pilot' and args.method != 'ce_retention' and early_kill(history, baseline):
                    killed = True
                    break
                if args.method == 'native' and args.mode == 'pilot' and epoch == 2:
                    from vlm.patch16_native_targets import native_budget_decision
                    native_review_reason = native_budget_decision(epoch, qualified)
                    break
        diagnostics(model, audit_dataset, args, args.method, root)
        export = export_rgb(model, args.method, source, pixels, root, 'last_rgb.pt')
        report = {'status': 'pass' if args.mode == 'smoke' else terminal_status(best, qualified, baseline, killed),
                  'method': args.method, 'mode': args.mode, 'best': best, 'best_promising': qualified,
                  'stop_reason': 'early_regression' if killed else 'smoke_cap' if args.mode == 'smoke' else 'two_epoch_control_cap' if args.method == 'ce_retention' else native_review_reason or 'ten_epoch_cap',
                  'epochs': len(history), 'steps': total_steps, 'elapsed_seconds': time.monotonic() - started,
                  'ended_at': utc(), 'prealignment': prealignment, 'rgb_export': export,
                  'raw_accuracy_gain': best['whole']['raw_accuracy'] - baseline['whole']['raw_accuracy'],
                  'test_evaluated': False}
        write_json(root / 'run_report.json', report)
        emit(root, 'complete', status=report['status'])
    except Exception as error:
        report = {'status': 'failed', 'mode': args.mode, 'method': args.method, 'error': repr(error),
                  'test_evaluated': False, 'raw_accuracy_gain': None, 'ended_at': utc()}
        write_json(root / 'run_report.json', report)
        emit(root, 'failed', error=repr(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=('semantic', 'temporal', 'spatial', 'native', 'ce_retention'), required=True)
    parser.add_argument('--mode', choices=('audit', 'smoke', 'pilot'), required=True)
    parser.add_argument('--artifacts-dir', type=Path)
    parser.add_argument('--resume', type=Path, help='Completed-epoch checkpoint; use a fresh artifacts directory and current audit/smoke gates')
    parser.add_argument('--audit-dir', type=Path)
    parser.add_argument('--smoke-dir', type=Path)
    parser.add_argument('--precision-dir', type=Path)
    parser.add_argument('--native-reference', type=Path, default=Path('artifacts/patch16_skeleton_v1/native_attention_01/native/pilot/seed42'))
    parser.add_argument('--native-target-cache', type=Path, default=Path('cache/patch16_native_targets_v1'))
    parser.add_argument('--precision', choices=('fp32', 'amp_fp16'), default='fp32')
    parser.add_argument('--anchor-cache', type=Path, help='Complete, provenance-verified FP32 train anchor cache')
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--adapter', type=Path, default=BASE / 'best_adapter.pt')
    parser.add_argument('--baseline-logits', type=Path, default=BASE / 'best_validation_logits.pt')
    parser.add_argument('--baseline-lock', type=Path, default=BASE / 'run_lock.json')
    parser.add_argument('--baseline-views', type=Path, default=Path('artifacts/patch16_resolution/whole_zoom_validation/view_predictions.pt'))
    parser.add_argument('--teacher', type=Path, default=Path('artifacts/skeletonclip_gate3/xclip_seed42/best_checkpoint.pt'))
    parser.add_argument('--split-manifest', type=Path, default=Path('artifacts/xclip/selected_split.json'))
    parser.add_argument('--paired-manifest', type=Path, default=Path('artifacts/skeleton_data/paired_manifest.json'))
    parser.add_argument('--skeleton-cache', type=Path, default=Path('cache/skeletonclip_gate3'))
    parser.add_argument('--rgb-cache', type=Path, default=Path('cache/skeletonclip_phase3_rgb_uint8'))
    parser.add_argument('--target-cache', type=Path, default=Path('cache/patch16_skeleton_targets_v1'))
    args = parser.parse_args()
    if args.method in ('native', 'ce_retention') and (args.anchor_cache is None or args.precision != 'amp_fp16'):
        parser.error('Native requires --anchor-cache and the fixed --precision amp_fp16 recipe')
    if args.workers < 0:
        parser.error('workers must be nonnegative')
    args.workers = min(args.workers, 24)
    run(args)


if __name__ == '__main__':
    main()
