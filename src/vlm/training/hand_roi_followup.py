"""Expanded RGB interaction evaluation and bounded local-expert fine-tuning."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import av
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import XCLIPModel, XCLIPProcessor
from vlm.experiments.video_sampling import read_video_pyav, uniform_frame_indices
from vlm.models.lora import load_expanded_lora_state_dict, load_lora_state_dict, lora_state_dict, merge_lora, merged_base_state_dict
from vlm.models.xclip_vision_lora import XCLIPVisionClassifier, XCLIPVisionLoRAClassifier
from vlm.reporting import write_json
from vlm.skeleton_teacher_cache import sha256_file
from vlm.training.rgb_input_audit import PRODUCT_EXCLUDED_IDS, ViewDataset, action_names
from vlm.training.ski_semantic import MODEL, MODEL_REVISION

ADAPTER = Path('artifacts/skeletonclip_phase3/vision_batch16_workers24_uint8_seed42/vision_lora_ce_seed42/best_adapter.pt')
REFERENCE = Path('artifacts/whole_zoom_validation_v1/view_predictions.pt')
SPLIT = Path('artifacts/xclip/selected_split.json')
PRIOR = Path('artifacts/rgb_hand_roi_audit_v1/view_predictions.pt')
FOCUS = (2, 3, 13, 19, 33, 37, 39, 41, 45, 47, 49)
TARGETS = (2, 41, 49)


def sample_id(row):
    return Path(row['video_path']).stem.removesuffix('_rgb')


def load_model(source, *, upper_blocks=2, expand=False, device="cuda", vision_layout="upper"):
    if vision_layout not in ("upper", "middle4", "mixed4") or (vision_layout != "upper" and upper_blocks != 4):
        raise ValueError("Alternative layouts require four trainable vision blocks")
    middle = vision_layout == "middle4"
    if upper_blocks not in (2, 4) or (expand and upper_blocks != 4):
        raise ValueError("Expected upper2 reload or upper4 expansion/reload")
    model = XCLIPVisionLoRAClassifier(
        XCLIPModel.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=True),
        num_classes=120, head_dropout=.25, feature_noise_std=.01,
        upper_blocks=6 if middle else upper_blocks,
        vision_block_indices=(6, 7, 10, 11) if vision_layout == "mixed4" else None, lora_rank=16, lora_alpha=32, lora_dropout=.05).to(device)
    if expand:
        added = [f'vision_model.encoder.layers.{block}.self_attn.{projection}'
                 for block in ((6, 7, 8, 9) if middle else (6, 7) if vision_layout == "mixed4" else (8, 9)) for projection in ('q_proj', 'v_proj')]
        if model.target_modules != added + source['model']['target_modules']:
            raise ValueError('Added blocks differ from the declared vision layout')
        load_expanded_lora_state_dict(model, source['model']['lora_state_dict'],
                                     source_targets=source['model']['target_modules'], added_targets=added)
    else:
        if model.target_modules != source['model']['target_modules']:
            raise ValueError('Adapter target modules differ')
        load_lora_state_dict(model, source['model']['lora_state_dict'])
    model.classifier.load_state_dict(source['model']['classifier_state_dict'], strict=True)
    if middle:
        for block in (10, 11):
            for projection in ('q_proj', 'v_proj'):
                model.vision_model.encoder.layers[block].self_attn.get_submodule(projection).requires_grad_(False)
    return model.eval()


class InteractionDataset(Dataset):
    def __init__(self, rows, crops, processor):
        self.rows, self.crops, self.processor = rows, crops, processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, position):
        row = self.rows[position]
        crop = self.crops[sample_id(row)]
        path = Path(row['video_path'])
        if sha256_file(path) != crop['rgb_sha256']:
            raise RuntimeError('RGB source changed after localization')
        with av.open(str(path)) as container:
            container.streams.video[0].codec_context.thread_count = 2
            indices = uniform_frame_indices(8, container.streams.video[0].frames)
            if indices.tolist() != crop['frame_indices']:
                raise RuntimeError('RGB frame indices changed')
            frames = read_video_pyav(container, indices)
        if crop['fallback'] is None:
            height, width = frames.shape[1:3]
            left, top, right, bottom = crop['boxes']['interaction']
            if not (0 <= left < right <= width and 0 <= top < bottom <= height and right-left == bottom-top):
                raise RuntimeError('Invalid interaction crop')
            frames = frames[:, top:bottom, left:right]
        pixels = self.processor(images=list(frames), return_tensors='pt')['pixel_values'].squeeze(0)
        return pixels, int(row['class_id']) - 1


def make_loader(rows, crops, processor, args, shuffle=False):
    return DataLoader(InteractionDataset(rows, crops, processor), batch_size=args.batch_size,
                      num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
                      shuffle=shuffle, generator=torch.Generator().manual_seed(42))


def evaluate_metrics(logits, reference, selected, labels, ids, prior_ids):
    probabilities = {name: reference['logits'][name][selected].softmax(-1) for name in ('whole', 'center_zoom70')}
    probabilities['whole_zoom_mean'] = (probabilities['whole'] + probabilities['center_zoom70']) / 2
    probabilities['interaction'] = logits.softmax(-1)
    probabilities['whole_zoom_interaction_mean'] = (
        probabilities['whole'] + probabilities['center_zoom70'] + probabilities['interaction']) / 3
    baseline = probabilities['whole_zoom_mean'].argmax(-1).eq(labels)
    heldout = torch.tensor([sid not in prior_ids for sid in ids])
    retained = ~torch.isin(labels, torch.tensor(PRODUCT_EXCLUDED_IDS)-1)
    targets = torch.isin(labels, torch.tensor(TARGETS)-1)
    focus = torch.isin(labels, torch.tensor(FOCUS)-1)
    attention = torch.isin(labels, torch.arange(41, 48))
    masks = {'all': torch.ones(len(labels), dtype=torch.bool), 'retained86': retained,
             'targets': targets, 'focus_negatives': focus & ~targets, 'other_retained': retained & ~focus,
             'excluded34_negatives': ~retained, 'attention_A042_A048': attention}
    def summarize(mask):
        result = {}
        for view, p in probabilities.items():
            ok = p.argmax(-1).eq(labels)
            n = int(mask.sum())
            result[view] = {'samples': n, 'correct': int(ok[mask].sum()),
                            'accuracy': float(ok[mask].double().mean()) if n else None,
                            'fixed': int((mask & ok & ~baseline).sum()), 'regressed': int((mask & ~ok & baseline).sum())}
        return result
    report = {'all_selected': {name: summarize(mask) for name, mask in masks.items()},
              'new_samples_only': {name: summarize(mask & heldout) for name, mask in masks.items()},
              'prior176_only': {name: summarize(mask & ~heldout) for name, mask in masks.items()},
              'per_class': {f'A{cls:03d}': summarize(labels.eq(cls-1)) for cls in range(1, 121)}}
    t = report['new_samples_only']['targets']['whole_zoom_interaction_mean']
    r = report['new_samples_only']['retained86']['whole_zoom_interaction_mean']
    report['training_gate'] = {'passed': t['fixed'] > t['regressed'] and r['fixed'] >= r['regressed'],
                               'policy': 'Positive target net gain on new samples and nonnegative retained86 net gain; exploratory go/no-go, not significance or adoption',
                               'new_target_net': t['fixed']-t['regressed'], 'new_retained_net': r['fixed']-r['regressed']}
    return report, probabilities


def compare_local_experts(probabilities, labels, ids, previous, diagnostic_ids):
    """Paired comparison against the saved upper2 ensemble with all outputs kept."""
    old_ids = previous['sample_ids']
    if len(set(old_ids)) != len(old_ids) or not set(ids) <= set(old_ids):
        raise ValueError('Upper2 comparison ID coverage mismatch')
    lookup = {sid: i for i, sid in enumerate(old_ids)}
    positions = torch.tensor([lookup[sid] for sid in ids])
    if not torch.equal(labels, previous['labels'][positions]):
        raise ValueError('Upper2 comparison labels mismatch')
    old = previous['probabilities']['whole_zoom_interaction_mean'][positions]
    new = probabilities['whole_zoom_interaction_mean']
    if old.shape != (len(ids), 120) or new.shape != old.shape or not torch.isfinite(old).all() or not torch.isfinite(new).all():
        raise ValueError('Expected finite 120-output ensemble probabilities')
    before, after = old.argmax(-1).eq(labels), new.argmax(-1).eq(labels)
    retained = ~torch.isin(labels, torch.tensor(PRODUCT_EXCLUDED_IDS)-1)
    masks = {'all': torch.ones(len(ids), dtype=torch.bool), 'retained86': retained,
             'targets': torch.isin(labels, torch.tensor(TARGETS)-1),
             'neck_A047': labels.eq(46), 'attention_A042_A048': torch.isin(labels, torch.arange(41, 48)),
             'not_in_prior_diagnostic': torch.tensor([sid not in diagnostic_ids for sid in ids]) & retained}
    masks.update({f'A{cls:03d}': labels.eq(cls-1) for cls in range(1, 121) if cls not in PRODUCT_EXCLUDED_IDS})
    return {name: {'samples': int(mask.sum()), 'before_correct': int(before[mask].sum()),
                   'after_correct': int(after[mask].sum()), 'fixed': int((mask & after & ~before).sum()),
                   'regressed': int((mask & ~after & before).sum()),
                   'net': int(after[mask].sum()-before[mask].sum())} for name, mask in masks.items()}


def run(args):
    torch.manual_seed(42)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    upper_blocks = getattr(args, 'upper_blocks', 2)
    vision_layout = getattr(args, 'vision_layout', 'upper')
    middle = vision_layout == 'middle4'
    alternative = vision_layout != 'upper'
    if alternative and upper_blocks != 4:
        raise ValueError('Alternative layouts require --upper-blocks 4')
    if upper_blocks == 4 and args.mode == 'audit':
        raise ValueError('Upper4 is a separately trained expert; use smoke or pilot')
    root = args.artifacts_dir
    if (root/'experiment_lock.json').exists():
        raise ValueError('Use a fresh artifacts directory')
    split = json.loads(SPLIT.read_text())
    source = torch.load(ADAPTER, map_location='cpu', weights_only=False)
    reference = torch.load(REFERENCE, map_location='cpu', weights_only=True)
    prior = torch.load(PRIOR, map_location='cpu', weights_only=True)
    if source['identity']['split_sha256'] != sha256_file(SPLIT) or source['identity']['model_revision'] != MODEL_REVISION or source['identity']['test_evaluated'] is not False:
        raise ValueError('Baseline provenance mismatch')
    selection = json.loads(args.val_selection.read_text())
    val_lookup = {sample_id(r): r for r in split['val']}
    ids = selection['sample_ids']
    if len(set(ids)) != len(ids) or not set(ids) <= set(val_lookup):
        raise ValueError('Invalid validation selection')
    rows = [val_lookup[sid] for sid in ids]
    crop_payload = json.loads(args.val_crops.read_text())
    if crop_payload['localization'] != 'RGB-only MediaPipe PoseLandmarker lite':
        raise ValueError('Non-RGB crop source')
    crops = {r['sample_id']: r for r in crop_payload['samples']}
    if set(crops) != set(ids):
        raise ValueError('Crop IDs differ from selection')
    ref_index = {sid: i for i, sid in enumerate(reference['sample_ids'])}
    selected = torch.tensor([ref_index[sid] for sid in ids])
    labels = torch.tensor([int(row['class_id'])-1 for row in rows])
    if not torch.equal(labels, reference['labels'][selected]):
        raise ValueError('Reference labels mismatch')
    prior_ids = set(prior['sample_ids'])
    hashes = {name: sha256_file(path) for name, path in {
        'runner': Path(__file__), 'adapter': ADAPTER, 'reference': REFERENCE, 'prior': PRIOR,
        'split': SPLIT, 'val_selection': args.val_selection, 'val_crops': args.val_crops}.items()}
    comparison_path = Path('artifacts/hand_roi_followup_v1/pilot/view_predictions.pt')
    if upper_blocks == 4:
        hashes['upper2_predictions'] = sha256_file(comparison_path)
    upper4_comparison_path = Path('artifacts/hand_roi_upper4_v1/pilot/view_predictions.pt')
    if alternative:
        hashes['upper4_predictions'] = sha256_file(upper4_comparison_path)
    middle4_comparison_path = Path('artifacts/hand_roi_middle4_v1/pilot/view_predictions.pt')
    if vision_layout == 'mixed4':
        hashes['middle4_predictions'] = sha256_file(middle4_comparison_path)
    dependency_paths = {'lora': Path('src/vlm/models/lora.py'),
                        'vision_model': Path('src/vlm/models/xclip_vision_lora.py')}
    hashes.update({name: sha256_file(path) for name, path in dependency_paths.items()})
    lock = {'vision_layout': vision_layout, 'upper_blocks': upper_blocks, 'mode': args.mode, 'training': args.mode in ('pilot', 'smoke'), 'test_evaluated': False,
            'sample_ids': ids, 'source_sha256': hashes, 'seed': 42, 'batch_size': args.batch_size,
            'workers': args.workers, 'view': 'interaction only, whole and zoom baseline frozen',
            'ensemble': 'Fixed equal probability mean whole+zoom+interaction',
            'excluded_source_ids': PRODUCT_EXCLUDED_IDS,
            'selection': 'Full72 validation per focus class; fixed4 per other class; separate prior176 and new1052',
            'fallbacks': sum(crops[sid]['fallback'] is not None for sid in ids)}
    train_rows, train_crops = None, None
    if args.mode in ('pilot', 'smoke'):
        expanded = json.loads(args.expanded_report.read_text())
        if not expanded['training_gate']['passed'] or expanded['source_sha256']['val_selection'] != hashes['val_selection'] or expanded['source_sha256']['val_crops'] != hashes['val_crops']:
            raise ValueError('Expanded gate not passed or evaluation changed')
        train_selection = json.loads(args.train_selection.read_text())
        train_payload = json.loads(args.train_crops.read_text())
        if train_payload['localization'] != crop_payload['localization'] or train_payload['pose_model_sha256'] != crop_payload['pose_model_sha256']:
            raise ValueError('Train/val localization mismatch')
        train_lookup = {sample_id(r): r for r in split['train']}
        train_ids = train_selection['sample_ids']
        if len(set(train_ids)) != len(train_ids) or set(train_ids) & set(ids) or not set(train_ids) <= set(train_lookup):
            raise ValueError('Train split/IDs mismatch')
        train_rows = [train_lookup[sid] for sid in train_ids]
        train_crops = {r['sample_id']: r for r in train_payload['samples']}
        if set(train_crops) != set(train_ids):
            raise ValueError('Train crops mismatch')
        lock.update({'train_ids': train_ids, 'epochs': 1, 'optimizer': 'AdamW', 'lr': 1e-5,
                     'weight_decay': 5e-4, 'label_smoothing': .1, 'gradient_clip': 1.0,
                     'schedule': 'cosine one pass', 'whole_zoom_policy': 'Original baseline retained unchanged; train a separate local expert'})
        hashes.update({name: sha256_file(path) for name, path in {'train_selection': args.train_selection,
                       'train_crops': args.train_crops, 'expanded_report': args.expanded_report}.items()})
    if upper_blocks == 4 and args.mode == 'pilot' and (len(train_rows) != 2280 or len(ids) != 1228 or args.batch_size != 4):
        raise ValueError('Upper4 pilot requires 2280 train, 1228 validation, batch4/570 steps')
    if args.mode == 'smoke':
        train_rows = train_rows[:16]
        train_crops = {sample_id(row): train_crops[sample_id(row)] for row in train_rows}
        rows, ids, selected, labels = rows[:16], ids[:16], selected[:16], labels[:16]
        lock.update({'smoke_only': True, 'sample_ids': ids, 'train_ids': [sample_id(row) for row in train_rows]})
    if args.mode == 'pilot':
        smoke = json.loads(args.smoke_report.read_text())
        if (smoke.get('status') != 'smoke_pass' or smoke.get('upper_blocks', 2) != upper_blocks or smoke.get('vision_layout', 'upper') != vision_layout
                or any(smoke['source_sha256'].get(name) != hashes[name]
                       for name in ('runner', 'lora', 'vision_model', 'adapter', 'split', 'val_selection', 'val_crops'))):
            raise ValueError('Pilot requires smoke from this exact runner')
    root.mkdir(parents=True, exist_ok=True)
    write_json(root/'experiment_lock.json', lock)
    (root/'runner_snapshot.py').write_text(Path(__file__).read_text())
    def progress(state, **fields):
        value = {'state': state, 'test_evaluated': False, **fields}
        write_json(root/'progress.json', value)
        print(json.dumps(value), flush=True)
    processor = XCLIPProcessor.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=True)
    for name, path in dependency_paths.items():
        (root/f'{name}_snapshot.py').write_text(path.read_text())
    model = load_model(source, upper_blocks=upper_blocks, expand=upper_blocks == 4, vision_layout=vision_layout)
    lock['target_modules'] = model.target_modules
    lock['vision_train_blocks'] = [6, 7, 8, 9] if middle else [6, 7, 10, 11] if vision_layout == 'mixed4' else list(range(12-upper_blocks, 12))
    lock['vision_frozen_adapter_blocks'] = [10, 11] if middle else []
    lock['rng_policy'] = 'seed42; extra zero-B adapters consume additional initialization/dropout RNG; data shuffle unchanged'
    write_json(root/'experiment_lock.json', lock)
    parity_rows = rows[:args.batch_size]
    pixels, _ = next(iter(DataLoader(ViewDataset(parity_rows, processor, ('whole',)), batch_size=args.batch_size)))
    with torch.inference_mode():
        before = model(pixels[:, 0].cuda()).cpu()
    error = float((before - reference['logits']['whole'][selected[:len(parity_rows)]]).abs().max())
    if error > 1e-3 or not torch.equal(before.argmax(-1), reference['logits']['whole'][selected[:len(parity_rows)]].argmax(-1)):
        raise RuntimeError('Whole baseline warm-start parity failed')
    progress('warm_start_verified', max_abs_error=error)
    train_seen, gradients, added_gradients, steps = 0, {}, {}, 0
    vision_gradients, tail_updates = {}, {}
    mixed_tail = {name: value.detach().cpu().clone() for name, value in model.named_parameters()
                  if vision_layout == 'mixed4' and name.startswith(('vision_model.encoder.layers.10.', 'vision_model.encoder.layers.11.'))
                  and name.endswith(('lora_a', 'lora_b'))}
    frozen_adapters = {name: value.detach().cpu().clone() for name, value in model.named_parameters()
                       if middle and name.startswith(('vision_model.encoder.layers.10.', 'vision_model.encoder.layers.11.'))
                       and name.endswith(('lora_a', 'lora_b'))}
    if args.mode in ('pilot', 'smoke'):
        parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        if any(not (name.startswith('classifier.') or name.endswith(('lora_a', 'lora_b'))) for name, _ in parameters):
            raise RuntimeError('Unexpected trainable base weights')
        write_json(root/'trainable_parameters.json', {'names': [name for name, _ in parameters], 'count': sum(p.numel() for _, p in parameters)})
        if alternative:
            expected_count = json.loads(Path('artifacts/hand_roi_upper4_v1/pilot/trainable_parameters.json').read_text())['count']
            if sum(p.numel() for _, p in parameters) != expected_count:
                raise RuntimeError('Four-block trainable parameter budgets differ')
        loader = make_loader(train_rows, train_crops, processor, args, shuffle=True)
        optimizer = torch.optim.AdamW([p for _, p in parameters], lr=1e-5, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(loader))
        model.train()
        started = time.monotonic()
        for step, (pixels, y) in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(pixels.cuda()), y.cuda(), label_smoothing=.1)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            if step == 1:
                gradients = {family: sum(float(p.grad.abs().sum()) for name, p in parameters if name.startswith(prefix) and p.grad is not None)
                             for family, prefix in [('vision', 'vision_model.'), ('mit', 'mit.'), ('classifier', 'classifier.')]}
                if not all(math_value > 0 and np.isfinite(math_value) for math_value in gradients.values()):
                    raise RuntimeError('Missing/nonfinite gradient family')
                vision_gradients = {target: float(dict(model.named_parameters())[target+'.lora_b'].grad.abs().sum())
                                    for target in model.target_modules if target.startswith('vision_model.')
                                    and dict(model.named_parameters())[target+'.lora_b'].requires_grad}
                if not all(value > 0 and np.isfinite(value) for value in vision_gradients.values()):
                    raise RuntimeError('Missing/nonfinite trainable vision target gradients')
                if upper_blocks == 4:
                    added_gradients = {target: float(dict(model.named_parameters())[target+'.lora_b'].grad.abs().sum())
                                       for target in model.target_modules[:8 if middle else 4]}
                    if not all(value > 0 and np.isfinite(value) for value in added_gradients.values()):
                        raise RuntimeError('Missing/nonfinite added vision q/v gradients')
                if middle and any(dict(model.named_parameters())[name].grad is not None for name in frozen_adapters):
                    raise RuntimeError('Frozen blocks10/11 received gradients')
            torch.nn.utils.clip_grad_norm_([p for _, p in parameters], 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            train_seen += len(y)
            steps = step
            if step % 25 == 0 or step == len(loader):
                progress('local_expert_training', step=step, steps=len(loader), samples=train_seen,
                         loss=float(loss.detach()), seconds=time.monotonic()-started)
        if train_seen != len(train_rows):
            raise RuntimeError('Training sample coverage mismatch')
        if any(not torch.equal(dict(model.named_parameters())[name].detach().cpu(), original) for name, original in frozen_adapters.items()):
            raise RuntimeError('Frozen blocks10/11 adapters changed')
        if mixed_tail:
            tail_updates = {name: float((dict(model.named_parameters())[name].detach().cpu()-original).abs().max())
                            for name, original in mixed_tail.items()}
            if len(tail_updates) != 8 or not all(value > 0 and np.isfinite(value) for value in tail_updates.values()):
                raise RuntimeError('Mixed4 tail adapters failed to update')
        local_identity = {'schema_version': 1, 'model_name': MODEL, 'model_revision': MODEL_REVISION,
                          'split_sha256': hashes['split'], 'init_adapter_sha256': hashes['adapter'],
                          'recipe': f'RGB interaction local expert {vision_layout} upper{upper_blocks}',
                          'vision_layout': vision_layout, 'vision_train_blocks': lock['vision_train_blocks'],
                          'vision_frozen_adapter_blocks': lock['vision_frozen_adapter_blocks'],
                          'upper_blocks': upper_blocks, 'optimizer_steps': steps, 'seed': 42,
                          'train_samples': train_seen, 'validation_samples': len(ids), 'epochs': 1,
                          'train_subset_sha256': hashlib.sha256(json.dumps(lock['train_ids']).encode()).hexdigest(),
                          'source_sha256': hashes, 'test_evaluated': False, 'arm': 'rgb_interaction_local', 'amp': False}
        torch.save({'identity': local_identity, 'baseline_identity': source['identity'],
                    'model': {'target_modules': model.target_modules, 'lora_state_dict': lora_state_dict(model),
                              'classifier_state_dict': model.classifier.state_dict()}, 'lock': lock}, root/'local_adapter.pt')
        model.eval()
    reusable = {}
    if args.mode == 'audit':
        prior_crops = {r['sample_id']: r for r in json.loads(Path('artifacts/rgb_hand_roi_localization_v1/crop_manifest.json').read_text())['samples']}
        for i, sid in enumerate(prior['sample_ids']):
            if sid in crops:
                if any(crops[sid][key] != prior_crops[sid][key] for key in ('rgb_sha256', 'frame_indices', 'boxes', 'fallback')):
                    raise RuntimeError('Reused crop geometry/source changed')
                reusable[sid] = prior['logits']['interaction'][i]
    pending_rows = [row for row in rows if sample_id(row) not in reusable]
    loader = make_loader(pending_rows, crops, processor, args)
    outputs = dict(reusable)
    offset = 0
    started = time.monotonic()
    with torch.inference_mode():
        for batch, (pixels, y) in enumerate(loader, 1):
            logits = model(pixels.cuda()).cpu()
            if not torch.isfinite(logits).all():
                raise RuntimeError('Nonfinite local predictions')
            batch_rows = pending_rows[offset:offset+len(y)]
            if not torch.equal(y, torch.tensor([int(row['class_id'])-1 for row in batch_rows])):
                raise RuntimeError('Validation ordering changed')
            for row, values in zip(batch_rows, logits):
                outputs[sample_id(row)] = values
            offset += len(y)
            if batch % 25 == 0 or batch == len(loader):
                progress('validation', samples=len(outputs), total=len(ids), seconds=time.monotonic()-started)
    logits = torch.stack([outputs[sid] for sid in ids])
    report, probabilities = evaluate_metrics(logits, reference, selected, labels, ids, prior_ids)
    report.update({'vision_target_gradients': vision_gradients, 'tail_adapter_updates': tail_updates, 'vision_layout': vision_layout, 'frozen_adapter_parity': True, 'upper_blocks': upper_blocks, 'optimizer_steps': steps, 'added_target_gradients': added_gradients, 'mode': args.mode, 'training': args.mode in ('pilot', 'smoke'), 'test_evaluated': False,
                   'source_sha256': hashes, 'fallbacks': lock['fallbacks'], 'train_samples': train_seen,
                   'gradient_families': gradients, 'warm_start_parity_max_error': error,
                   'inference_seconds': time.monotonic()-started, 'reused_predictions': len(reusable)})
    if args.mode in ('pilot', 'smoke'):
        with torch.inference_mode():
            example, _ = next(iter(make_loader(rows[:args.batch_size], crops, processor, args)))
            original = model(example.cuda()).cpu()
        merge_lora(model)
        merged = merged_base_state_dict(model)
        restored = XCLIPVisionClassifier(XCLIPModel.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=True),
                                        num_classes=120, head_dropout=.25, feature_noise_std=.01).cuda().eval()
        restored.load_state_dict(merged, strict=True)
        with torch.inference_mode():
            reload_error = float((restored(example.cuda()).cpu()-original).abs().max())
        if reload_error > 1e-4:
            raise RuntimeError('Merged local expert reload parity failed')
        torch.save(merged, root/'merged_local_rgb_state.pt')
        report['merged_reload_max_abs_error'] = reload_error
        report['status'] = 'smoke_pass' if args.mode == 'smoke' else 'pilot_complete'
        expanded_predictions = torch.load(args.expanded_report.parent/'view_predictions.pt', map_location='cpu', weights_only=True)
        expanded_index = {sid:i for i,sid in enumerate(expanded_predictions['sample_ids'])}
        prior_local = expanded_predictions['probabilities']['whole_zoom_interaction_mean'][torch.tensor([expanded_index[sid] for sid in ids])]
        prior_ok = prior_local.argmax(-1).eq(labels)
        trained_ok = probabilities['whole_zoom_interaction_mean'].argmax(-1).eq(labels)
        masks = {'all': torch.ones(len(ids), dtype=torch.bool), 'new_samples': torch.tensor([sid not in prior_ids for sid in ids]),
                 'retained86': ~torch.isin(labels, torch.tensor(PRODUCT_EXCLUDED_IDS)-1),
                 'targets': torch.isin(labels, torch.tensor(TARGETS)-1),
                 'attention_A042_A048': torch.isin(labels, torch.arange(41,48))}
        report['vs_untrained_local'] = {name:{'samples':int(mask.sum()), 'before_correct':int(prior_ok[mask].sum()),
                                       'after_correct':int(trained_ok[mask].sum()), 'fixed':int((mask & trained_ok & ~prior_ok).sum()),
                                       'regressed':int((mask & ~trained_ok & prior_ok).sum())} for name,mask in masks.items()}
    if upper_blocks == 4:
        previous = torch.load(comparison_path, map_location='cpu', weights_only=True)
        report['vs_upper2_local'] = compare_local_experts(probabilities, labels, ids, previous, set(selection['sample_ids']))
        if sha256_file(comparison_path) != hashes['upper2_predictions']:
            raise RuntimeError('Upper2 comparison changed during run')
    if alternative:
        previous = torch.load(upper4_comparison_path, map_location='cpu', weights_only=True)
        report['vs_upper4_local'] = compare_local_experts(probabilities, labels, ids, previous, set(selection['sample_ids']))
        if sha256_file(upper4_comparison_path) != hashes['upper4_predictions']:
            raise RuntimeError('Upper4 comparison changed during run')
    if vision_layout == 'mixed4':
        previous = torch.load(middle4_comparison_path, map_location='cpu', weights_only=True)
        report['vs_middle4_local'] = compare_local_experts(probabilities, labels, ids, previous, set(selection['sample_ids']))
        if sha256_file(middle4_comparison_path) != hashes['middle4_predictions']:
            raise RuntimeError('Middle4 comparison changed during run')
    torch.save({'logits': {'interaction': logits}, 'probabilities': probabilities, 'labels': labels,
                'sample_ids': ids, 'training': args.mode in ('pilot', 'smoke'), 'test_evaluated': False}, root/'view_predictions.pt')
    if any(sha256_file(path) != hashes[name] for name, path in {
        **dependency_paths, 'runner': Path(__file__), 'adapter': ADAPTER, 'reference': REFERENCE,
        'split': SPLIT, 'val_selection': args.val_selection, 'val_crops': args.val_crops,
        **({'train_selection': args.train_selection, 'train_crops': args.train_crops} if train_seen else {})}.items()):
        raise RuntimeError('Source/input changed during run')
    write_json(root/'report.json', report)
    progress('complete', training_gate=report['training_gate'], targets=report['new_samples_only']['targets'],
             retained86=report['new_samples_only']['retained86'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vision-layout', choices=('upper', 'middle4', 'mixed4'), default='upper')
    parser.add_argument('--upper-blocks', type=int, choices=(2, 4), default=2)
    parser.add_argument('--mode', choices=('audit', 'smoke', 'pilot'), default='audit')
    parser.add_argument('--artifacts-dir', type=Path, required=True)
    parser.add_argument('--val-selection', type=Path, default=Path('artifacts/hand_roi_followup_v1/val_selection.json'))
    parser.add_argument('--val-crops', type=Path, default=Path('artifacts/hand_roi_followup_v1/val_localization/crop_manifest.json'))
    parser.add_argument('--train-selection', type=Path, default=Path('artifacts/hand_roi_followup_v1/train_selection.json'))
    parser.add_argument('--train-crops', type=Path, default=Path('artifacts/hand_roi_followup_v1/train_localization/crop_manifest.json'))
    parser.add_argument('--expanded-report', type=Path, default=Path('artifacts/hand_roi_followup_v1/expanded_audit/report.json'))
    parser.add_argument('--smoke-report', type=Path, default=Path('artifacts/hand_roi_followup_v1/smoke/report.json'))
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or not 1 <= args.batch_size <= 4:
        parser.error('workers and batch-size must be1–4')
    run(args)
