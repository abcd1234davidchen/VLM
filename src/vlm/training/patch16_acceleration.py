"""Bounded temporal/spatial precision and throughput trials alongside the semantic pilot.

GPU timings are explicitly contended measurements, not isolated throughput or
full-pilot accuracy evidence. Optional held-out screening never launches a pilot.
"""

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from transformers import XCLIPProcessor

from vlm.patch16_skeleton_targets import TargetCache, atomic_save
from vlm.processed_uint8_cache import ProcessedUInt8Cache
from vlm.skeleton_teacher_cache import sha256_file
from vlm.training.decision_distillation import score_kl
from vlm.training.patch16_skeleton import BASE, STUDENT, TEACHER_TEXT, build_model, build_candidate, parity, snapshot_sources, forward_aux, auxiliary, classification_report
from vlm.training.ski_semantic import balanced
from vlm.training.skeletonclip_phase4_gate2 import _sample_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--method', choices=('temporal', 'spatial', 'native'), default='temporal')
    parser.add_argument('--anchor-cache', type=Path)
    parser.add_argument('--validate-precision', action='store_true', help='Compare batch4 arms on 120 held-out whole-view samples after training; not full-pilot accuracy')
    args = parser.parse_args()
    if args.steps != 8:
        parser.error('Parallel screening is bounded to eight measured steps per trial')
    root = args.artifacts_dir
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'benchmark_lock.json').exists():
        raise ValueError('Existing benchmark namespace cannot be overwritten')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    free, total = torch.cuda.mem_get_info()
    # Reserve at least 1 GiB free for the live semantic job and cap this process.
    fraction = min(.5, (free - 1024 ** 3) / total)
    if fraction < .15:
        raise RuntimeError('Insufficient safe memory budget for parallel benchmark')
    torch.cuda.set_per_process_memory_fraction(fraction)
    adapter = BASE / 'best_adapter.pt'
    split_path = Path('artifacts/xclip/selected_split.json')
    paired_path = Path('artifacts/skeleton_data/paired_manifest.json')
    source = torch.load(adapter, map_location='cpu', weights_only=False)
    identity = source['identity']
    if ((identity['model_name'], identity['model_revision']) != STUDENT or source['epoch'] != 20
            or identity['split_sha256'] != sha256_file(split_path) or identity['test_evaluated'] is not False):
        raise ValueError('Baseline provenance differs')
    split = json.loads(split_path.read_text())
    paired = json.loads(paired_path.read_text())
    if len(split['train']) != 80640 or len(split['val']) != 12960 or len(paired['train']) != 45236:
        raise ValueError('Canonical train split differs')
    processor = XCLIPProcessor.from_pretrained(STUDENT[0], revision=STUDENT[1], local_files_only=True)
    legacy = XCLIPProcessor.from_pretrained(TEACHER_TEXT[0], revision=TEACHER_TEXT[1], local_files_only=True)
    if processor.image_processor.to_dict() != legacy.image_processor.to_dict():
        raise ValueError('Processor cache compatibility differs')
    pixels_cache = ProcessedUInt8Cache(Path('cache/skeletonclip_phase3_rgb_uint8'), legacy,
                                      model_name=TEACHER_TEXT[0], model_revision=TEACHER_TEXT[1])
    if args.method == 'native':
        if not args.anchor_cache or not args.validate_precision:
            raise ValueError('Native screen requires --anchor-cache and --validate-precision')
        from vlm.patch16_native_targets import NativeTargetCache
        targets = NativeTargetCache('cache/patch16_skeleton_targets_v1', 'cache/patch16_native_targets_v1', processor.image_processor.to_dict())
    else:
        targets = TargetCache(Path('cache/patch16_skeleton_targets_v1'), args.method, processor.image_processor.to_dict())
    source_files = [Path(__file__), Path(__file__).with_name('patch16_skeleton.py'),
                    Path(__file__).parents[1] / 'patch16_skeleton_targets.py',
                    Path(__file__).parents[1] / 'patch16_skeleton_contracts.py',
                    Path(__file__).parents[1] / 'models/patch16_temporal.py',
                    Path(__file__).parents[1] / 'models/xclip_vision_lora.py',
                    Path(__file__).parents[1] / 'models/lora.py',
                    Path(__file__).parents[1] / 'processed_uint8_cache.py',
                    Path(__file__).parents[1] / 'phase4_alignment.py',
                    Path(__file__).parents[1] / 'spatial_joint_targets.py',
                    Path(__file__).with_name('spatial_attention.py'),
                    Path(__file__).with_name('decision_distillation.py')]
    if args.method == 'native':
        source_files.extend([Path(__file__).parents[1] / 'models/patch16_native.py', Path(__file__).parents[1] / 'patch16_native_targets.py', Path(__file__).with_name('patch16_native_runtime.py')])
    hashes = {str(p): sha256_file(p) for p in source_files}
    hashes.update({str(p): sha256_file(p) for p in (adapter, split_path, paired_path)})
    lock = {'source_sha256': hashes, 'method': args.method, 'precision_validation': args.validate_precision, 'student': STUDENT, 'cache_identity': TEACHER_TEXT,
            'started_at': datetime.now(timezone.utc).isoformat(), 'steps_per_trial': 8, 'warmup_steps': 2,
            'workers': 0, 'cpu_threads': 2, 'memory_budget_fraction': fraction, 'parallel': True,
            'purpose': 'Bounded numerical/throughput screen; optional held-out diagnostic, no pilot launch',
            'test_evaluated': False}
    with (root / 'benchmark_lock.json').open('x') as stream:
        json.dump(lock, stream, indent=2)
    snapshot_sources(root, hashes)
    def progress(state, **fields):
        record = {'state': state, 'timestamp': datetime.now(timezone.utc).isoformat(), **fields}
        temporary = root / 'progress.tmp'
        temporary.write_text(json.dumps(record, indent=2)); temporary.replace(root / 'progress.json')
        print(json.dumps(record), flush=True)
    progress('building_fixed_train_batches')
    lookup = {r['sample_id']: r for r in paired['train']}
    records, ids = [], []
    for row in balanced(split['train'], 2000, 42):
        sample = _sample_id(row['video_path'])
        pair = lookup.get(sample)
        if pair is None:
            continue
        if pair['class_index'] != int(row['class_id']) - 1 or Path(pair['rgb_path']).resolve() != Path(row['video_path']).resolve():
            raise ValueError('Train pairing identity differs')
        payload = targets.load(pair)
        if not torch.as_tensor(payload['valid']).any():
            continue
        records.append((pixels_cache.load(Path(row['video_path'])), int(row['class_id']) - 1, payload['target'], torch.as_tensor(payload['valid'])))
        ids.append(sample)
        if len(records) == 32:
            break
    if len(records) != 32:
        raise RuntimeError('Fewer than 32 valid train-only auxiliary samples')
    x_cpu = torch.stack([r[0] for r in records])
    y = torch.tensor([r[1] for r in records], device='cuda')
    target = torch.stack([r[2] for r in records]).cuda()
    valid = torch.stack([r[3] for r in records]).to(device='cuda', dtype=torch.bool)
    x = x_cpu.cuda()
    anchor = build_model('anchor', source).cuda().requires_grad_(False).eval()
    progress('preparing_frozen_anchor_cache')
    cache_start = time.monotonic()
    with torch.inference_mode():
        anchor_scores = torch.cat([anchor(x[start:start + 4]).float() for start in range(0, 32, 4)])
        # Repeated online FP32 forward must agree with the cached target exactly
        # at the original batch geometry; larger batches tolerate FP32 rounding.
        cache_error = parity(anchor(x[:4]).cpu(), anchor_scores[:4].cpu(), atol=1e-5, rtol=1e-5)
        larger_cache_error = parity(anchor(x[:16]).cpu(), anchor_scores[:16].cpu(), atol=2e-4, rtol=1e-5)
    atomic_save({'sample_ids': ids, 'labels': y.cpu(), 'logits': anchor_scores.cpu(),
                 'identity': lock, 'test_evaluated': False}, root / 'sample_anchor_cache.pt')
    if args.method == 'native':
        from vlm.training.patch16_anchor_cache import cache_identity, load_anchor_cache
        cache_key, all_ids, all_labels, _ = cache_identity(split['train'], pixels_cache, adapter, split_path)
        full_anchor = load_anchor_cache(args.anchor_cache, cache_key, all_ids, all_labels)
        positions = {sample: i for i, sample in enumerate(all_ids)}
        selected_anchor = full_anchor[[positions[sample] for sample in ids]].cuda()
        if not torch.equal(anchor_scores, selected_anchor):
            raise RuntimeError('Shared FP32 anchor differs from online fixed batch4 geometry')
        anchor_scores = selected_anchor
    cache_seconds = time.monotonic() - cache_start
    # A CPU template guarantees each trial starts from the original adapter.
    torch.manual_seed(42)
    template = build_candidate(args.method, source)
    validation_rows = balanced(split['val'], 120, 42) if args.validate_precision else []
    validation_ids = [_sample_id(row['video_path']) for row in validation_rows]
    validation_labels = torch.tensor([int(row['class_id']) - 1 for row in validation_rows])
    if set(ids) & set(validation_ids):
        raise ValueError('Validation overlaps train')
    def validate(candidate):
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(validation_rows), 4):
                # Evaluation stays FP32 for all arms to isolate training precision.
                pixels = torch.stack([pixels_cache.load(Path(row['video_path']))
                                      for row in validation_rows[start:start + 4]]).cuda()
                outputs.append(candidate(pixels).float().cpu())
        logits = torch.cat(outputs)
        if not torch.isfinite(logits).all():
            raise RuntimeError('Nonfinite held-out logits')
        metrics = classification_report(logits, validation_labels, validation_ids, {r['sample_id'] for r in paired['val']})
        metrics['correct'] = int(logits.argmax(-1).eq(validation_labels).sum())
        metrics['nll'] = float(F.cross_entropy(logits, validation_labels))
        return logits, metrics
    trials = [('fp32_b4_online_anchor', 4, False, False),
              ('fp32_b4_cached_anchor', 4, False, True),
              ('amp_b4_cached_anchor', 4, True, True),
              ('amp_b8_cached_anchor', 8, True, True),
              ('amp_b16_cached_anchor', 16, True, True)]
    if args.validate_precision:
        trials = trials[:3]
    results = []
    predictions = {}
    for name, size, amp, cached in trials:
        model = optimizer = scaler = None
        named, parameters = [], []
        initial_amp = None
        try:
            model = copy.deepcopy(template).cuda().eval()
            named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            if any(not (n.endswith(('lora_a', 'lora_b')) or n.startswith(('classifier.', 'temporal_gate.', 'spatial_head.'))) for n, _ in named):
                raise RuntimeError('Unexpected trainable parameters')
            parameters = [p for _, p in named]
            optimizer = torch.optim.AdamW(parameters, lr=2.25e-5, weight_decay=5e-4, betas=(.9, .999), eps=1e-8)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=201600)
            scaler = torch.amp.GradScaler('cuda', enabled=amp, init_scale=256, growth_interval=1000000)
            with torch.inference_mode():
                parity(model(x[:4]).cpu(), anchor_scores[:4].cpu(), atol=2e-4, rtol=1e-5)
                with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                    initial_amp = model(x[:size]).float()
                if not torch.isfinite(initial_amp).all():
                    raise RuntimeError('Nonfinite initial mixed-precision logits')
                agreement = float(initial_amp.argmax(-1).eq(anchor_scores[:size].argmax(-1)).float().mean())
                initial_error = float((initial_amp - anchor_scores[:size]).abs().max())
            def step(index):
                selected = (torch.arange(size, device='cuda') + index * size) % len(x)
                pixels = x[selected]
                optimizer.zero_grad(set_to_none=True)
                # Targets retain original FP32 anchor semantics in every arm.
                with torch.no_grad():
                    reference = anchor_scores[selected] if cached else anchor(pixels).float()
                with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                    logits, weights = forward_aux(model, pixels, args.method)
                ce = F.cross_entropy(logits.float(), y[selected], label_smoothing=.05)
                retention = score_kl(logits.float(), reference)
                extra = auxiliary(weights.float(), y[selected], target[selected], valid[selected], args.method, None)
                loss = ce + retention + extra
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite benchmark loss')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True)
                scaler.step(optimizer); scaler.update(); scheduler.step()
                return {'ce': float(ce.detach()), 'retention': float(retention.detach()),
                        'auxiliary': float(extra.detach()), 'gradient_norm': float(norm)}
            progress('trial', name=name, batch_size=size, amp=amp, cached_anchor=cached)
            for index in range(2):
                step(index)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            started = time.monotonic(); losses = []
            for index in range(8):
                losses.append(step(index + 2))
            torch.cuda.synchronize(); elapsed = time.monotonic() - started
            gradient_families = {prefix: sum(float(p.grad.abs().sum()) for n, p in named if n.startswith(prefix) and p.grad is not None)
                                 for prefix in (('vision_model.', 'mit.', 'classifier.') if args.method == 'native' else ('vision_model.', 'mit.', 'classifier.', 'temporal_gate.' if args.method == 'temporal' else 'spatial_head.'))}
            if not all(value > 0 for value in gradient_families.values()):
                raise RuntimeError(f'Gradient family missing: {gradient_families}')
            if any(p.grad is not None for p in anchor.parameters()):
                raise RuntimeError('Frozen anchor received gradients')
            result = {'name': name, 'status': 'pass', 'batch_size': size, 'amp': amp, 'cached_anchor': cached,
                      'seconds': elapsed, 'steps': 8, 'samples': 8 * size, 'examples_per_second': 8 * size / elapsed,
                      'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'initial_argmax_agreement': agreement,
                      'initial_max_logit_error': initial_error, 'losses': losses, 'gradients': gradient_families}
            if args.validate_precision:
                progress('heldout_validation', name=name, samples=120)
                predictions[name], result['heldout'] = validate(model)
                atomic_save({'sample_ids': validation_ids, 'labels': validation_labels,
                             'logits': predictions[name], 'evaluation_precision': 'fp32',
                             'training_precision': 'amp_fp16' if amp else 'fp32',
                             'method': args.method, 'test_evaluated': False}, root / name / 'validation_predictions.pt')
                # Exact cached-vs-online equality includes the learned auxiliary head.
                if args.method == 'native':
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                        screen_logits, screen_attention = forward_aux(model, x[:4], args.method)
                    atomic_save({'attention': screen_attention.float().cpu(), 'logits': screen_logits.float().cpu(), 'target': target[:4].cpu(), 'valid': valid[:4].cpu(), 'sample_ids': ids[:4], 'test_evaluated': False}, root / name / 'native_attention.pt')
                parameters_cpu = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
                atomic_save({'state_dict': parameters_cpu, 'optimizer': optimizer.state_dict(),
                             'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                             'identity': lock, 'test_evaluated': False}, root / name / 'screen_checkpoint.pt')
        except torch.cuda.OutOfMemoryError as error:
            result = {'name': name, 'status': 'failed', 'reason': 'benchmark_memory_budget', 'error': repr(error)}
        except (RuntimeError, ValueError) as error:
            result = {'name': name, 'status': 'failed', 'error': repr(error)}
        results.append(result)
        named.clear(); parameters.clear()
        del model, optimizer, scaler, initial_amp
        torch.cuda.empty_cache()
        progress('trial_complete', result=result)
    baseline = results[0]
    for result in results:
        if baseline['status'] == 'pass' and result['status'] == 'pass':
            result['contended_speedup'] = result['examples_per_second'] / baseline['examples_per_second']
    passing = [r for r in results if r['status'] == 'pass']
    fastest = max(passing, key=lambda r: r['examples_per_second'])['name'] if passing else None
    def exact_tree(a, b):
        if isinstance(a, torch.Tensor):
            return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
        if isinstance(a, dict):
            return isinstance(b, dict) and set(a) == set(b) and all(exact_tree(v, b[k]) for k, v in a.items())
        if isinstance(a, (list, tuple)):
            return isinstance(b, type(a)) and len(a) == len(b) and all(exact_tree(x,y) for x,y in zip(a,b))
        return a == b
    precision_checks = {}
    if args.validate_precision and all(r['status'] == 'pass' for r in results):
        online, cached, mixed = [r['name'] for r in results]
        a = torch.load(root / online / 'screen_checkpoint.pt', map_location='cpu', weights_only=False)
        b = torch.load(root / cached / 'screen_checkpoint.pt', map_location='cpu', weights_only=False)
        exact_weights = all(torch.equal(value, b['state_dict'][key]) for key, value in a['state_dict'].items())
        exact_losses = results[0]['losses'] == results[1]['losses']
        exact_predictions = torch.equal(predictions[online], predictions[cached])
        precision_checks = {'cached_fp32_weights_exact': exact_weights, 'cached_fp32_loss_trace_exact': exact_losses,
                            'cached_fp32_validation_logits_exact': exact_predictions,
                            'amp_vs_fp32_prediction_agreement': float(predictions[online].argmax(-1).eq(predictions[mixed].argmax(-1)).float().mean()),
                            'amp_vs_fp32_max_logit_error': float((predictions[online] - predictions[mixed]).abs().max()),
                            'amp_accuracy_delta': results[2]['heldout']['raw_accuracy'] - results[0]['heldout']['raw_accuracy'],
                            'amp_macro_f1_delta': results[2]['heldout']['raw_macro_f1'] - results[0]['heldout']['raw_macro_f1'],
                            'cached_fp32_optimizer_exact': exact_tree(a['optimizer'], b['optimizer']),
                            'cached_fp32_scheduler_exact': exact_tree(a['scheduler'], b['scheduler']),
                            'full_validation_required': True, 'pilot_accuracy_equivalence_verified': False}
        if not (exact_weights and exact_losses and exact_predictions):
            raise RuntimeError('Cached FP32 arm is not equivalent to online FP32')
    report = {'status': 'pass' if baseline['status'] == 'pass' and passing and (not args.validate_precision or all(r['status'] == 'pass' for r in results)) else 'failed', 'method': args.method, 'precision_checks': precision_checks, 'validation_sample_ids': validation_ids,
              'raw_accuracy_gain': None, 'parallel_with_semantic': True, 'isolated_throughput_verified': False,
              'classification_improvement_verified': False, 'sample_ids': ids,
              'anchor_cache_error': cache_error, 'larger_batch_anchor_cache_error': larger_cache_error,
              'anchor_cache_preparation_seconds': cache_seconds, 'results': results, 'fastest_contended': fastest,
              'next_gate': 'Isolated recipe-specific audit/smoke, then full 12960-sample whole/zoom validation each epoch; no full accuracy conclusion from this subset',
              'test_evaluated': False, 'ended_at': datetime.now(timezone.utc).isoformat()}
    if args.method == 'native':
        from vlm.training.patch16_native_runtime import native_recipe_identity
        report['native_recipe_identity'] = native_recipe_identity(adapter, split_path, paired_path, args.anchor_cache)
        report['parallel_with_semantic'] = False
        report['isolated_throughput_verified'] = True
        required = ('cached_fp32_weights_exact', 'cached_fp32_loss_trace_exact', 'cached_fp32_validation_logits_exact', 'cached_fp32_optimizer_exact', 'cached_fp32_scheduler_exact')
        if precision_checks.get('amp_vs_fp32_prediction_agreement') != 1 or not all(precision_checks.get(k) for k in required):
            report['status'] = 'failed'
    (root / 'run_report.json').write_text(json.dumps(report, indent=2))
    progress('complete', status=report['status'], fastest_contended=fastest)


if __name__ == '__main__':
    main()
