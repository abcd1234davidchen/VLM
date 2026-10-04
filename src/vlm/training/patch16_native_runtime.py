"""Native recipe identity, production audits and fixed masking diagnostics."""
from pathlib import Path
import json

import torch
from PIL import Image

from vlm.skeleton_teacher_cache import sha256_file
from vlm.patch16_native_targets import native_attention_metrics, native_patch_distribution
from vlm.patch16_skeleton_targets import atomic_save


def native_recipe_identity(adapter, split, paired, anchor_cache):
    base = Path(__file__).parents[1]
    files = [Path(__file__), Path(__file__).with_name('patch16_skeleton.py'),
             Path(__file__).with_name('patch16_acceleration.py'), base / 'models/patch16_native.py',
             base / 'models/xclip_vision_lora.py', base / 'models/lora.py',
             base / 'patch16_native_targets.py', base / 'patch16_skeleton_targets.py',
             base / 'phase4_alignment.py', base / 'processed_uint8_cache.py',
             Path(adapter), Path(split), Path(paired), Path(anchor_cache) / 'logits.pt']
    return {'sources': {str(p.resolve()): sha256_file(p) for p in files},
            'batch_size': 4, 'native_layers': [10, 11], 'native_heads': [0, 1, 2, 3],
            'target_version': 'native-p16-occupancy-gaussian1-uniform20-v1', 'backend': 'eager',
            'precision': 'amp_fp16', 'loss_precision': 'fp32', 'validation_precision': 'fp32'}


def require_native_precision(root, args):
    report = json.loads((Path(root) / 'run_report.json').read_text())
    expected = native_recipe_identity(args.adapter, args.split_manifest, args.paired_manifest, args.anchor_cache)
    checks = report.get('precision_checks', {})
    if (report.get('status') != 'pass' or report.get('method') != 'native'
            or report.get('test_evaluated') is not False or report.get('native_recipe_identity') != expected
            or checks.get('amp_vs_fp32_prediction_agreement') != 1
            or not all(checks.get(k) for k in ('cached_fp32_weights_exact', 'cached_fp32_loss_trace_exact',
                                             'cached_fp32_validation_logits_exact', 'cached_fp32_optimizer_exact',
                                             'cached_fp32_scheduler_exact'))):
        raise ValueError('Native precision gate missing, incompatible or not passing')
    if args.precision != 'amp_fp16':
        raise ValueError('Native gate is for the fixed AMP recipe only')


def audit_native_q_gradient(model):
    names = dict(model.named_parameters())
    q = names['vision_model.encoder.layers.11.self_attn.q_proj.lora_b']
    v = names['vision_model.encoder.layers.11.self_attn.v_proj.lora_b']
    if q.grad is None or not torch.isfinite(q.grad).all() or q.grad.abs().sum() <= 0:
        raise RuntimeError('Native loss did not reach last vision Q')
    if v.grad is not None and (not torch.isfinite(v.grad).all() or v.grad.abs().sum() != 0):
        raise RuntimeError('Native attention-only loss unexpectedly reached same-layer V')
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise RuntimeError('Native loss reached frozen base')
    return {'last_vision_q': float(q.grad.abs().sum()), 'last_vision_v': 0.}


def native_parity_audit(model, anchor, dataset, args, root):
    from vlm.training.patch16_skeleton import loader, parity, emit
    outputs, maps, targets, masks, occupancies = [], [], [], [], []
    errors = []
    with torch.inference_mode():
        for pixels, _, target, valid in loader(dataset, args):
            x = pixels.cuda()
            enabled, attention = model(x, return_native_attentions=True)
            disabled = model(x)
            errors.append(parity(enabled.cpu(), disabled.cpu(), atol=1e-5, rtol=1e-5))
            parity(enabled.cpu(), anchor(x).cpu(), atol=1e-5, rtol=1e-5)
            if attention.shape[1:] != (8, 2, 198) or not torch.allclose(attention.sum(-1), torch.ones_like(attention[..., 0]), atol=1e-5):
                raise RuntimeError('Native token shape or attention row normalization differs')
            outputs.append(enabled.cpu());maps.append(attention.cpu());targets.append(target);masks.append(valid)
            emit(root, 'native_parity_audit', samples=sum(len(x) for x in outputs), total=len(dataset))
    ids = []
    for row in dataset.rows:
        from vlm.training.skeletonclip_phase4_gate2 import _sample_id
        sample = _sample_id(row['video_path']);ids.append(sample)
        occupancies.append(dataset.targets.load(dataset.paired[sample])['occupancy'])
    atomic_save({'sample_ids': ids, 'logits': torch.cat(outputs), 'attention': torch.cat(maps),
                 'target': torch.cat(targets), 'valid': torch.cat(masks), 'occupancy': torch.stack(occupancies),
                 'test_evaluated': False}, root / 'native_attention_audit.pt')
    from vlm.reporting import write_json
    write_json(root / 'native_parity.json', {'status': 'pass', 'samples': len(dataset),
               'max_enabled_disabled_logit_error': max(errors), 'all_argmax_identical': True,
               'token_order': 'CLS[0], patches[1:197], message[197]', 'test_evaluated': False})


def native_diagnostics(model, dataset, args, root):
    from vlm.training.patch16_skeleton import loader, emit
    from vlm.training.skeletonclip_phase4_gate2 import _sample_id
    from vlm.reporting import write_json
    records, causal, offset = [], [], 0
    with torch.inference_mode():
        for pixels, labels, target, valid in loader(dataset, args):
            logits, attention = model(pixels.cuda(), return_native_attentions=True)
            probabilities = native_patch_distribution(attention).mean(-2).cpu()
            for index in range(len(labels)):
                sample_id = _sample_id(dataset.rows[offset + index]['video_path'])
                occupancy = dataset.targets.load(dataset.paired[sample_id])['occupancy']
                metrics = native_attention_metrics(attention[index:index+1], occupancy[None].cuda(), valid[index:index+1].cuda())
                records.append({'sample_id': sample_id, **metrics})
                if not valid[index].any() or len(causal) >= 32:
                    continue
                scores = probabilities[index]
                seed = torch.Generator().manual_seed(42 + len(causal))
                choices = {'highest': scores.topk(5, dim=-1).indices, 'lowest': scores.topk(5, dim=-1, largest=False).indices,
                           'random': torch.stack([torch.randperm(196, generator=seed)[:5] for _ in range(8)])}
                effects = {}
                for name, selected in choices.items():
                    changed = pixels[index:index+1].clone()
                    for frame in range(8):
                        for patch in selected[frame]:
                            y, x = divmod(int(patch), 14)
                            changed[:, frame, :, y*16:(y+1)*16, x*16:(x+1)*16] = 0
                    effects[name] = float(logits[index, labels[index]] - model(changed.cuda())[0, labels[index]])
                causal.append({'sample_id': sample_id, 'logit_drop': effects,
                               'selected_patches': {k: v.tolist() for k, v in choices.items()}})
                destination = root / 'native_overlays' / sample_id;destination.mkdir(parents=True, exist_ok=True)
                mean = torch.tensor(dataset.cache.processor.image_processor.image_mean).view(3, 1, 1)
                std = torch.tensor(dataset.cache.processor.image_processor.image_std).view(3, 1, 1)
                for frame in range(8):
                    image = (pixels[index, frame] * std + mean).clamp(0, 1)
                    heat = scores[frame].reshape(14, 14).repeat_interleave(16, 0).repeat_interleave(16, 1)
                    heat /= heat.max().clamp_min(1e-8)
                    image[0] = .6 * image[0] + .4 * heat
                    Image.fromarray((image.permute(1, 2, 0) * 255).byte().numpy()).save(destination / f'frame_{frame}.png')
            offset += len(labels)
            emit(root, 'native_diagnostics', samples=offset, total=len(dataset), masking_samples=len(causal))
    if len(causal) != 32:
        raise RuntimeError('Native masking audit needs 32 valid held-out samples')
    report = {'samples': len(records), 'rows': records, 'causal': causal,
              'masking': {'patches_per_frame': 5, 'replacement': 'normalized RGB zero', 'seed': 42},
              'test_evaluated': False}
    write_json(root / 'mechanism_diagnostics.json', report)
    return report
