"""Evaluate the frozen RGB local expert on every retained86 validation example."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import XCLIPProcessor

from vlm.reporting import write_json
from vlm.skeleton_teacher_cache import sha256_file
from vlm.training.hand_roi_followup import (
    ADAPTER, MODEL, MODEL_REVISION, REFERENCE, SPLIT, evaluate_metrics,
    load_model, make_loader, sample_id,
)
from vlm.training.rgb_input_audit import PRODUCT_EXCLUDED_IDS


def run(args):
    torch.set_num_threads(2)
    root = args.artifacts_dir
    if (root / 'experiment_lock.json').exists():
        raise ValueError('Use a fresh evaluation directory')
    selection = json.loads(args.selection.read_text())
    split = json.loads(SPLIT.read_text())
    rows = [r for r in split['val'] if int(r['class_id']) not in PRODUCT_EXCLUDED_IDS]
    ids = [sample_id(r) for r in rows]
    if selection['sample_ids'] != ids or len(ids) != 8640 or len(set(ids)) != len(ids):
        raise ValueError('Expected every canonical retained86 validation example')
    payload = json.loads(args.crops.read_text())
    pilot_root = args.local_adapter.parent
    pilot_lock = json.loads((pilot_root / 'experiment_lock.json').read_text())
    pilot_crops_path = Path('artifacts/hand_roi_followup_v1/val_localization/crop_manifest.json')
    if sha256_file(pilot_crops_path) != pilot_lock['source_sha256']['val_crops']:
        raise ValueError('Pilot crop manifest changed')
    pilot_crops = json.loads(pilot_crops_path.read_text())
    if payload['localization'] != pilot_crops['localization'] or payload['pose_model_sha256'] != pilot_crops['pose_model_sha256']:
        raise ValueError('Localization recipe changed')
    crops = {r['sample_id']: r for r in payload['samples']}
    old_crops = {r['sample_id']: r for r in pilot_crops['samples']}
    if set(crops) != set(ids):
        raise ValueError('Crop coverage mismatch')
    source = torch.load(args.local_adapter, map_location='cpu', weights_only=False)
    identity = source['identity']
    if identity['init_adapter_sha256'] != sha256_file(ADAPTER) or identity['split_sha256'] != sha256_file(SPLIT) or identity['model_revision'] != MODEL_REVISION or identity['test_evaluated'] is not False:
        raise ValueError('Local adapter provenance mismatch')
    if identity['train_samples'] != 2280 or identity['epochs'] != 1:
        raise ValueError('Expected the completed single-pass pilot')
    reference = torch.load(REFERENCE, map_location='cpu', weights_only=True)
    index = {sid: i for i, sid in enumerate(reference['sample_ids'])}
    selected = torch.tensor([index[sid] for sid in ids])
    labels = torch.tensor([int(r['class_id']) - 1 for r in rows])
    if not torch.equal(labels, reference['labels'][selected]):
        raise ValueError('Reference ordering/labels mismatch')
    pilot_path = pilot_root / 'view_predictions.pt'
    pilot = torch.load(pilot_path, map_location='cpu', weights_only=True)
    outputs = {}
    for i, sid in enumerate(pilot['sample_ids']):
        if sid in crops:
            if any(crops[sid][k] != old_crops[sid][k] for k in ('rgb_sha256', 'frame_indices', 'boxes', 'fallback')):
                raise ValueError('Cached prediction crop/source changed')
            if pilot['labels'][i].item() != labels[ids.index(sid)].item():
                raise ValueError('Cached prediction labels changed')
            outputs[sid] = pilot['logits']['interaction'][i]
    hashes = {name: sha256_file(path) for name, path in {
        'runner': Path(__file__), 'shared_runner': Path('src/vlm/training/hand_roi_followup.py'),
        'local_adapter': args.local_adapter, 'baseline_adapter': ADAPTER, 'reference': REFERENCE,
        'split': SPLIT, 'selection': args.selection, 'crops': args.crops,
        'pilot_predictions': pilot_path, 'pilot_crop_manifest': pilot_crops_path,
    }.items()}
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / 'experiment_lock.json', {
        'sample_ids': ids, 'source_sha256': hashes, 'training': False, 'test_evaluated': False,
        'policy': 'All retained86 canonical validation, keep120 logits; fixed equal whole+zoom+trained interaction probability mean',
        'excluded_source_ids': PRODUCT_EXCLUDED_IDS, 'reused_predictions': len(outputs),
        'batch_size': args.batch_size, 'workers': args.workers,
    })
    (root / 'runner_snapshot.py').write_text(Path(__file__).read_text())
    (root / 'shared_runner_snapshot.py').write_text(Path('src/vlm/training/hand_roi_followup.py').read_text())
    started = time.monotonic()

    def progress(state, **fields):
        value = {'state': state, 'samples': len(outputs), 'total': len(ids),
                 'seconds': time.monotonic() - started, **fields}
        write_json(root / 'progress.json', value)
        print(json.dumps(value), flush=True)

    processor = XCLIPProcessor.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=True)
    model = load_model(source)
    check_ids = list(outputs)[:args.batch_size]
    lookup = {sample_id(r): r for r in rows}
    with torch.inference_mode():
        pixels, _ = next(iter(make_loader([lookup[sid] for sid in check_ids], crops, processor, args)))
        error = float((model(pixels.cuda()).cpu() - torch.stack([outputs[sid] for sid in check_ids])).abs().max())
        if error > 1e-4:
            raise RuntimeError('Frozen local checkpoint/cache parity failed')
        progress('checkpoint_verified', cached_parity_max_error=error)
        pending = [r for r in rows if sample_id(r) not in outputs]
        offset = 0
        loader = make_loader(pending, crops, processor, args)
        for batch, (pixels, y) in enumerate(loader, 1):
            logits = model(pixels.cuda()).cpu()
            batch_rows = pending[offset:offset + len(y)]
            if not torch.isfinite(logits).all() or not torch.equal(y, torch.tensor([int(r['class_id']) - 1 for r in batch_rows])):
                raise RuntimeError('Invalid predictions or ordering')
            for row, values in zip(batch_rows, logits):
                outputs[sample_id(row)] = values
            offset += len(y)
            if batch % 25 == 0 or batch == len(loader):
                progress('validation')
                torch.save({'sample_ids': list(outputs), 'logits': torch.stack(list(outputs.values()))}, root / 'partial_predictions.pt')
    logits = torch.stack([outputs[sid] for sid in ids])
    report, probabilities = evaluate_metrics(logits, reference, selected, labels, ids, set(pilot['sample_ids']))
    report.pop('training_gate')
    report['prior_diagnostic_only'] = report.pop('prior176_only')
    report['not_in_prior_diagnostic'] = report.pop('new_samples_only')
    report.update({'status': 'complete', 'training': False, 'test_evaluated': False,
                   'samples': len(ids), 'source_sha256': hashes, 'seconds': time.monotonic() - started,
                   'cached_parity_max_error': error, 'fallbacks': sum(c['fallback'] is not None for c in crops.values())})
    torch.save({'sample_ids': ids, 'labels': labels, 'logits': {'interaction': logits},
                'probabilities': probabilities, 'training': False, 'test_evaluated': False}, root / 'view_predictions.pt')
    if any(sha256_file(path) != hashes[name] for name, path in {
        'local_adapter': args.local_adapter, 'baseline_adapter': ADAPTER,
        'reference': REFERENCE, 'split': SPLIT, 'selection': args.selection, 'crops': args.crops,
    }.items()):
        raise RuntimeError('Input changed during evaluation')
    write_json(root / 'report.json', report)
    summary = report['all_selected']['retained86']
    lines = ['# RGB hand interaction：完整 86 類 validation', '',
             '評估固定的 2,280 筆、單 pass local expert；沒有追加訓練或修改原 baseline。所有 canonical validation 中 ground truth 屬於保留 86 類的 8,640 支影片都納入，分類仍保留 120 logits；預測到排除類別照樣算錯。沒有讀取 test。', '',
             '融合固定為 whole、zoom、trained interaction 的 softmax 機率等權平均。這是完整開發 validation，包含先前用於診斷的樣本；另列出其餘樣本，不宣稱為獨立 test。', '',
             '| 輸入 | 答對／總數 | Accuracy | 修正 | 退步 |',
             '|---|---:|---:|---:|---:|']
    for view, value in summary.items():
        lines.append(f"| {view} | {value['correct']}/{value['samples']} | {100*value['accuracy']:.4f}% | {value['fixed']} | {value['regressed']} |")
    lines += ['', '## 先前診斷樣本以外', '',
              '| 群組 | 原 whole＋zoom | 加入訓練後 interaction | 淨變動 |',
              '|---|---:|---:|---:|']
    for group, values in report['not_in_prior_diagnostic'].items():
        before = values['whole_zoom_mean']; after = values['whole_zoom_interaction_mean']
        lines.append(f"| {group} | {before['correct']}/{before['samples']} | {after['correct']}/{after['samples']} | {after['correct']-before['correct']:+d} |")
    lines += ['', '## 每個保留類別', '',
              '| 類別 | 原 whole＋zoom | 加入訓練後 interaction | 淨變動 |',
              '|---|---:|---:|---:|']
    for cls, values in report['per_class'].items():
        if int(cls[1:]) in PRODUCT_EXCLUDED_IDS:
            continue
        before = values['whole_zoom_mean']; after = values['whole_zoom_interaction_mean']
        lines.append(f"| {cls} | {before['correct']}/{before['samples']} | {after['correct']}/{after['samples']} | {after['correct']-before['correct']:+d} |")
    before = summary['whole_zoom_mean']; after = summary['whole_zoom_interaction_mean']
    lines += ['', '## 結果', '',
              f"完整保留 86 類：{before['correct']}/{before['samples']} → {after['correct']}/{after['samples']}，變動 {100*(after['accuracy']-before['accuracy']):+.4f} 個百分點，修正 {after['fixed']} 筆、退步 {after['regressed']} 筆。這次評估不自動替換部署模型。", '',
              f"RGB 定位 fallback {report['fallbacks']} 支，全部保留；局部 checkpoint 重載與快取 prediction 最大差異 {error:.8g}。原 checkpoint 與所有輸入的 SHA256 在評估結束時再次驗證。", '',
              f'Artifacts：`{root}`；定位 metadata：`{args.crops}`。', '']
    Path('docs/hand-roi-full86-v1-results.md').write_text('\n'.join(lines))
    progress('complete', retained86=report['all_selected']['retained86'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, default=Path('artifacts/hand_roi_full86_v1/val_selection.json'))
    parser.add_argument('--crops', type=Path, default=Path('artifacts/hand_roi_full86_v1/localization/crop_manifest.json'))
    parser.add_argument('--local-adapter', type=Path, default=Path('artifacts/hand_roi_followup_v1/pilot/local_adapter.pt'))
    parser.add_argument('--artifacts-dir', type=Path, default=Path('artifacts/hand_roi_full86_v1/evaluation'))
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 4 or not 1 <= args.workers <= 4:
        parser.error('batch-size and workers must be1–4')
    run(args)
