"""Resumable, shared FP32 whole-view anchor logits for the canonical train split."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import inspect
import os
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import XCLIPProcessor

from vlm.patch16_skeleton_targets import atomic_save
from vlm.processed_uint8_cache import ProcessedUInt8Cache
from vlm.skeleton_teacher_cache import sha256_file
from vlm.training.patch16_skeleton import BASE, STUDENT, TEACHER_TEXT, build_model, parity, snapshot_sources
from vlm.training.skeletonclip_phase4_gate2 import _sample_id


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def fingerprint(path):
    path = Path(path).resolve()
    stat = path.stat()
    return [str(path), stat.st_size, stat.st_mtime_ns]


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def cache_identity(rows, pixels_cache, adapter, split_path):
    ids = [_sample_id(row['video_path']) for row in rows]
    labels = [int(row['class_id']) - 1 for row in rows]
    if len(set(ids)) != len(rows) or not all(0 <= label < 120 for label in labels):
        raise ValueError('Duplicate train IDs or invalid raw120 labels')
    sources = [fingerprint(row['video_path']) for row in rows]
    model_dir = Path(__file__).parents[1] / 'models'
    identity = {'schema_version': 1, 'model': list(STUDENT), 'adapter_sha256': sha256_file(adapter),
                'adapter_epoch': 20, 'anchor_builder_sha256': hashlib.sha256(inspect.getsource(build_model).encode()).hexdigest(), 'split_sha256': sha256_file(split_path),
                'ordered_ids_labels_sha256': digest_json([ids, labels]),
                'video_fingerprints_sha256': digest_json(sources),
                'pixel_cache_config_hash': pixels_cache.config_hash,
                'model_sources': {name: sha256_file(model_dir / name) for name in ('xclip_vision_lora.py', 'lora.py')},
                'view': 'whole', 'precision': 'fp32', 'anchor_policy': 'eval, frozen, original RGB baseline',
                'batch_size': 4, 'samples': len(rows), 'classes': 120, 'test_evaluated': False}
    return identity, ids, labels, sources


def validate_payload(payload, identity, ids, labels):
    logits = payload['logits']
    if (payload.get('identity') != identity or payload.get('test_evaluated') is not False
            or payload['sample_ids'] != ids or not torch.equal(payload['labels'], torch.tensor(labels))
            or logits.dtype != torch.float32 or logits.shape != (len(ids), 120)
            or not torch.isfinite(logits).all()):
        raise ValueError('Anchor cache identity, row order, labels or finite FP32 logits differ')
    return logits


def load_anchor_cache(root, identity, ids, labels):
    """Never consume a partially built or mismatched cache."""
    root = Path(root)
    manifest = json.loads((root / 'cache_manifest.json').read_text())
    if manifest.get('state') != 'complete' or manifest['identity'] != identity:
        raise ValueError('Anchor cache is incomplete or incompatible')
    path = root / 'logits.pt'
    if sha256_file(path) != manifest['logits_sha256']:
        raise ValueError('Anchor cache checksum differs')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    return validate_payload(payload, identity, ids, labels)


class AnchorPixels(Dataset):
    def __init__(self, rows, cache, start):
        self.rows, self.cache, self.start = rows, cache, start

    def __len__(self):
        return len(self.rows) - self.start

    def __getitem__(self, index):
        index += self.start
        return index, self.cache.load(Path(self.rows[index]['video_path']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts-dir', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, default=BASE / 'best_adapter.pt')
    parser.add_argument('--split', type=Path, default=Path('artifacts/xclip/selected_split.json'))
    parser.add_argument('--rgb-cache', type=Path, default=Path('cache/skeletonclip_phase3_rgb_uint8'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-new-samples', type=int, default=0, help='Optional contention probe; pause after a multiple of 1024 new samples')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--yield-seconds', type=float, default=.05)
    parser.add_argument('--yield-until-pid', type=int, help='Apply the yielding delay only while this training PID is alive')
    args = parser.parse_args()
    if args.workers < 0 or args.yield_seconds < 0 or args.max_new_samples < 0 or args.max_new_samples % 1024:
        parser.error('Workers/delay must be nonnegative; sample limit must be zero or a multiple of 1024')
    args.workers = min(args.workers, 24)
    root = args.artifacts_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    guard = (root / 'builder.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(2)
    split = json.loads(args.split.read_text())
    rows = split['train']
    if len(rows) != 80640:
        raise ValueError('Expected canonical 80640-sample train split')
    train_ids = {_sample_id(row['video_path']) for row in rows}
    if train_ids & {_sample_id(row['video_path']) for row in split['val']}:
        raise ValueError('Train/validation overlap')
    source = torch.load(args.adapter, map_location='cpu', weights_only=False)
    original = source['identity']
    if ((original['model_name'], original['model_revision']) != STUDENT or source['epoch'] != 20
            or original['split_sha256'] != sha256_file(args.split) or original['arm'] != 'vision'
            or original['test_evaluated'] is not False):
        raise ValueError('Original anchor provenance differs')
    processor = XCLIPProcessor.from_pretrained(STUDENT[0], revision=STUDENT[1], local_files_only=True)
    legacy = XCLIPProcessor.from_pretrained(TEACHER_TEXT[0], revision=TEACHER_TEXT[1], local_files_only=True)
    if processor.image_processor.to_dict() != legacy.image_processor.to_dict():
        raise ValueError('Pixel processor compatibility differs')
    pixels_cache = ProcessedUInt8Cache(args.rgb_cache, legacy, model_name=TEACHER_TEXT[0], model_revision=TEACHER_TEXT[1])
    identity, ids, labels, sources = cache_identity(rows, pixels_cache, args.adapter, args.split)
    files = [Path(__file__), Path(__file__).with_name('patch16_skeleton.py'),
             Path(__file__).parents[1] / 'models/xclip_vision_lora.py',
             Path(__file__).parents[1] / 'models/lora.py',
             Path(__file__).parents[1] / 'processed_uint8_cache.py',
             Path(__file__).parents[1] / 'xclip_features.py',
             Path(__file__).parents[1] / 'patch16_skeleton_targets.py']
    hashes = {str(path): sha256_file(path) for path in files}
    hashes.update({str(args.adapter): sha256_file(args.adapter), str(args.split): sha256_file(args.split)})
    lock_path = root / 'cache_lock.json'
    if lock_path.exists():
        if not args.resume:
            raise ValueError('Cache namespace exists; use --resume')
        lock = json.loads(lock_path.read_text())
        if lock['identity'] != identity or {k: v for k, v in lock['source_sha256'].items() if Path(k).name not in ('patch16_skeleton.py', 'patch16_anchor_cache.py')} != {k: v for k, v in hashes.items() if Path(k).name not in ('patch16_skeleton.py', 'patch16_anchor_cache.py')}:
            raise ValueError('Cannot resume cache built with different inputs or model/data dependencies')
        session_root = root / 'resume_sessions' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        session_root.mkdir(parents=True)
        write_json(session_root / 'resume_lock.json', {'source_sha256': hashes, 'identity': identity,
                   'workers': args.workers, 'yield_seconds': args.yield_seconds, 'yield_until_pid': args.yield_until_pid})
        snapshot_sources(session_root, hashes)
    else:
        lock = {'identity': identity, 'source_sha256': hashes, 'started_at': datetime.now(timezone.utc).isoformat(),
                'workers': args.workers, 'cpu_threads': 2, 'yield_seconds': args.yield_seconds, 'test_evaluated': False}
        with lock_path.open('x') as stream:
            json.dump(lock, stream, indent=2)
        snapshot_sources(root, hashes)
        write_json(root / 'input_fingerprints.json', {'sample_ids': ids, 'labels': labels, 'videos': sources})
    def emit(state, done, **extra):
        progress = {'state': state, 'timestamp': datetime.now(timezone.utc).isoformat(),
                    'samples': done, 'total': len(rows), 'test_evaluated': False, **extra}
        write_json(root / 'progress.json', progress)
        print(json.dumps(progress), flush=True)
    if (root / 'cache_manifest.json').exists():
        load_anchor_cache(root, identity, ids, labels)
        emit('complete', len(rows), resumed_complete_cache=True)
        return
    # Only consecutive validated shards can be reused; orphaned temp files are ignored.
    blocks, shard_manifest = [], []
    done = 0
    index_path = root / 'shard_index.json'
    committed = json.loads(index_path.read_text()) if index_path.exists() else []
    for record in committed:
        path = root / record['path']
        if path.parent != root / 'shards' or sha256_file(path) != record['sha256']:
            raise ValueError('Committed shard checksum/path differs')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        stop = payload['stop']
        if payload['start'] != done or not done < stop <= len(rows) or done % 4 or stop % 4:
            raise ValueError('Noncontiguous cache shard')
        blocks.append(validate_payload(payload, identity, ids[done:stop], labels[done:stop]))
        shard_manifest.append({'path': str(path.relative_to(root)), 'start': done, 'stop': stop, 'sha256': sha256_file(path)})
        done = stop
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    free, total = torch.cuda.mem_get_info()
    fraction = min(.25, (free - 1024 ** 3) / total)
    if fraction < .12:
        raise RuntimeError('Insufficient GPU headroom alongside live training')
    torch.cuda.set_per_process_memory_fraction(fraction)
    anchor = build_model('anchor', source).cuda().requires_grad_(False).eval()
    (root / 'run_report.json').unlink(missing_ok=True)
    initial_done = done
    dataset = AnchorPixels(rows, pixels_cache, done)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0, prefetch_factor=2 if args.workers else None)
    started = time.monotonic()
    pending = []
    delay = args.yield_seconds
    shard_start = done
    emit('building_anchor_cache', done, memory_budget_fraction=fraction, resumed=bool(done))
    with torch.inference_mode(), torch.autocast('cuda', enabled=False):
        for indices, pixels in loader:
            if indices.tolist() != list(range(done, done + len(pixels))):
                raise ValueError('Cache loader row order differs')
            values = anchor(pixels.cuda(non_blocking=True)).float().cpu()
            if values.shape != (len(pixels), 120) or not torch.isfinite(values).all():
                raise RuntimeError('Nonfinite or malformed anchor logits')
            if not pending:
                error = parity(anchor(pixels.cuda()).float().cpu(), values, atol=1e-5, rtol=1e-5)
                if error != 0:
                    raise RuntimeError('Repeated FP32 anchor/cache logits are not exact')
            pending.append(values)
            done += len(pixels)
            if done - shard_start == 1024 or done == len(rows):
                if [fingerprint(row['video_path']) for row in rows[shard_start:done]] != sources[shard_start:done]:
                    raise RuntimeError('Video source changed while building cache')
                block = torch.cat(pending)
                payload = {'identity': identity, 'sample_ids': ids[shard_start:done],
                           'labels': torch.tensor(labels[shard_start:done]), 'logits': block,
                           'start': shard_start, 'stop': done, 'test_evaluated': False}
                path = root / 'shards' / f'{shard_start:06d}_{done:06d}.pt'
                atomic_save(payload, path)
                blocks.append(block)
                shard_manifest.append({'path': str(path.relative_to(root)), 'start': shard_start, 'stop': done, 'sha256': sha256_file(path)})
                write_json(index_path, shard_manifest)
                pending.clear(); shard_start = done
                elapsed = time.monotonic() - started
                rate = (done - initial_done) / elapsed
                emit('building_anchor_cache', done, examples_per_second=rate,
                     eta_seconds=(len(rows) - done) / rate, completed_shards=len(blocks))
                if args.max_new_samples and done - initial_done >= args.max_new_samples and done < len(rows):
                    write_json(root / 'run_report.json', {'status': 'paused', 'reason': 'bounded contention probe',
                               'samples': done, 'examples_per_second': rate, 'elapsed_seconds_this_session': elapsed,
                               'raw_accuracy_gain': None, 'test_evaluated': False})
                    emit('paused', done, examples_per_second=rate, elapsed_seconds=elapsed)
                    return
            if done % 256 == 0 and pending:
                elapsed = time.monotonic() - started
                rate = (done - initial_done) / elapsed
                emit('building_anchor_cache', done, committed_samples=shard_start,
                     examples_per_second=rate, eta_seconds=(len(rows)-done)/rate)
            if args.yield_until_pid and done % 256 == 0 and delay:
                try:
                    stat = Path(f'/proc/{args.yield_until_pid}/stat').read_text()
                    alive = stat.rsplit(')', 1)[1].split()[0] != 'Z'
                except (FileNotFoundError, ProcessLookupError):
                    alive = False
                if not alive:
                    delay = 0
                    emit('building_anchor_cache', done, yielding_ended=True, training_pid=args.yield_until_pid)
            if delay:
                time.sleep(delay)
    if [fingerprint(row['video_path']) for row in rows] != sources:
        raise RuntimeError('Video sources changed before cache publication')
    payload = {'identity': identity, 'sample_ids': ids, 'labels': torch.tensor(labels),
               'logits': torch.cat(blocks), 'test_evaluated': False}
    validate_payload(payload, identity, ids, labels)
    atomic_save(payload, root / 'logits.pt')
    write_json(root / 'cache_manifest.json', {'state': 'complete', 'identity': identity,
               'logits_sha256': sha256_file(root / 'logits.pt'), 'shards': shard_manifest, 'test_evaluated': False})
    load_anchor_cache(root, identity, ids, labels)
    write_json(root / 'run_report.json', {'status': 'pass', 'samples': done, 'identity': identity,
               'elapsed_seconds_this_session': time.monotonic() - started, 'resumed_samples': initial_done,
               'cache': str(root / 'logits.pt'), 'raw_accuracy_gain': None, 'test_evaluated': False})
    emit('complete', done)


if __name__ == '__main__':
    main()
