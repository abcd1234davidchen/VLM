"""Sequential audit/smoke/pilot workflow, launched through spatial_dispatch/tmux.

Fresh methods start from the original baseline; resumed pilots retain state.
A failed gate stops that method;
no budget, precision, batch size, loss or weight is changed on failure.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from vlm.reporting import write_json
from vlm.skeleton_teacher_cache import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts-root', type=Path, required=True)
    parser.add_argument('--methods', nargs='+', choices=('semantic', 'temporal', 'spatial', 'native'), default=['semantic', 'temporal', 'spatial'])
    parser.add_argument('--resume', nargs=2, action='append', default=[], metavar=('METHOD', 'CHECKPOINT'),
                        help='Resume a method pilot from last_checkpoint.pt after current audit/smoke gates')
    parser.add_argument('--precision', choices=('fp32', 'amp_fp16'), default='fp32')
    parser.add_argument('--anchor-cache', type=Path)
    parser.add_argument('--wait-anchor-cache-pid', type=int, help='Wait for this running cache builder before starting any method')
    args = parser.parse_args()
    if 'native' in args.methods and (args.anchor_cache is None or args.precision != 'amp_fp16'):
        parser.error('Native workflow requires --anchor-cache and --precision amp_fp16')
    if args.wait_anchor_cache_pid and not args.anchor_cache:
        parser.error('--wait-anchor-cache-pid requires --anchor-cache')
    resumes = {}
    for method, checkpoint_path in args.resume:
        if method not in args.methods or method in resumes:
            parser.error('Resume method must appear once and be included in --methods')
        path = Path(checkpoint_path).resolve()
        if path.name != 'last_checkpoint.pt' or not path.is_file():
            parser.error('Resume requires an existing last_checkpoint.pt')
        resumes[method] = str(path)
    root = args.artifacts_root
    root.mkdir(parents=True, exist_ok=True)
    record = {'methods': args.methods, 'resume_checkpoints': resumes, 'source_sha256': sha256_file(Path(__file__)),
              'runner_sha256': sha256_file(Path(__file__).with_name('patch16_skeleton.py')),
              'precision': args.precision, 'anchor_cache': str(args.anchor_cache) if args.anchor_cache else None,
              'started_at': datetime.now(timezone.utc).isoformat(), 'test_evaluated': False}
    with (root / 'workflow_lock.json').open('x') as stream:
        json.dump(record, stream, indent=2)
    environment = dict(os.environ)
    environment['MPLCONFIGDIR'] = '/tmp/vlm-matplotlib'
    if args.wait_anchor_cache_pid:
        cache_root = args.anchor_cache.resolve()
        while True:
            manifest, report = cache_root / 'cache_manifest.json', cache_root / 'run_report.json'
            if manifest.exists() and report.exists() and json.loads(report.read_text()).get('status') == 'pass':
                if json.loads(manifest.read_text()).get('state') != 'complete':
                    raise RuntimeError('Cache report/manifest disagree')
                break
            try:
                alive = Path(f'/proc/{args.wait_anchor_cache_pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
            except FileNotFoundError:
                alive = False
            if not alive:
                failure = {'status': 'failed', 'error': 'Anchor cache builder ended before verified completion',
                           'raw_accuracy_gain': None, 'test_evaluated': False}
                write_json(root / 'run_report.json', failure)
                raise RuntimeError(failure['error'])
            progress = {'state': 'waiting_anchor_cache', 'method': args.methods[0],
                        'cache': str(cache_root), 'builder_pid': args.wait_anchor_cache_pid, 'test_evaluated': False}
            cache_progress = cache_root / 'progress.json'
            if cache_progress.exists():
                progress['cache_progress'] = json.loads(cache_progress.read_text())
            write_json(root / 'progress.json', progress)
            time.sleep(10)
        if sha256_file(Path(__file__).with_name('patch16_skeleton.py')) != record['runner_sha256'] or sha256_file(Path(__file__)) != record['source_sha256']:
            raise RuntimeError('Queued workflow sources changed before startup')
    results = {}
    for method in args.methods:
        results[method] = {}
        for mode in (('audit', 'precision', 'smoke', 'pilot') if method == 'native' else ('audit', 'smoke', 'pilot')):
            phase = root / method / mode / 'seed42'
            if (phase / 'experiment_lock.json').exists():
                raise RuntimeError('Workflow cannot overwrite or resume a locked run')
            command = [sys.executable, '-m', 'vlm.training.patch16_skeleton', '--method', method,
                       '--mode', mode, '--artifacts-dir', str(phase),
                       '--audit-dir', str(root / method / 'audit/seed42'),
                       '--smoke-dir', str(root / method / 'smoke/seed42')]
            command.extend(['--precision', args.precision])
            if method == 'native':
                command.extend(['--precision-dir', str(root / method / 'precision/seed42')])
            if mode == 'precision':
                command = [sys.executable, '-m', 'vlm.training.patch16_acceleration', '--method', 'native', '--validate-precision', '--artifacts-dir', str(phase)]
            if args.anchor_cache:
                command.extend(['--anchor-cache', str(args.anchor_cache.resolve())])
            if mode == 'pilot' and method in resumes:
                command.extend(['--resume', resumes[method]])
            progress = {'state': 'running', 'method': method, 'mode': mode, 'artifacts': str(phase), 'test_evaluated': False}
            write_json(root / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
            phase.mkdir(parents=True, exist_ok=True)
            with (phase / 'console.log').open('xb', buffering=0) as stream:
                child = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
                write_json(phase / 'process.json', {'pid': child.pid, 'command': command, 'test_evaluated': False})
                code = child.wait()
            path = phase / 'run_report.json'
            report = json.loads(path.read_text()) if path.exists() else {'status': 'failed', 'error': 'No terminal report; inspect console.log'}
            results[method][mode] = {'exit_code': code, 'report': str(path), 'status': report['status']}
            write_json(root / 'workflow_results.json', results)
            if code or mode != 'pilot' and report['status'] != 'pass':
                print(json.dumps({'state': 'method_stopped', 'method': method, 'mode': mode, **results[method][mode]}), flush=True)
                break
    report = {'status': 'complete', 'results': results, 'raw_accuracy_gain': None,
              'test_evaluated': False, 'ended_at': datetime.now(timezone.utc).isoformat()}
    write_json(root / 'run_report.json', report)
    write_json(root / 'progress.json', {'state': 'complete', **report})
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
