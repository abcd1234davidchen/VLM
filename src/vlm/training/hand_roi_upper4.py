"""Run the fixed four-block smoke, single-pass pilot, and retained86 evaluation."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

from vlm.reporting import write_json


def run(root: Path, *, vision_layout: str = "upper") -> None:
    if vision_layout not in ("upper", "middle4", "mixed4"):
        raise ValueError("Unknown four-block layout")
    if root.exists():
        raise ValueError('Use a fresh upper4 artifacts directory')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required; run on the GPU host')
    gpu = subprocess.run(['nvidia-smi'], check=True, capture_output=True, text=True).stdout
    root.mkdir(parents=True)
    (root / 'gpu_before.txt').write_text(gpu)
    (root / 'runner_snapshot.py').write_text(Path(__file__).read_text())
    stages = [
        ('smoke', ['vlm.training.hand_roi_followup', '--mode', 'smoke', '--upper-blocks', '4',
                   '--artifacts-dir', str(root / 'smoke')]),
        ('pilot', ['vlm.training.hand_roi_followup', '--mode', 'pilot', '--upper-blocks', '4',
                   '--smoke-report', str(root / 'smoke/report.json'), '--artifacts-dir', str(root / 'pilot')]),
        ('evaluation', ['vlm.training.hand_roi_full_validation', '--upper-blocks', '4',
                        '--local-adapter', str(root / 'pilot/local_adapter.pt'),
                        '--artifacts-dir', str(root / 'evaluation')]),
    ]
    for stage, arguments in stages:
        arguments += ['--vision-layout', vision_layout]
        write_json(root / 'progress.json', {'state': stage, 'test_evaluated': False})
        with (root / f'{stage}.log').open('w') as log:
            result = subprocess.run([sys.executable, '-m', *arguments], stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            write_json(root / 'progress.json', {'state': 'failed', 'stage': stage,
                                              'returncode': result.returncode, 'test_evaluated': False})
            raise RuntimeError(f'{stage} failed; inspect {root / (stage + ".log")}')
    write_json(root / 'progress.json', {'state': 'complete', 'test_evaluated': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vision-layout', choices=('upper', 'middle4', 'mixed4'), default='upper')
    parser.add_argument('--artifacts-dir', type=Path)
    args = parser.parse_args()
    root = args.artifacts_dir or Path('artifacts/hand_roi_mixed4_v1' if args.vision_layout == 'mixed4' else 'artifacts/hand_roi_middle4_v1' if args.vision_layout == 'middle4' else 'artifacts/hand_roi_upper4_v1')
    run(root, vision_layout=args.vision_layout)
