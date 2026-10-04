"""Independent matched CE/retention workflow and training provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from vlm.patch16_skeleton_targets import atomic_save
from vlm.reporting import write_json
from vlm.skeleton_teacher_cache import sha256_file


class IndexedControlDataset(Dataset):
    def __init__(self, dataset):self.dataset = dataset
    def __len__(self):return len(self.dataset)
    def __getitem__(self, index):return (*self.dataset[index], index)


def audit_native_recipe(reference, lock, root):
    reference = Path(reference)
    previous = json.loads((reference / 'experiment_lock.json').read_text())
    fields = ('seed', 'student', 'batch_size', 'precision', 'gradient_accumulation', 'epochs_cap',
              'scheduler_steps', 'learning_rate', 'weight_decay', 'betas', 'eps', 'label_smoothing',
              'clip', 'retention_weight', 'retention_temperature', 'workers', 'cpu_threads',
              'processor', 'train_ids', 'validation_ids', 'attention_backend', 'anchor_cache_sha256')
    if lock['mode'] != 'pilot':
        fields = tuple(key for key in fields if key not in ('train_ids', 'validation_ids'))
    differences = [key for key in fields if json.loads(json.dumps(lock.get(key))) != previous.get(key)]
    if differences:
        raise ValueError(f'Control/Native recipes differ: {differences}')
    snapshot = reference / 'source_snapshot'
    verified = {}
    for path, digest in previous['source_sha256'].items():
        if Path(path).suffix != '.py':continue
        relative = Path(path).resolve().relative_to(Path.cwd())
        saved = snapshot / relative
        verified[path] = saved.exists() and sha256_file(saved) == digest
    if not all(verified.values()):raise ValueError('Native historical source snapshot differs')
    native_runner = (snapshot / 'src/vlm/training/patch16_skeleton.py').read_text()
    if "init_scale=256, growth_interval=1000000" not in native_runner:
        raise ValueError('Native scaler recipe cannot be verified')
    for path in ('src/vlm/models/patch16_native.py', 'src/vlm/models/xclip_vision_lora.py', 'src/vlm/models/lora.py', 'src/vlm/processed_uint8_cache.py'):
        absolute = str(Path(path).resolve())
        if sha256_file(Path(path)) != previous['source_sha256'].get(absolute):
            raise ValueError(f'Matched model/input dependency changed: {path}')
    report = {'status': 'pass', 'classification': 'recipe_matched_exploratory', 'bitwise_replay_claimed': False,
              'recipe_fields_checked': list(fields), 'recipe_differences': differences,
              'native_snapshot_verified': verified, 'native_lock_sha256': sha256_file(reference / 'experiment_lock.json'),
              'historical_training_boundary_rng_available': False,
              'historical_epoch_order_digest_available': False,
              'historical_order_reconstruction': 'replay seed42 loader algorithm, compare completed generator states',
              'native_scaler': {'init_scale': 256, 'growth_interval': 1000000}, 'test_evaluated': False}
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / 'matched_recipe.json', report)
    return report


def control_gradients(model, dataset, args, root):
    from vlm.training.patch16_skeleton import loader, forward_aux, auxiliary, math_isfinite
    from vlm.training.decision_distillation import score_kl
    pixels, labels, target, valid = [v.cuda() for v in next(iter(loader(dataset, args)))]
    named = dict(model.named_parameters())
    results = {}
    for kind in ('ce', 'retention'):
        model.zero_grad(set_to_none=True)
        logits, maps = forward_aux(model, pixels, 'ce_retention')
        extra = auxiliary(maps, labels, target, valid, 'ce_retention', None)
        if extra.requires_grad or float(extra) != 0 or valid.any():
            raise RuntimeError('Control auxiliary loss must be absent')
        if kind == 'ce':loss = F.cross_entropy(logits, labels, label_smoothing=.05)
        else:
            scores = logits.detach().clone().requires_grad_(True)
            with torch.no_grad():scores[:, 0] += 1
            loss = score_kl(logits, scores)
        loss.backward()
        gradient = {prefix: sum(float(p.grad.abs().sum()) for n,p in named.items() if n.startswith(prefix) and p.grad is not None)
                    for prefix in ('vision_model.', 'mit.', 'classifier.')}
        if not all(value > 0 and math_isfinite(value) for value in gradient.values()):
            raise RuntimeError(f'Control gradient failure: {gradient}')
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise RuntimeError('Control updated frozen base')
        if kind == 'retention' and scores.grad is not None:raise RuntimeError('Retention target received gradient')
        results[kind] = gradient
    model.zero_grad(set_to_none=True)
    write_json(root / 'control_gradients.json', {'gradients': results, 'auxiliary': 0, 'target_used': False, 'test_evaluated': False})
    return pixels


def save_training_boundary(model, training, scaler, root):
    import random
    import numpy as np
    atomic_save({'data_generator_state': training.generator.get_state(), 'torch_rng_state': torch.get_rng_state(),
                 'cuda_rng_states': torch.cuda.get_rng_state_all(), 'python_rng_state': random.getstate(),
                 'numpy_rng_state': np.random.get_state(), 'grad_scaler': scaler.state_dict(),
                 'trainable': {n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},
                 'test_evaluated': False}, root / 'training_boundary.pt')


def sample_order_loader(training, reference, root, ids, start_epoch=1):
    class RecordedLoader:
        def __init__(self):
            self.epoch = start_epoch - 1
            self.expected_generator = torch.Generator().manual_seed(42)
            if start_epoch > 1:
                torch.empty((), dtype=torch.int64).random_(generator=self.expected_generator)
                for _ in range(1, start_epoch):
                    torch.randperm(len(ids), generator=self.expected_generator)
                    torch.randperm(len(ids), generator=self.expected_generator)
        def __getattr__(self, name):return getattr(training, name)
        def __iter__(self):
            self.epoch += 1
            # Historical persistent loader draws one base seed at epoch1 only.
            if self.epoch == 1:
                torch.empty((), dtype=torch.int64).random_(generator=self.expected_generator)
            expected = torch.randperm(len(ids), generator=self.expected_generator).tolist()
            torch.randperm(len(ids), generator=self.expected_generator) # RandomSampler's final empty slice draw.
            order = []
            for batch in training:
                order.extend(batch[-1].tolist())
                yield batch[:-1]
            if order != expected:raise RuntimeError('Control shuffle differs from reconstructed Native algorithm')
            state = torch.load(Path(reference) / f'epoch{self.epoch:02d}/checkpoint.pt', map_location='cpu', weights_only=False)
            same = torch.equal(training.generator.get_state(), state['data_generator_state'])
            if not same:raise RuntimeError('Control epoch-end generator differs from historical Native')
            digest = hashlib.sha256(''.join(ids[index]+'\n' for index in order).encode()).hexdigest()
            atomic_save({'sample_ids': [ids[index] for index in order], 'test_evaluated': False}, root / f'epoch{self.epoch:02d}/sample_order.pt')
            write_json(root / f'epoch{self.epoch:02d}/sample_order.json',
                       {'samples': len(order), 'sha256': digest, 'reconstructed_native_order_equal': True,
                        'native_completed_generator_equal': same, 'historical_recorded_order_available': False,
                        'bitwise_replay_claimed': False, 'test_evaluated': False})
    return RecordedLoader()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts-root', type=Path, required=True)
    parser.add_argument('--anchor-cache', type=Path, required=True)
    parser.add_argument('--native-reference', type=Path, default=Path('artifacts/patch16_skeleton_v1/native_attention_01/native/pilot/seed42'))
    args = parser.parse_args();root = args.artifacts_root
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'workflow_lock.json').open('x') as stream:
        json.dump({'method': 'ce_retention', 'stages': ['audit','smoke','pilot','comparison'], 'epochs': 2,
                   'native_reference': str(args.native_reference.resolve()), 'test_evaluated': False}, stream, indent=2)
    results = {}
    for mode in ('audit', 'smoke', 'pilot'):
        phase = root / mode / 'seed42'
        command = [sys.executable, '-m', 'vlm.training.patch16_skeleton', '--method', 'ce_retention', '--mode', mode,
                   '--artifacts-dir', str(phase), '--audit-dir', str(root/'audit/seed42'), '--smoke-dir', str(root/'smoke/seed42'),
                   '--precision', 'amp_fp16', '--anchor-cache', str(args.anchor_cache.resolve()), '--native-reference', str(args.native_reference.resolve())]
        phase.mkdir(parents=True, exist_ok=True)
        write_json(root/'progress.json', {'state':'running','method':'ce_retention','mode':mode,'artifacts':str(phase),'test_evaluated':False})
        with (phase/'console.log').open('xb',buffering=0) as stream:
            child = subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
            write_json(phase/'process.json',{'pid':child.pid,'command':command,'test_evaluated':False});code=child.wait()
        report_path=phase/'run_report.json';report=json.loads(report_path.read_text()) if report_path.exists() else {'status':'failed'}
        results[mode]={'exit_code':code,'status':report['status'],'report':str(report_path)};write_json(root/'workflow_results.json',results)
        if code or (mode!='pilot' and report['status']!='pass'):
            write_json(root/'run_report.json',{'status':'failed','results':results,'raw_accuracy_gain':None,'test_evaluated':False});return
    from vlm.training.patch16_control_compare import compare_runs
    compare_runs(args.native_reference,root/'pilot/seed42',root/'comparison')
    write_json(root/'run_report.json',{'status':'complete','results':results,'comparison':str(root/'comparison/run_report.json'),'raw_accuracy_gain':None,'test_evaluated':False})
    write_json(root/'progress.json',{'state':'complete','test_evaluated':False})


if __name__ == '__main__':main()
