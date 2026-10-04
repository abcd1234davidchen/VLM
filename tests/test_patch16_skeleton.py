"""Critical timing, deployment and selection contracts for skeleton pilots."""
import copy
import hashlib
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import torch
from transformers import XCLIPConfig, XCLIPModel

from vlm.models.lora import load_lora_state_dict, lora_state_dict, merge_lora, merged_base_state_dict
from vlm.models.patch16_temporal import XCLIPTemporalClassifier, XCLIPTemporalLoRAClassifier
from vlm.models.xclip_vision_lora import XCLIPVisionLoRAClassifier
from vlm.patch16_skeleton_contracts import early_kill, promising, temporal_loss, terminal_status, view_probabilities
from vlm.patch16_skeleton_targets import build_target, select_body_ids, temporal_target


class Contracts(unittest.TestCase):
    def test_cached_anchor_stays_aligned_when_training_is_shuffled(self):
        from vlm.training.patch16_skeleton import CachedAnchorDataset
        rows = torch.utils.data.TensorDataset(*(torch.arange(12) for _ in range(4)))
        dataset = CachedAnchorDataset(rows, torch.arange(12).float()[:, None])
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True,
                                             generator=torch.Generator().manual_seed(42))
        for batch in loader:
            self.assertTrue(torch.equal(batch[0].float(), batch[4].squeeze(1)))
        with self.assertRaises(ValueError):
            CachedAnchorDataset(rows, torch.zeros(11, 120))

    def test_absolute_source_snapshot_stays_inside_artifacts_and_checks_hash(self):
        from vlm.training.patch16_skeleton import snapshot_sources
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            source = project / 'src/runner.py'
            source.parent.mkdir()
            source.write_bytes(b'original source')
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            artifacts = project / 'artifacts/run'
            with patch('vlm.training.patch16_skeleton.Path.cwd', return_value=project):
                snapshot_sources(artifacts, {str(source): digest})
                self.assertEqual((artifacts / 'source_snapshot/src/runner.py').read_bytes(), b'original source')
                self.assertEqual(source.read_bytes(), b'original source')
                source.write_bytes(b'changed source')
                with self.assertRaises(RuntimeError):
                    snapshot_sources(project / 'artifacts/other', {str(source): digest})

    def test_body_selection_preserves_distinct_method_policies(self):
        def body(identity):
            return {'body_id': identity, 'joints': [{'state': 2}] * 25}
        selected = {0: [body(1), body(2)], 1: [body(3)], 2: [body(3)], 3: [body(3)]}
        self.assertEqual(select_body_ids(selected, 'spatial', [0]), [1, 2])
        self.assertEqual(select_body_ids(selected, 'temporal', [0]), [3, 1])

    def test_spatial196_tracks_only_valid_in_crop_joints(self):
        with tempfile.TemporaryDirectory() as directory:
            raw, video = Path(directory) / 'sample.skeleton', Path(directory) / 'sample.avi'
            video.write_bytes(b'fixture')
            lines = ['8']
            for frame in range(8):
                lines.extend(['1', '123 0 0 0 0 0 0 0 0 2', '25'])
                for joint in range(25):
                    x, y = (223, 223) if joint == 10 else (16, 16) if joint < 10 else (300, 300)
                    state = 2 if frame == 0 else 0
                    lines.append(f'0 0 0 0 0 {x} {y} 1 0 0 0 {state}')
            raw.write_text('\n'.join(lines) + '\n')
            container = MagicMock()
            container.__enter__.return_value.streams.video = [SimpleNamespace(frames=8, width=224, height=224)]
            row = {'rgb_path': str(video), 'skeleton_path': str(raw), 'sample_id': 'sample', 'skeleton': {'frame_count': 8}}
            with patch('vlm.patch16_skeleton_targets.av.open', return_value=container):
                payload = build_target(row, 'spatial', {})
        self.assertEqual(payload['target'].shape, (8, 196))
        self.assertEqual(payload['target'][0].nonzero().flatten().tolist(), [15, 195])
        self.assertEqual(payload['valid'].tolist(), [True] + [False] * 7)
        self.assertEqual(float(payload['target'][1:].sum()), 0)

    def test_empty_temporal_and_target_direction(self):
        scores = torch.randn(2, 8, requires_grad=True)
        weights = scores.softmax(-1)
        loss = temporal_loss(weights, torch.zeros_like(weights), torch.zeros(2, dtype=torch.bool))
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()
        self.assertTrue(torch.equal(scores.grad, torch.zeros_like(scores)))
        target = torch.tensor([[.8, .2, 0, 0, 0, 0, 0, 0.]])
        valid = torch.ones(1, dtype=torch.bool)
        self.assertLess(float(temporal_loss(target, target, valid)), float(temporal_loss(torch.full_like(target, .125), target, valid)))

    def test_probability_mean_not_logit_mean(self):
        a, b = torch.zeros(1, 120), torch.zeros(1, 120)
        a[0, 0] = 10; b[0, 1] = 3
        p = view_probabilities({'whole': a, 'center_zoom70': b})['ensemble']
        self.assertTrue(torch.equal(p, (a.softmax(-1) + b.softmax(-1)) / 2))
        self.assertFalse(torch.allclose(p, ((a + b) / 2).softmax(-1)))

    def test_guardrails_same_checkpoint_and_epoch_two_kill(self):
        baseline = {'correct': 9606, 'raw_accuracy': 9606 / 12960, 'raw_macro_f1': .75, 'attention_correct': 358,
                    'groups': {'unpaired': {'accuracy': .7}, 'A060_A120': {'accuracy': .7}}}
        good = copy.deepcopy(baseline); good['correct'] += 26
        self.assertTrue(promising(good, baseline))
        bad = copy.deepcopy(good); bad['attention_correct'] -= 1
        self.assertFalse(promising(bad, baseline))
        bad = copy.deepcopy(good); bad['groups']['unpaired']['accuracy'] -= .0021
        self.assertFalse(promising(bad, baseline))
        bases = {v: baseline for v in ('whole', 'center_zoom70', 'ensemble')}
        epoch = {v: {**baseline, 'raw_accuracy': baseline['raw_accuracy'] - .005} for v in bases}
        self.assertFalse(early_kill([epoch], bases))
        self.assertTrue(early_kill([epoch, epoch], bases))
        epoch2 = copy.deepcopy(epoch); epoch2['whole']['raw_accuracy'] += .00001
        self.assertFalse(early_kill([epoch, epoch2], bases))
        self.assertEqual(terminal_status({'whole': good, 'ensemble': good}, {'whole': None, 'ensemble': None}, bases), 'inconclusive')

    def test_timing_static_missing_and_translation_invariance(self):
        def sequence(moving=True, offset=0):
            result = {}
            for frame in range(24):
                joints = [{'xyz': [offset + j * .0625 + (frame * frame * .015625 if j > 0 and moving else 0), j * .125, j * .25], 'state': 2} for j in range(25)]
                result[frame] = [{'body_id': 1, 'joints': joints}]
            return result
        indices = [0, 3, 6, 10, 13, 16, 20, 23]
        result = temporal_target(sequence(), 24, indices, [1])
        self.assertTrue(result['valid'])
        self.assertAlmostEqual(float(result['target'].sum()), 1, places=6)
        self.assertTrue(torch.allclose(result['target'], temporal_target(sequence(offset=100), 24, indices, [1])['target'], atol=1e-6))
        self.assertFalse(temporal_target(sequence(moving=False), 24, indices, [1])['valid'])
        self.assertFalse(temporal_target(sequence(), 24, indices, [9])['valid'])

    def test_temporal_identity_frozen_anchor_and_merged_gate_reload(self):
        torch.set_num_threads(2)
        config = XCLIPConfig(text_config={'hidden_size': 32, 'intermediate_size': 64, 'num_hidden_layers': 1, 'num_attention_heads': 4, 'vocab_size': 100},
                             vision_config={'hidden_size': 32, 'intermediate_size': 64, 'num_hidden_layers': 2, 'num_attention_heads': 4, 'image_size': 32, 'patch_size': 16, 'num_frames': 8, 'mit_hidden_size': 16, 'mit_intermediate_size': 32, 'mit_num_attention_heads': 4, 'mit_num_hidden_layers': 1}, projection_dim=16, prompt_num_attention_heads=4)
        backbone = XCLIPModel(config)
        kwargs = dict(num_classes=120, head_dropout=.25, upper_blocks=1, lora_rank=2, lora_alpha=4, lora_dropout=.05)
        model = XCLIPTemporalLoRAClassifier(copy.deepcopy(backbone), **kwargs).eval()
        anchor = XCLIPVisionLoRAClassifier(copy.deepcopy(backbone), **kwargs).eval()
        load_lora_state_dict(anchor, lora_state_dict(model)); anchor.classifier.load_state_dict(model.classifier.state_dict())
        anchor.requires_grad_(False)
        frozen = copy.deepcopy(anchor.state_dict())
        pixels = torch.randn(1, 8, 3, 32, 32)
        logits, weights = model(pixels, return_temporal=True)
        self.assertTrue(torch.equal(logits, anchor(pixels)))
        self.assertTrue(torch.equal(weights, torch.full_like(weights, .125)))
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        logits.square().mean().backward(); opt.step()
        self.assertTrue(all(torch.equal(v, frozen[n]) for n, v in anchor.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in anchor.parameters()))
        expected = model(pixels).detach()
        merge_lora(model)
        restored = XCLIPTemporalClassifier(copy.deepcopy(backbone), num_classes=120, head_dropout=.25).eval()
        restored.load_state_dict(merged_base_state_dict(model), strict=True)
        self.assertTrue(torch.allclose(restored(pixels), expected, atol=1e-5, rtol=1e-5))
        plain_keys = set(anchor.state_dict())
        self.assertTrue(any(k.startswith('temporal_gate.') for k in restored.state_dict() if k not in plain_keys))

    def test_resume_preserves_updates_rng_and_next_epoch_order_for_all_methods(self):
        import random
        import numpy as np
        from torch.utils.data import TensorDataset
        from vlm.models.ski_semantic import SKISemanticClassifier
        from vlm.models.xclip_vision_lora import XCLIPSpatialLoRAClassifier
        from vlm.training.patch16_skeleton import checkpoint, forward_aux, loader, restore_training
        torch.set_num_threads(2)
        config = XCLIPConfig(text_config={'hidden_size': 32, 'intermediate_size': 64, 'num_hidden_layers': 1, 'num_attention_heads': 4, 'vocab_size': 100},
                             vision_config={'hidden_size': 32, 'intermediate_size': 64, 'num_hidden_layers': 2, 'num_attention_heads': 4, 'image_size': 32, 'patch_size': 16, 'num_frames': 8, 'mit_hidden_size': 16, 'mit_intermediate_size': 32, 'mit_num_attention_heads': 4, 'mit_num_hidden_layers': 1}, projection_dim=16, prompt_num_attention_heads=4)
        kwargs = dict(num_classes=120, head_dropout=.25, upper_blocks=1, lora_rank=2, lora_alpha=4, lora_dropout=.05)
        torch.set_num_threads(1)
        for method, cls in [('semantic', SKISemanticClassifier), ('temporal', XCLIPTemporalLoRAClassifier), ('spatial', XCLIPSpatialLoRAClassifier)]:
            for workers in (0, 1):
                with self.subTest(method=method, workers=workers):
                    torch.manual_seed(42)
                    extra = {'text_prototypes': torch.randn(59, 16), 'semantic_scale': 2.} if method == 'semantic' else {}
                    model = cls(XCLIPModel(config), **kwargs, **extra).eval()
                    template = copy.deepcopy(model)
                    dataset = TensorDataset(torch.arange(8), torch.randn(8, 8, 3, 32, 32))
                    training = loader(dataset, SimpleNamespace(workers=workers), shuffle=True)
                    training.pin_memory = False
                    if workers:
                        training.multiprocessing_context = 'spawn'
                    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
                    def epoch(candidate, opt, sched, batches):
                        order = []
                        for indices, pixels in batches:
                            opt.zero_grad(set_to_none=True)
                            logits, auxiliary = forward_aux(candidate, pixels, method)
                            loss = logits.square().mean() + auxiliary.square().mean()
                            loss.backward(); opt.step(); sched.step()
                            order.extend(indices.tolist())
                        return order
                    epoch(model, optimizer, scheduler, training)
                    with patch('torch.cuda.get_rng_state_all', return_value=[]):
                        state = copy.deepcopy(checkpoint(model, optimizer, scheduler, 1, training, [{}], {}))
                    rng = (random.random(), float(np.random.rand()), torch.rand(3))
                    expected_order = epoch(model, optimizer, scheduler, training)
                    resumed = template
                    resumed_opt = torch.optim.AdamW([p for p in resumed.parameters() if p.requires_grad], lr=.1)
                    resumed_sched = torch.optim.lr_scheduler.CosineAnnealingLR(resumed_opt, T_max=20)
                    resumed_loader = loader(dataset, SimpleNamespace(workers=workers), shuffle=True)
                    resumed_loader.pin_memory = False
                    if workers:
                        resumed_loader.multiprocessing_context = 'spawn'
                    restore_training(state, resumed, resumed_opt, resumed_sched, resumed_loader)
                    self.assertEqual(random.random(), rng[0]); self.assertEqual(float(np.random.rand()), rng[1])
                    self.assertTrue(torch.equal(torch.rand(3), rng[2]))
                    self.assertEqual(epoch(resumed, resumed_opt, resumed_sched, resumed_loader), expected_order)
                    self.assertEqual(scheduler.state_dict(), resumed_sched.state_dict())
                    resumed_weights = resumed.state_dict()
                    for name, value in model.state_dict().items():
                        self.assertTrue(torch.equal(value, resumed_weights[name]), name)
                    resumed_moments = resumed_opt.state_dict()['state']
                    for key, values in optimizer.state_dict()['state'].items():
                        for name, value in values.items():
                            self.assertTrue(torch.equal(value, resumed_moments[key][name]))

    def test_resume_recovers_winner_when_interrupted_during_selection_update(self):
        from vlm.training.patch16_skeleton import inherit_selection
        from vlm.patch16_skeleton_targets import atomic_save
        with tempfile.TemporaryDirectory() as directory:
            source, root = Path(directory) / 'previous', Path(directory) / 'continued'
            model = torch.nn.Linear(2, 1)
            saved = {'epoch': 1, 'state_dict': {'weight': torch.ones_like(model.weight), 'bias': torch.ones_like(model.bias)}}
            atomic_save(saved, source / 'epoch01/checkpoint.pt')
            atomic_save({**saved, 'epoch': 2}, source / 'best_whole.pt')
            def export(winner, method, baseline, pixels, destination, filename):
                self.assertTrue(torch.equal(winner.weight, saved['state_dict']['weight']))
                atomic_save({'winner_epoch': 1}, destination / filename)
            with patch('vlm.training.patch16_skeleton.export_rgb', side_effect=export):
                inherit_selection(source, root, {'whole': {'epoch': 1, 'correct': 5}}, {}, model, 'temporal', {}, None)
            self.assertEqual(torch.load(root / 'best_whole.pt', weights_only=False)['epoch'], 1)
            self.assertEqual(torch.load(root / 'best_whole_rgb.pt', weights_only=False)['winner_epoch'], 1)

    def test_resume_rejects_changed_recipe_sources_or_incomplete_epoch(self):
        from vlm.training.patch16_skeleton import validate_resume, resume_selection
        import vlm.training.patch16_skeleton as runner
        lock = {'method': 'temporal', 'mode': 'pilot', 'epochs_cap': 10, 'batch_size': 4,
                'train_ids': list(range(8)), 'test_evaluated': False,
                'source_sha256': {runner.__file__: 'current', '/tmp/data.json': 'data'}}
        state = {'identity': copy.deepcopy(lock), 'epoch': 1, 'history': [{}], 'scheduler': {'last_epoch': 2},
                 **{k: None for k in ('optimizer', 'data_generator_state', 'torch_rng_state', 'cuda_rng_states', 'python_rng_state', 'numpy_rng_state')}}
        state['identity']['source_sha256'][runner.__file__] = 'previous-audited-runner'
        self.assertEqual(validate_resume(state, lock), 2)
        for key, value in [('batch_size', 8), ('method', 'spatial'), ('test_evaluated', True)]:
            changed = copy.deepcopy(lock); changed[key] = value
            with self.assertRaises(ValueError):
                validate_resume(state, changed)
        changed = copy.deepcopy(lock); changed['source_sha256']['/tmp/data.json'] = 'changed'
        with self.assertRaises(ValueError):
            validate_resume(state, changed)
        broken = copy.deepcopy(state); broken['scheduler']['last_epoch'] = 1
        with self.assertRaises(ValueError):
            validate_resume(broken, lock)
        # Ties must retain the earlier checkpoint across a continuation.
        baseline = {'correct': 10, 'raw_macro_f1': .5, 'attention_correct': 2,
                    'groups': {'unpaired': {'accuracy': .5}, 'A060_A120': {'accuracy': .5}}}
        good = {**baseline, 'correct': 36}
        best, qualified = resume_selection([dict(whole=good, ensemble=good)] * 2, dict(whole=baseline, ensemble=baseline), 'pilot')
        self.assertEqual(best['whole']['epoch'], 1)
        self.assertEqual(qualified['ensemble']['epoch'], 1)


if __name__ == '__main__':
    unittest.main()
