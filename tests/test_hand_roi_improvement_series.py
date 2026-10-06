"""Protect data isolation, input geometry, and fusion contracts for the bounded series."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from vlm.training.hand_roi_improvement_series import (
    EQUAL, FOCUS, calibration_mask, choose_config, expanded_box, expanded_selection,
    finish_stage, fuse, stage_complete, temporal_indices,
)


class ImprovementSeriesTests(unittest.TestCase):
    def test_participant_isolation_and_stable_membership(self):
        ids = [f'S001C001P{p:03d}R001A{c:03d}' for p in range(1, 19) for c in (1, 2)]
        mask = calibration_mask(ids)
        self.assertEqual(int(mask.sum()), 18)
        for start in range(0, len(ids), 2):
            self.assertEqual(bool(mask[start]), bool(mask[start+1]))
        reverse = calibration_mask(ids[::-1])
        self.assertTrue(torch.equal(mask, reverse.flip(0)))

    def test_confirmation_labels_cannot_affect_calibration(self):
        generator = torch.Generator().manual_seed(7)
        logits = {name: torch.randn(12, 120, generator=generator)
                  for name in ('whole', 'center_zoom70', 'interaction')}
        labels = torch.randint(120, (12,), generator=generator)
        mask = torch.arange(12) < 6
        first = choose_config(logits, labels, mask)
        changed_labels = labels.clone()
        changed_labels[~mask] = (changed_labels[~mask]+31) % 120
        changed_logits = {name: values.clone() for name, values in logits.items()}
        for values in changed_logits.values():
            values[~mask] *= -100
        self.assertEqual(first, choose_config(changed_logits, changed_labels, mask))

    def test_temporal_fusion_averages_probabilities_before_view_fusion(self):
        a = torch.zeros(1, 2, 120)
        a[0, 0, 0], a[0, 1, 1] = 20, 2
        logits = {'whole': torch.zeros(1, 120), 'center_zoom70': torch.zeros(1, 120), 'interaction': a}
        config = {'weights': [0., 0., 1.], 'temperatures': [1., 1., 2.]}
        expected = (a/2).softmax(-1).mean(1)
        self.assertTrue(torch.equal(fuse(logits, config), expected))
        self.assertFalse(torch.allclose(expected, (a.mean(1)/2).softmax(-1)))
        self.assertTrue(torch.allclose(fuse(logits, EQUAL).sum(-1), torch.ones(1)))

    def test_invalid_fusion_is_rejected(self):
        logits = {name: torch.zeros(1, 120) for name in ('whole', 'center_zoom70', 'interaction')}
        for weights, temperatures in [([1, 1, 1], [1]*3), ([-1, 1, 1], [1]*3),
                                       ([1, 0, 0], [0, 1, 1]), ([1, 0, 0], [float('nan'), 1, 1])]:
            with self.assertRaises(ValueError):
                fuse(logits, {'weights': weights, 'temperatures': temperatures})

    def test_short_video_sampling_and_boundary_crop(self):
        for frames in (1, 3, 8, 100):
            groups = temporal_indices(frames, True)
            self.assertEqual([len(g) for g in groups], [8, 8])
            self.assertTrue(all(np.all((g >= 0) & (g < frames)) for g in groups))
            self.assertEqual(int(groups[0][0]), 0)
            self.assertEqual(int(groups[1][-1]), frames-1)
        self.assertEqual(expanded_box([80, 40, 100, 60], 100, 60), [75, 35, 100, 60])
        self.assertEqual(expanded_box([0, 0, 60, 60], 100, 60), [0, 0, 60, 60])
        with self.assertRaises(ValueError):
            expanded_box([0, 0, 20, 10], 100, 60)

    def test_expansion_preserves_old_ids_and_exact_quotas(self):
        rows, old = [], []
        for cls in range(1, 121):
            quota, previous = (514, 128) if cls in FOCUS else (34, 8)
            for index in range(quota):
                sid = f'S{index:03d}C001P001R001A{cls:03d}'
                rows.append({'video_path': f'{sid}_rgb.avi', 'class_id': cls})
                if index < previous:
                    old.append(sid)
        selected = expanded_selection({'train': rows}, old)
        self.assertEqual(len(set(selected)), 9120)
        self.assertTrue(set(old) <= set(selected))
        self.assertEqual(selected, expanded_selection({'train': rows[::-1]}, old))
        for cls in range(1, 121):
            self.assertEqual(sum(sid.endswith(f'A{cls:03d}') for sid in selected), 512 if cls in FOCUS else 32)

    def test_completed_stage_rejects_changed_artifacts(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root/'experiment_lock.json').write_text(json.dumps({'seed': 42}))
            (root/'fusion').mkdir()
            (root/'fusion/report.json').write_text('{}')
            finish_stage(root, 'fusion')
            self.assertTrue(stage_complete(root, 'fusion'))
            (root/'fusion/report.json').write_text('{"changed": true}')
            with self.assertRaises(ValueError):
                stage_complete(root, 'fusion')


if __name__ == '__main__':
    unittest.main()
