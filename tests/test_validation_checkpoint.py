from __future__ import annotations

import unittest

import torch
from torch import nn

from vlm.training.engine import ValidationCheckpoint


class ValidationCheckpointTest(unittest.TestCase):
    def test_restores_best_epoch_instead_of_last_epoch(self) -> None:
        model = nn.Linear(1, 1, bias=False)
        checkpoint = ValidationCheckpoint()

        with torch.no_grad():
            model.weight.fill_(1.0)
        self.assertTrue(checkpoint.consider(model, epoch=1, score=0.8))

        with torch.no_grad():
            model.weight.fill_(2.0)
        self.assertFalse(checkpoint.consider(model, epoch=2, score=0.7))

        with torch.no_grad():
            model.weight.fill_(3.0)
        checkpoint.restore(model)

        self.assertEqual(checkpoint.epoch, 1)
        self.assertEqual(model.weight.item(), 1.0)

    def test_uses_lexicographic_metric_tiebreakers(self) -> None:
        model = nn.Linear(1, 1, bias=False)
        checkpoint = ValidationCheckpoint()

        checkpoint.consider(model, epoch=1, score=(0.7, 0.6, 0.9))
        updated = checkpoint.consider(
            model,
            epoch=2,
            score=(0.7, 0.7, 0.5),
        )

        self.assertTrue(updated)
        self.assertEqual(checkpoint.epoch, 2)


if __name__ == "__main__":
    unittest.main()
