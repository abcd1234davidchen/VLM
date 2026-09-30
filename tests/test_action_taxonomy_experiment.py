from __future__ import annotations

import unittest

import torch

from vlm.action_taxonomy import build_reduced_action_taxonomy
from vlm.experiments.action_taxonomy import evaluate_taxonomy


class ActionTaxonomyExperimentTest(unittest.TestCase):
    def test_public_irrelevant_merges_internal_groups(self) -> None:
        taxonomy = build_reduced_action_taxonomy()
        logits = torch.full((3, 120), -100.0)
        labels = torch.tensor([68, 72, 0])  # A069, A073, A001
        logits[0, 72] = 5.0  # true hand-sign sample predicted fine-object, same public label
        logits[1, 72] = 5.0  # A073 exact
        logits[2, 68] = 5.0  # relevant sample falsely predicted irrelevant
        report, _, _ = evaluate_taxonomy(logits, labels, taxonomy)
        self.assertAlmostEqual(report["exact_106_class_accuracy"], 1 / 3)
        self.assertAlmostEqual(report["public_accuracy"], 2 / 3)
        self.assertEqual(report["irrelevant_public_recall"], 1.0)
        self.assertEqual(report["relevant_false_irrelevant_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
