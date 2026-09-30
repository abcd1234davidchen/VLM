from __future__ import annotations

import unittest

import torch
from torch.utils.data import TensorDataset

from vlm.action_taxonomy import (
    IRRELEVANT_GROUPS,
    REVIEW_CLASS_IDS,
    RemappedLabelDataset,
    aggregate_source_logits,
    annotate_output_prediction,
    build_reduced_action_taxonomy,
    public_prediction_metrics,
    remap_source_labels,
)


class ActionTaxonomyTest(unittest.TestCase):
    def test_contract_has_103_named_and_three_internal_outputs(self) -> None:
        taxonomy = build_reduced_action_taxonomy()
        self.assertEqual(len(taxonomy.source_class_ids), 120)
        self.assertEqual(len(taxonomy.output_class_ids), 106)
        self.assertEqual(len(set(taxonomy.source_to_output.values())), 106)
        self.assertTrue(set(REVIEW_CLASS_IDS).isdisjoint({source for rows in IRRELEVANT_GROUPS.values() for source in rows}))
        for class_id in REVIEW_CLASS_IDS:
            self.assertEqual(taxonomy.source_to_output[class_id], class_id)
        for output_id, source_ids in IRRELEVANT_GROUPS.items():
            self.assertTrue(all(taxonomy.source_to_output[source_id] == output_id for source_id in source_ids))
            self.assertEqual(taxonomy.public_label(output_id), "irrelevant")
            self.assertAlmostEqual(
                taxonomy.loss_weights()[taxonomy.output_class_ids.index(output_id)].item(),
                1 / len(source_ids),
            )

    def test_probability_mass_and_labels_are_aggregated(self) -> None:
        taxonomy = build_reduced_action_taxonomy()
        logits = torch.full((2, 120), -100.0)
        logits[0, 68] = 2.0
        logits[0, 69] = 1.0
        logits[1, 74] = 3.0
        aggregated = aggregate_source_logits(logits, taxonomy)
        hand_index = taxonomy.output_class_ids.index(121)
        self.assertTrue(torch.allclose(aggregated[0, hand_index], torch.logsumexp(torch.tensor([2.0, 1.0]), 0)))
        mapped = remap_source_labels(torch.tensor([68, 74]), taxonomy)
        self.assertEqual(mapped.tolist(), [hand_index, taxonomy.output_class_ids.index(75)])

    def test_dataset_replaces_only_the_label(self) -> None:
        taxonomy = build_reduced_action_taxonomy()
        base = TensorDataset(torch.tensor([[1.0], [2.0]]), torch.tensor([68, 74]))
        dataset = RemappedLabelDataset(base, [69, 75], taxonomy)
        self.assertEqual(dataset[0][0].item(), 1.0)
        self.assertEqual(dataset[0][1].item(), taxonomy.output_class_ids.index(121))
        self.assertEqual(dataset[1][1].item(), taxonomy.output_class_ids.index(75))

    def test_public_prediction_contract(self) -> None:
        taxonomy = build_reduced_action_taxonomy()
        row = annotate_output_prediction(69, 122, taxonomy)
        self.assertEqual(row["true_output_class_id"], 121)
        self.assertEqual(row["predicted_output_name"], "irrelevant_fine_object")
        self.assertEqual(row["predicted_public_label"], "irrelevant")
        metrics = public_prediction_metrics([69, 75], [122, 121], taxonomy)
        self.assertEqual(metrics["irrelevant_public_recall"], 1.0)
        self.assertEqual(metrics["relevant_false_irrelevant_rate"], 1.0)
        self.assertEqual(metrics["public_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
