"""Reduced NTU120 taxonomy with internal irrelevant groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset


SOURCE_CLASS_IDS = tuple(range(1, 121))
IRRELEVANT_GROUPS: dict[int, tuple[int, ...]] = {
    121: (69, 70, 71, 72, 77, 93),
    122: (73, 74, 76, 81, 82, 83, 84, 91),
    123: (65, 66, 94),
}
IRRELEVANT_GROUP_NAMES = {
    121: "irrelevant_hand_sign",
    122: "irrelevant_fine_object",
    123: "irrelevant_specialized",
}
PUBLIC_IRRELEVANT_LABEL = "irrelevant"
REVIEW_CLASS_IDS = (75, 78, 86)


@dataclass(frozen=True)
class ActionTaxonomy:
    """Map the 120 source actions to named actions plus internal groups."""

    source_class_ids: tuple[int, ...]
    output_class_ids: tuple[int, ...]
    source_to_output: Mapping[int, int]
    output_names: Mapping[int, str]
    irrelevant_output_ids: tuple[int, ...]

    def output_index(self, source_class_id: int) -> int:
        output_id = self.source_to_output[source_class_id]
        return self.output_class_ids.index(output_id)

    def public_label(self, output_class_id: int) -> str:
        if output_class_id in self.irrelevant_output_ids:
            return PUBLIC_IRRELEVANT_LABEL
        return f"A{output_class_id:03d}"

    def loss_weights(self) -> Tensor:
        """Keep each original source class's total CE influence comparable."""
        group_sizes = {
            output_id: sum(
                mapped == output_id for mapped in self.source_to_output.values()
            )
            for output_id in self.output_class_ids
        }
        return torch.tensor(
            [1.0 / group_sizes[output_id] for output_id in self.output_class_ids],
            dtype=torch.float32,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "source_class_ids": list(self.source_class_ids),
            "output_class_ids": list(self.output_class_ids),
            "source_to_output": {
                str(key): value for key, value in self.source_to_output.items()
            },
            "output_names": {
                str(key): value for key, value in self.output_names.items()
            },
            "irrelevant_output_ids": list(self.irrelevant_output_ids),
            "public_irrelevant_label": PUBLIC_IRRELEVANT_LABEL,
            "review_class_ids": list(REVIEW_CLASS_IDS),
            "loss_weights": self.loss_weights().tolist(),
        }


def build_reduced_action_taxonomy() -> ActionTaxonomy:
    grouped = {
        source_id: output_id
        for output_id, source_ids in IRRELEVANT_GROUPS.items()
        for source_id in source_ids
    }
    if set(grouped) & set(REVIEW_CLASS_IDS):
        raise RuntimeError("Review classes must remain named outputs")
    retained = tuple(class_id for class_id in SOURCE_CLASS_IDS if class_id not in grouped)
    output_ids = retained + tuple(IRRELEVANT_GROUPS)
    source_to_output = {
        class_id: grouped.get(class_id, class_id) for class_id in SOURCE_CLASS_IDS
    }
    output_names = {
        **{class_id: f"A{class_id:03d}" for class_id in retained},
        **IRRELEVANT_GROUP_NAMES,
    }
    return ActionTaxonomy(
        source_class_ids=SOURCE_CLASS_IDS,
        output_class_ids=output_ids,
        source_to_output=source_to_output,
        output_names=output_names,
        irrelevant_output_ids=tuple(IRRELEVANT_GROUPS),
    )


class RemappedLabelDataset(Dataset):
    """Replace a dataset's class-index target while preserving its inputs."""

    def __init__(self, base: Dataset, source_class_ids: Sequence[int], taxonomy: ActionTaxonomy) -> None:
        if len(base) != len(source_class_ids):
            raise ValueError("Dataset and source class IDs differ in length")
        self.base = base
        self.labels = tuple(taxonomy.output_index(class_id) for class_id in source_class_ids)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[object, ...]:
        row = self.base[index]
        if not isinstance(row, tuple) or len(row) < 2:
            raise ValueError("Expected dataset rows with input and label")
        return row[0], torch.tensor(self.labels[index], dtype=torch.long), *row[2:]


def aggregate_source_logits(source_logits: Tensor, taxonomy: ActionTaxonomy) -> Tensor:
    """Aggregate 120-class logits into taxonomy outputs by probability mass."""
    if source_logits.ndim != 2 or source_logits.shape[1] != len(taxonomy.source_class_ids):
        raise ValueError("Expected source logits shaped [N, 120]")
    columns = []
    for output_id in taxonomy.output_class_ids:
        source_indices = [
            index
            for index, source_id in enumerate(taxonomy.source_class_ids)
            if taxonomy.source_to_output[source_id] == output_id
        ]
        columns.append(torch.logsumexp(source_logits[:, source_indices], dim=1))
    return torch.stack(columns, dim=1)


def remap_source_labels(source_labels: Tensor, taxonomy: ActionTaxonomy) -> Tensor:
    """Map zero-based source label indices to zero-based taxonomy indices."""
    if source_labels.ndim != 1:
        raise ValueError("Expected rank-1 source labels")
    mapped = [taxonomy.output_index(taxonomy.source_class_ids[int(index)]) for index in source_labels]
    return torch.tensor(mapped, dtype=torch.long, device=source_labels.device)


def annotate_output_prediction(
    source_class_id: int,
    predicted_output_id: int,
    taxonomy: ActionTaxonomy,
) -> dict[str, object]:
    """Return internal and public labels for one reduced-taxonomy prediction."""
    true_output_id = taxonomy.source_to_output[source_class_id]
    return {
        "true_output_class_id": true_output_id,
        "true_output_name": taxonomy.output_names[true_output_id],
        "true_public_label": taxonomy.public_label(true_output_id),
        "predicted_output_name": taxonomy.output_names[predicted_output_id],
        "predicted_public_label": taxonomy.public_label(predicted_output_id),
    }


def public_prediction_metrics(
    source_class_ids: Sequence[int],
    predicted_output_ids: Sequence[int],
    taxonomy: ActionTaxonomy,
) -> dict[str, float | int]:
    """Measure named-action accuracy and public irrelevant behavior."""
    if len(source_class_ids) != len(predicted_output_ids):
        raise ValueError("Truth and prediction lengths differ")
    irrelevant = set(taxonomy.irrelevant_output_ids)
    relevant_total = irrelevant_total = 0
    relevant_correct = irrelevant_detected = relevant_rejected = public_correct = 0
    for source_id, predicted_id in zip(source_class_ids, predicted_output_ids):
        true_output = taxonomy.source_to_output[source_id]
        if true_output in irrelevant:
            irrelevant_total += 1
            if predicted_id in irrelevant:
                irrelevant_detected += 1
                public_correct += 1
        else:
            relevant_total += 1
            if predicted_id == true_output:
                relevant_correct += 1
                public_correct += 1
            if predicted_id in irrelevant:
                relevant_rejected += 1
    total = relevant_total + irrelevant_total
    return {
        "public_accuracy": public_correct / total if total else 0.0,
        "relevant_named_accuracy": relevant_correct / relevant_total if relevant_total else 0.0,
        "irrelevant_public_recall": irrelevant_detected / irrelevant_total if irrelevant_total else 0.0,
        "relevant_false_irrelevant_rate": relevant_rejected / relevant_total if relevant_total else 0.0,
        "relevant_samples": relevant_total,
        "irrelevant_samples": irrelevant_total,
    }
