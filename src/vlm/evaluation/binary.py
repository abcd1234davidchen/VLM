"""Binary safe/danger metrics and per-video predictions."""

from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import DataLoader

from vlm.embeddings import CachedSample


BINARY_CLASS_NAMES = ("safe", "danger")


def binary_class_name(class_index: int) -> str:
    return BINARY_CLASS_NAMES[class_index]


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
) -> dict[str, object]:
    confusion = [
        [0 for _ in BINARY_CLASS_NAMES]
        for _ in BINARY_CLASS_NAMES
    ]
    for truth, predicted in zip(y_true, y_pred):
        confusion[truth][predicted] += 1

    total = len(y_true)
    correct = sum(
        confusion[index][index]
        for index in range(len(BINARY_CLASS_NAMES))
    )
    per_class: dict[str, dict[str, float | int]] = {}
    for index, name in enumerate(BINARY_CLASS_NAMES):
        true_positive = confusion[index][index]
        false_positive = (
            sum(
                confusion[row][index]
                for row in range(len(BINARY_CLASS_NAMES))
            )
            - true_positive
        )
        false_negative = sum(confusion[index]) - true_positive
        support = sum(confusion[index])
        precision = safe_divide(
            true_positive,
            true_positive + false_positive,
        )
        recall = safe_divide(
            true_positive,
            true_positive + false_negative,
        )
        f1 = safe_divide(2 * precision * recall, precision + recall)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    macro_f1 = (
        sum(metrics["f1"] for metrics in per_class.values())
        / len(BINARY_CLASS_NAMES)
    )
    weighted_f1 = safe_divide(
        sum(
            metrics["f1"] * metrics["support"]
            for metrics in per_class.values()
        ),
        total,
    )
    balanced_accuracy = (
        sum(metrics["recall"] for metrics in per_class.values())
        / len(BINARY_CLASS_NAMES)
    )
    return {
        "accuracy": safe_divide(correct, total),
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": confusion,
        "confusion_matrix_labels": list(BINARY_CLASS_NAMES),
        "per_class": per_class,
    }


def validation_selection_score(
    metrics: dict[str, object],
) -> tuple[float, ...]:
    return (
        float(metrics["macro_f1"]),
        float(metrics["balanced_accuracy"]),
        float(metrics["accuracy"]),
    )


def evaluate_binary(
    model: nn.Module,
    loader: DataLoader,
    samples: list[CachedSample],
    device: torch.device,
    danger_threshold: float = 0.5,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    predictions: list[dict[str, object]] = []
    original_class_correct: defaultdict[int, int] = defaultdict(int)
    original_class_total: defaultdict[int, int] = defaultdict(int)
    sample_index = 0

    with torch.inference_mode():
        for embeddings, lengths, labels in loader:
            embeddings = embeddings.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            probabilities = model(embeddings, lengths).softmax(dim=1)
            predicted = (
                probabilities[:, 1] >= danger_threshold
            ).long()

            for truth, prediction, probs in zip(
                labels.cpu().tolist(),
                predicted.cpu().tolist(),
                probabilities.cpu().tolist(),
            ):
                sample = samples[sample_index]
                y_true.append(truth)
                y_pred.append(prediction)
                original_class_total[sample.class_id] += 1
                if truth == prediction:
                    original_class_correct[sample.class_id] += 1
                predictions.append(
                    {
                        "video": sample.video_path.name,
                        "path": str(sample.video_path),
                        "original_class_id": sample.class_id,
                        "true_label": binary_class_name(truth),
                        "predicted_label": binary_class_name(prediction),
                        "safe_probability": probs[0],
                        "danger_probability": probs[1],
                    }
                )
                sample_index += 1

    metrics = compute_metrics(y_true, y_pred)
    metrics["original_class_accuracy"] = {
        f"{class_id:03d}": {
            "accuracy": safe_divide(
                original_class_correct[class_id],
                total,
            ),
            "correct": original_class_correct[class_id],
            "total": total,
        }
        for class_id, total in sorted(original_class_total.items())
    }
    return metrics, predictions
