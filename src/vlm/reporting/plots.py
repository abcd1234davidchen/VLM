"""Reusable training-history plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_accuracy_plot(
    history: list[dict[str, float]],
    path: Path,
    *,
    train_key: str,
    evaluation_key: str,
    evaluation_label: str,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [record["epoch"] for record in history]
    train_accuracy = [record[train_key] for record in history]
    evaluation_accuracy = [
        record[evaluation_key]
        for record in history
    ]

    _, axis = plt.subplots(figsize=(10, 6))
    axis.plot(
        epochs,
        train_accuracy,
        label="Train Accuracy",
        marker="o",
    )
    axis.plot(
        epochs,
        evaluation_accuracy,
        label=evaluation_label,
        marker="o",
    )
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Accuracy")
    axis.legend()
    axis.grid(True)
    plt.savefig(path)
    plt.close()
