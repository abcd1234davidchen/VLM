"""Multiclass accuracy and per-video prediction rows."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from vlm.embeddings import CachedSample


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    samples: list[CachedSample],
    device: torch.device,
    class_ids: list[int],
) -> tuple[float, list[dict[str, object]]]:
    model.eval()
    predictions: list[dict[str, object]] = []
    correct = 0
    sample_index = 0

    with torch.inference_mode():
        for embeddings, lengths, labels in loader:
            embeddings = embeddings.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            predicted = model(embeddings, lengths).argmax(dim=1)
            correct += (predicted == labels).sum().item()

            for truth, prediction in zip(
                labels.cpu().tolist(),
                predicted.cpu().tolist(),
            ):
                sample = samples[sample_index]
                predictions.append(
                    {
                        "video": sample.video_path.name,
                        "true_class_id": sample.class_id,
                        "predicted_class_id": class_ids[prediction],
                        "true_class_index": truth,
                        "predicted_class_index": prediction,
                    }
                )
                sample_index += 1

    return correct / len(loader.dataset), predictions
