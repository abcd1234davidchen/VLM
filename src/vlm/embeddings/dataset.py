"""PyTorch dataset backed by cached embedding tensors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


@dataclass
class CachedSample:
    embedding_path: Path
    video_path: Path
    class_id: int
    class_index: int
    split: str
    valid_length: int
    augmentation_name: str | None = None

    def to_record(self) -> dict[str, object]:
        return {
            "embedding_path": str(self.embedding_path),
            "video_path": str(self.video_path),
            "class_id": self.class_id,
            "class_index": self.class_index,
            "split": self.split,
            "valid_length": self.valid_length,
            "augmentation_name": self.augmentation_name,
        }


class CachedEmbeddingDataset(Dataset):
    """Load cached embedding payloads into memory once."""

    def __init__(self, samples: list[CachedSample]) -> None:
        self.samples = samples
        self.items = [
            torch.load(sample.embedding_path, map_location="cpu")
            for sample in samples
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.items[index]
        embeddings = item["embeddings"].float()
        valid_length = torch.tensor(item["valid_length"], dtype=torch.long)
        label = torch.tensor(
            self.samples[index].class_index,
            dtype=torch.long,
        )
        return embeddings, valid_length, label


def collate_cached_embeddings(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad only to the longest sequence in the current batch."""
    embeddings, valid_lengths, labels = zip(*batch)
    return (
        pad_sequence(embeddings, batch_first=True),
        torch.stack(valid_lengths),
        torch.stack(labels),
    )
