"""Bidirectional GRU video classifier."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .temporal import masked_mean_max_pool


class GRUVideoClassifier(nn.Module):
    """Layer-normalized frame embeddings, packed BiGRU, and MLP logits."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.gru = nn.GRU(
            input_dim,
            hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        pooled_dim = hidden_size * 4
        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Dropout(dropout),
            nn.Linear(pooled_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.input_norm(embeddings)
        packed = pack_padded_sequence(
            embeddings,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.gru(packed)
        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
        )
        return self.classifier(masked_mean_max_pool(output, lengths))
