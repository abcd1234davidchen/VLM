"""Bidirectional GRU with learnable normalized class prototypes."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .temporal import masked_mean_max_pool


class PrototypeVideoClassifier(nn.Module):
    """Encode a video and score it against learnable class prototypes."""

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
        self.feature_norm = nn.LayerNorm(pooled_dim)
        self.feature_dropout = nn.Dropout(dropout)
        self.prototypes = nn.Parameter(torch.empty(num_classes, pooled_dim))
        self.logit_scale = nn.Parameter(torch.tensor(2.0))
        torch.nn.init.normal_(self.prototypes, mean=0.0, std=0.02)

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
        pooled = masked_mean_max_pool(output, lengths)
        pooled = self.feature_norm(pooled)
        pooled = self.feature_dropout(pooled)
        pooled = torch.nn.functional.normalize(pooled, dim=-1)
        prototypes = torch.nn.functional.normalize(
            self.prototypes,
            dim=-1,
        )
        return (
            self.logit_scale.exp().clamp(max=100.0)
            * pooled
            @ prototypes.t()
        )
