"""Classification head for frozen XCLIP video features."""

from __future__ import annotations

import torch
from torch import nn


class XCLIPFeatureClassifier(nn.Module):
    """Classify one fixed-size XCLIP representation per video."""

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)
