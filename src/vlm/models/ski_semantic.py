"""Partial-pair SKI-style semantic branch; RGB task inference stays unchanged."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from vlm.models.xclip_vision_lora import XCLIPVisionLoRAClassifier


class SKISemanticClassifier(XCLIPVisionLoRAClassifier):
    def __init__(self, xclip, *, text_prototypes: torch.Tensor, semantic_scale: float, **kwargs) -> None:
        super().__init__(xclip, **kwargs)
        if text_prototypes.shape != (59, int(xclip.projection_dim)):
            raise ValueError("Expected ordered A001-A059 text prototypes")
        self.register_buffer("text_prototypes", F.normalize(text_prototypes.float(), dim=-1))
        self.register_buffer("semantic_scale", torch.tensor(semantic_scale))
        self.semantic_projection = nn.Linear(int(xclip.projection_dim), int(xclip.projection_dim))
        with torch.no_grad():
            self.semantic_projection.weight.copy_(torch.eye(int(xclip.projection_dim)))
            self.semantic_projection.bias.zero_()

    def semantic_logits(self, features: torch.Tensor) -> torch.Tensor:
        embedding = F.normalize(self.semantic_projection(features).float(), dim=-1)
        return self.semantic_scale * embedding @ self.text_prototypes.t()

    def forward(self, pixel_values: torch.Tensor, *, return_semantic: bool = False,
                return_features: bool = False):
        logits, features = super().forward(pixel_values, return_features=True)
        if return_features:
            return logits, features
        return (logits, self.semantic_logits(features)) if return_semantic else logits
