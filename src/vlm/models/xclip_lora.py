"""XCLIP MIT LoRA classifier over cached projected frame features."""

from __future__ import annotations

import torch
from torch import nn
from transformers import XCLIPModel

from .lora import (
    TrainableParameterAudit,
    audit_trainable_parameters,
    inject_mit_attention_lora,
)


class XCLIPMITClassifier(nn.Module):
    """Classify cached projected frames with a plain, merge-ready MIT."""

    def __init__(
        self,
        xclip: XCLIPModel,
        *,
        num_classes: int,
        head_dropout: float,
    ) -> None:
        super().__init__()
        xclip.requires_grad_(False)
        self.mit = xclip.mit
        feature_dim = int(xclip.projection_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(head_dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        video_features = self.mit(features).pooler_output
        return self.classifier(video_features)


class XCLIPMITLoRAClassifier(XCLIPMITClassifier):
    def __init__(
        self,
        xclip: XCLIPModel,
        *,
        num_classes: int,
        head_dropout: float,
        lora_rank: int,
        lora_alpha: float,
        lora_dropout: float,
    ) -> None:
        super().__init__(
            xclip,
            num_classes=num_classes,
            head_dropout=head_dropout,
        )
        self.target_modules = inject_mit_attention_lora(
            self.mit,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
        self._assert_trainable_contract()

    def _assert_trainable_contract(self) -> None:
        unexpected = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
            and "lora_a" not in name
            and "lora_b" not in name
            and not name.startswith("classifier.")
        ]
        if unexpected:
            raise RuntimeError(
                "Unexpected trainable XCLIP parameters: "
                + ", ".join(unexpected)
            )

    def trainable_parameter_audit(self) -> TrainableParameterAudit:
        return audit_trainable_parameters(self)
