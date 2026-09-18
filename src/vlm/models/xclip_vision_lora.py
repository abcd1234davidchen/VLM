"""RGB-input XCLIP classifier with MIT and upper vision block LoRA."""

from __future__ import annotations

import torch
from torch import nn
from transformers import XCLIPModel

from .lora import (
    TrainableParameterAudit,
    audit_trainable_parameters,
    inject_mit_attention_lora,
    inject_vision_attention_lora,
)


class XCLIPVisionClassifier(nn.Module):
    def __init__(
        self,
        xclip: XCLIPModel,
        *,
        num_classes: int,
        head_dropout: float,
        feature_noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        xclip.requires_grad_(False)
        self.vision_model = xclip.vision_model
        self.visual_projection = xclip.visual_projection
        self.mit = xclip.mit
        self.feature_noise_std = feature_noise_std
        self.classifier = nn.Sequential(
            nn.LayerNorm(int(xclip.projection_dim)),
            nn.Dropout(head_dropout),
            nn.Linear(int(xclip.projection_dim), num_classes),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 5:
            raise ValueError("Expected RGB pixel_values [batch, frames, channels, height, width]")
        batch, frames = pixel_values.shape[:2]
        vision = self.vision_model(
            pixel_values=pixel_values.flatten(0, 1), return_dict=True
        )
        projected = self.visual_projection(vision.pooler_output)
        if self.training and self.feature_noise_std:
            projected = projected + torch.randn_like(projected) * self.feature_noise_std
        video = self.mit(projected.reshape(batch, frames, -1), return_dict=True)
        return self.classifier(video.pooler_output)


class XCLIPVisionLoRAClassifier(XCLIPVisionClassifier):
    def __init__(
        self,
        xclip: XCLIPModel,
        *,
        num_classes: int,
        head_dropout: float,
        upper_blocks: int,
        lora_rank: int,
        lora_alpha: float,
        lora_dropout: float,
        feature_noise_std: float = 0.0,
    ) -> None:
        super().__init__(xclip, num_classes=num_classes, head_dropout=head_dropout, feature_noise_std=feature_noise_std)
        self.target_modules = (
            inject_vision_attention_lora(
                self.vision_model,
                upper_blocks=upper_blocks,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )
            + inject_mit_attention_lora(
                self.mit,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )
        )
        unexpected = [
            name for name, parameter in self.named_parameters()
            if parameter.requires_grad
            and not (name.endswith("lora_a") or name.endswith("lora_b") or name.startswith("classifier."))
        ]
        if unexpected:
            raise RuntimeError(f"Unexpected trainable parameters: {unexpected}")

    def trainable_parameter_audit(self) -> TrainableParameterAudit:
        return audit_trainable_parameters(self)
