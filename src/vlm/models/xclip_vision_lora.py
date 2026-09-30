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

    def forward(self, pixel_values: torch.Tensor, *, return_features: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
        logits = self.classifier(video.pooler_output)
        return (logits, video.pooler_output) if return_features else logits


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


class XCLIPSpatialLoRAClassifier(XCLIPVisionLoRAClassifier):
    """RGB classifier with a training-only patch occupancy probe."""

    def __init__(self, xclip: XCLIPModel, **kwargs: object) -> None:
        super().__init__(xclip, **kwargs)
        self.spatial_head = nn.Linear(int(xclip.config.vision_config.hidden_size), 1)
        self.patch_size = int(xclip.config.vision_config.patch_size)

    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        return_spatial: bool = False,
        return_patch_tokens: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if pixel_values.ndim != 5:
            raise ValueError("Expected RGB pixel_values [batch, frames, channels, height, width]")
        batch, frames, _channels, height, width = pixel_values.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("Processor dimensions must be divisible by patch size")
        vision = self.vision_model(pixel_values=pixel_values.flatten(0, 1), return_dict=True)
        tokens = vision.last_hidden_state[:, 1:, :]
        expected_patches = (height // self.patch_size) * (width // self.patch_size)
        if tokens.shape[1] != expected_patches:
            raise RuntimeError(f"Expected {expected_patches} patch tokens, got {tokens.shape[1]}")
        projected = self.visual_projection(vision.pooler_output)
        if self.training and self.feature_noise_std:
            projected = projected + torch.randn_like(projected) * self.feature_noise_std
        video = self.mit(projected.reshape(batch, frames, -1), return_dict=True)
        logits = self.classifier(video.pooler_output)
        if not return_spatial and not return_patch_tokens:
            return logits
        outputs: list[torch.Tensor] = [logits]
        if return_spatial:
            outputs.append(self.spatial_head(tokens).reshape(batch, frames, expected_patches))
        if return_patch_tokens:
            outputs.append(tokens.reshape(batch, frames, expected_patches, -1))
        return tuple(outputs)


class XCLIPAttentionClassifier(XCLIPVisionClassifier):
    """RGB-only decision-coupled patch attention, identity at initialization.

    Patch pooling and CLS use the same frozen post-layer normalization so the
    residual mixes representations at the input of visual_projection.
    """

    def __init__(self, xclip: XCLIPModel, **kwargs: object) -> None:
        super().__init__(xclip, **kwargs)
        self.spatial_head = nn.Linear(int(xclip.config.vision_config.hidden_size), 1)
        self.pooling_alpha = nn.Parameter(torch.zeros(()))
        self.patch_size = int(xclip.config.vision_config.patch_size)

    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        return_spatial: bool = False,
        return_patch_tokens: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if pixel_values.ndim != 5:
            raise ValueError("Expected RGB pixel_values [batch, frames, channels, height, width]")
        batch, frames, _channels, height, width = pixel_values.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("Processor dimensions must be divisible by patch size")
        vision = self.vision_model(pixel_values=pixel_values.flatten(0, 1), return_dict=True)
        tokens = vision.last_hidden_state[:, 1:, :]
        patches = (height // self.patch_size) * (width // self.patch_size)
        if tokens.shape[1] != patches:
            raise RuntimeError(f"Expected {patches} patch tokens, got {tokens.shape[1]}")
        attention_logits = self.spatial_head(tokens).squeeze(-1)
        attention = attention_logits.float().softmax(-1).to(tokens.dtype)
        pooled = (attention.unsqueeze(-1) * tokens).sum(1)
        pooled = self.vision_model.post_layernorm(pooled)
        frame = vision.pooler_output + self.pooling_alpha * (pooled - vision.pooler_output)
        projected = self.visual_projection(frame)
        if self.training and self.feature_noise_std:
            projected = projected + torch.randn_like(projected) * self.feature_noise_std
        video = self.mit(projected.reshape(batch, frames, -1), return_dict=True)
        logits = self.classifier(video.pooler_output)
        outputs = [logits]
        if return_spatial:
            outputs.append(attention_logits.reshape(batch, frames, patches))
        if return_patch_tokens:
            outputs.append(tokens.reshape(batch, frames, patches, -1))
        return tuple(outputs) if len(outputs) > 1 else logits


class XCLIPAttentionLoRAClassifier(XCLIPAttentionClassifier, XCLIPVisionLoRAClassifier):
    """Attention pooling with the established upper vision and MIT LoRA whitelist."""
