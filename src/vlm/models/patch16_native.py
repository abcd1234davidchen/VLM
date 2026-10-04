"""Opt-in native CLS attention without changing the shared RGB/anchor model."""
from __future__ import annotations

import torch
from transformers import XCLIPModel

from vlm.models.xclip_vision_lora import XCLIPVisionLoRAClassifier


class XCLIPNativeLoRAClassifier(XCLIPVisionLoRAClassifier):
    """The original RGB state keys; native maps are returned only on request."""

    def __init__(self, xclip: XCLIPModel, **kwargs):
        xclip.set_attn_implementation('eager')
        super().__init__(xclip, **kwargs)

    def forward(self, pixel_values, *, return_features=False, return_native_attentions=False):
        if return_features and return_native_attentions:
            raise ValueError('Request features or native attentions, not both')
        if not return_native_attentions:
            return super().forward(pixel_values, return_features=return_features)
        if pixel_values.ndim != 5:
            raise ValueError('Expected RGB [batch, frames, channels, height, width]')
        batch, frames, _, height, width = pixel_values.shape
        patch_size = int(self.vision_model.config.patch_size)
        if height % patch_size or width % patch_size:
            raise ValueError('Input dimensions must be divisible by patch size')
        patches = (height // patch_size) * (width // patch_size)
        vision = self.vision_model(pixel_values=pixel_values.flatten(0, 1),
                                   return_dict=True, output_attentions=True)
        if vision.attentions is None or len(vision.attentions) < 2:
            raise RuntimeError('Native supervision requires eager attention and at least two layers')
        rows = []
        for attention in vision.attentions[-2:]:
            if attention is None or attention.shape[-2:] != (patches + 2, patches + 2) or attention.shape[1] < 4:
                raise RuntimeError('Expected CLS, patches, message and at least four heads')
            rows.append(attention[:, :4, 0, :].float().mean(1))
        projected = self.visual_projection(vision.pooler_output)
        if self.training and self.feature_noise_std:
            projected = projected + torch.randn_like(projected) * self.feature_noise_std
        video = self.mit(projected.reshape(batch, frames, -1), return_dict=True)
        logits = self.classifier(video.pooler_output)
        return logits, torch.stack(rows, dim=1).reshape(batch, frames, 2, patches + 2)
