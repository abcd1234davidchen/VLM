"""RGB-only temporal gate, preserved in the merged deployment model."""

import torch
from torch import nn

from vlm.models.xclip_vision_lora import XCLIPVisionClassifier, XCLIPVisionLoRAClassifier


class TemporalGate(nn.Module):
    def __init__(self, dimension=512):
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(dimension, 128), nn.GELU(), nn.Linear(128, 1))
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)

    def forward(self, frames):
        weights = self.scorer(frames).squeeze(-1).float().softmax(-1)
        return frames * (1 + weights.unsqueeze(-1) - 1 / frames.shape[1]), weights


class TemporalForward:
    def forward(self, pixel_values, *, return_temporal=False):
        if pixel_values.ndim != 5 or pixel_values.shape[1] != 8:
            raise ValueError("Expected eight RGB frames [B,8,3,H,W]")
        batch, frames = pixel_values.shape[:2]
        vision = self.vision_model(pixel_values=pixel_values.flatten(0, 1), return_dict=True)
        projected = self.visual_projection(vision.pooler_output).reshape(batch, frames, -1)
        gated, weights = self.temporal_gate(projected)
        video = self.mit(gated, return_dict=True)
        logits = self.classifier(video.pooler_output)
        return (logits, weights) if return_temporal else logits


class XCLIPTemporalClassifier(TemporalForward, XCLIPVisionClassifier):
    def __init__(self, xclip, **kwargs):
        super().__init__(xclip, **kwargs)
        self.temporal_gate = TemporalGate(int(xclip.projection_dim))


class XCLIPTemporalLoRAClassifier(TemporalForward, XCLIPVisionLoRAClassifier):
    def __init__(self, xclip, **kwargs):
        super().__init__(xclip, **kwargs)
        self.temporal_gate = TemporalGate(int(xclip.projection_dim))
