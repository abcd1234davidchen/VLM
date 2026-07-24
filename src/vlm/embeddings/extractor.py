"""Generate normalized CLIP frame embeddings for one video."""

from __future__ import annotations

import torch
from transformers import CLIPModel, CLIPProcessor

from vlm.data import (
    VideoSample,
    resize_embedding_sequence,
    sample_video_frames,
)

from .augmentation import FrameAugmentation, apply_frame_augmentation


def embed_video(
    sample: VideoSample,
    *,
    processor: CLIPProcessor,
    model: CLIPModel,
    device: torch.device,
    frame_stride: int,
    max_seq_len: int,
    batch_size: int,
    augmentation: FrameAugmentation | None = None,
) -> tuple[torch.Tensor, int]:
    frames = sample_video_frames(sample.video_path, frame_stride)
    if augmentation is not None:
        frames = apply_frame_augmentation(frames, augmentation)

    features: list[torch.Tensor] = []
    for start in range(0, len(frames), batch_size):
        inputs = processor(
            images=frames[start : start + batch_size],
            return_tensors="pt",
        )
        pixel_values = inputs["pixel_values"].to(device)
        with torch.inference_mode():
            batch_features = model.get_image_features(
                pixel_values=pixel_values
            )
        if hasattr(batch_features, "pooler_output"):
            batch_features = batch_features.pooler_output
        features.append(
            torch.nn.functional.normalize(batch_features, dim=-1).cpu()
        )

    stacked = torch.cat(features, dim=0)
    return resize_embedding_sequence(stacked, max_seq_len)
