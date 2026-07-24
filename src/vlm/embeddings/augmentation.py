"""Deterministic, video-level frame augmentations."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass

from PIL import Image, ImageEnhance, ImageOps

from vlm.data import VideoSample


@dataclass(frozen=True)
class FrameAugmentation:
    name: str
    seed: int
    rotation_degrees: float = 8.0
    color_jitter: float = 0.15
    crop_scale: float = 0.9

    def to_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "seed": self.seed,
            "rotation_degrees": self.rotation_degrees,
            "color_jitter": self.color_jitter,
            "crop_scale": self.crop_scale,
        }

    def cache_key(self) -> str:
        payload = json.dumps(self.to_metadata(), sort_keys=True)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        safe_name = "".join(
            character
            if character.isalnum() or character in "._-"
            else "_"
            for character in self.name
        )
        return f"{safe_name}_{digest}"


def stable_augmentation_seed(
    base_seed: int,
    split_name: str,
    sample: VideoSample,
    augmentation_name: str,
) -> int:
    payload = (
        f"{base_seed}|{split_name}|{sample.video_path}|"
        f"{sample.class_index}|{augmentation_name}"
    )
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def apply_frame_augmentation(
    frames: list[Image.Image],
    augmentation: FrameAugmentation,
) -> list[Image.Image]:
    if not frames:
        raise ValueError("Cannot augment an empty frame list")
    rng = random.Random(augmentation.seed)

    if augmentation.name == "horizontal_flip":
        return [ImageOps.mirror(frame) for frame in frames]
    if augmentation.name == "small_rotation":
        angle = rng.uniform(
            -augmentation.rotation_degrees,
            augmentation.rotation_degrees,
        )
        return [
            frame.rotate(angle, resample=Image.Resampling.BICUBIC)
            for frame in frames
        ]
    if augmentation.name == "color_jitter":
        low = 1.0 - augmentation.color_jitter
        high = 1.0 + augmentation.color_jitter
        brightness = rng.uniform(low, high)
        contrast = rng.uniform(low, high)
        return [
            ImageEnhance.Contrast(
                ImageEnhance.Brightness(frame).enhance(brightness)
            ).enhance(contrast)
            for frame in frames
        ]
    if augmentation.name == "random_crop":
        width, height = frames[0].size
        scale = rng.uniform(augmentation.crop_scale, 1.0)
        crop_width = max(1, int(width * scale))
        crop_height = max(1, int(height * scale))
        left = rng.randint(0, width - crop_width)
        top = rng.randint(0, height - crop_height)
        crop_box = (left, top, left + crop_width, top + crop_height)
        return [
            frame.crop(crop_box).resize(
                (width, height),
                Image.Resampling.BICUBIC,
            )
            for frame in frames
        ]
    raise ValueError(f"Unsupported frame augmentation: {augmentation.name}")
