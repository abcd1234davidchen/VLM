"""Exact XCLIP RGB pixels reconstructed from compact uint8 resized crops."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np
import torch

from vlm.xclip_features import FRAME_SAMPLING_STRATEGY, read_uniform_video_frames


class ProcessedUInt8Cache:
    def __init__(self, root: Path, processor: object, *, model_name: str, model_revision: str) -> None:
        self.root = root
        self.processor = processor
        config = {
            "schema_version": 1,
            "model_name": model_name,
            "model_revision": model_revision,
            "image_processor": processor.image_processor.to_dict(),
            "frame_sampling_strategy": FRAME_SAMPLING_STRATEGY,
            "num_frames": 8,
            "payload": "uint8_post_resize_crop_pre_rescale_normalize",
        }
        self.config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()

    def path_for(self, video_path: Path) -> Path:
        source = video_path.resolve()
        stat = source.stat()
        path_hash = hashlib.sha256(str(source).encode()).hexdigest()
        return self.root / self.config_hash / path_hash[:2] / f"{path_hash}_{stat.st_size}_{stat.st_mtime_ns}.npy"

    def load(self, video_path: Path) -> torch.Tensor:
        path = self.path_for(video_path)
        compact = None
        if path.exists():
            try:
                compact = np.load(path, mmap_mode="r", allow_pickle=False)
                if compact.dtype != np.uint8 or compact.shape != (8, 3, 224, 224):
                    compact = None
            except (OSError, ValueError, EOFError):
                compact = None
        if compact is None:
            frames = read_uniform_video_frames(video_path, 8)
            compact = self.processor(images=list(frames), return_tensors="np", do_rescale=False, do_normalize=False)["pixel_values"][0]
            if compact.dtype != np.uint8 or compact.shape != (8, 3, 224, 224):
                raise ValueError(f"Unexpected uint8 crop format for {video_path}: {compact.shape}, {compact.dtype}")
            if self.path_for(video_path) != path:
                raise RuntimeError(f"Source video changed during extraction: {video_path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.stem}.{os.getpid()}.{uuid.uuid4().hex}.npy")
            try:
                with temporary.open("wb") as stream:
                    np.save(stream, compact, allow_pickle=False)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        pixels = self.processor(
            images=list(compact.transpose(0, 2, 3, 1)),
            return_tensors="pt", do_resize=False, do_center_crop=False,
        )["pixel_values"].squeeze(0)
        if not isinstance(pixels, torch.Tensor) or pixels.dtype != torch.float32 or pixels.shape != (8, 3, 224, 224):
            raise ValueError(f"Unexpected reconstructed RGB pixels for {video_path}")
        return pixels
