"""Lossless, per-video XCLIP processor output cache for RGB training."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

import torch

from vlm.xclip_features import FRAME_SAMPLING_STRATEGY, read_uniform_video_frames


class ProcessedRGBCache:
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
        }
        self.config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()

    def path_for(self, video_path: Path) -> Path:
        source = str(video_path.resolve())
        path_hash = hashlib.sha256(source.encode()).hexdigest()
        return self.root / self.config_hash / path_hash[:2] / f"{path_hash}.pt"

    def _source(self, video_path: Path) -> dict[str, object]:
        stat = video_path.stat()
        return {"path": str(video_path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    def load(self, video_path: Path) -> torch.Tensor:
        path = self.path_for(video_path)
        source = self._source(video_path)
        if path.exists():
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
                pixels = payload["pixels"]
                if (payload["config_hash"] == self.config_hash and payload["source"] == source
                        and isinstance(pixels, torch.Tensor) and pixels.dtype == torch.float32
                        and tuple(pixels.shape) == (8, 3, 224, 224)):
                    return pixels
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                pass
        frames = read_uniform_video_frames(video_path, 8)
        pixels = self.processor(images=list(frames), return_tensors="pt")["pixel_values"].squeeze(0)
        if pixels.dtype != torch.float32 or tuple(pixels.shape) != (8, 3, 224, 224):
            raise ValueError(f"Unexpected processed RGB shape or dtype: {tuple(pixels.shape)}, {pixels.dtype}")
        if self._source(video_path) != source:
            raise RuntimeError(f"Source video changed during extraction: {video_path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            torch.save({"config_hash": self.config_hash, "source": source, "pixels": pixels}, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return pixels
