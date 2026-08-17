"""Frozen XCLIP video-feature extraction, caching, and datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import XCLIPModel, XCLIPProcessor

from vlm.data import VideoSample
from vlm.experiments.video_sampling import (
    read_video_pyav,
    uniform_frame_indices,
)


FRAME_SAMPLING_STRATEGY = "whole_video_uniform"
FEATURE_METHOD = "get_video_features"


@dataclass(frozen=True)
class CachedXCLIPFeature:
    feature_path: Path
    video_path: Path
    class_id: int
    class_index: int
    split: str


class CachedXCLIPFeatureDataset(Dataset):
    """Load fixed-size XCLIP video features into memory once."""

    def __init__(self, samples: list[CachedXCLIPFeature]) -> None:
        self.samples = samples
        self.items = [
            torch.load(sample.feature_path, map_location="cpu")
            for sample in samples
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.items[index]["feature"].float()
        label = torch.tensor(
            self.samples[index].class_index,
            dtype=torch.long,
        )
        return feature, label


def xclip_cache_config_key(model_name: str, num_frames: int) -> str:
    safe_model_name = "".join(
        character
        if character.isalnum() or character in "._-"
        else "_"
        for character in model_name
    )
    return (
        f"{safe_model_name}__{FRAME_SAMPLING_STRATEGY}"
        f"__frames_{num_frames}__{FEATURE_METHOD}"
    )


def build_xclip_cache_metadata(
    sample: VideoSample,
    *,
    model_name: str,
    num_frames: int,
) -> dict[str, object]:
    return {
        "model_name": model_name,
        "frame_sampling_strategy": FRAME_SAMPLING_STRATEGY,
        "num_frames": num_frames,
        "feature_method": FEATURE_METHOD,
        "video_path": str(sample.video_path),
    }


def get_xclip_feature_cache_path(
    cache_dir: Path,
    sample: VideoSample,
    *,
    model_name: str,
    num_frames: int,
) -> Path:
    path_hash = hashlib.sha256(
        sample.video_path.as_posix().encode("utf-8")
    ).hexdigest()[:12]
    stem = f"{sample.video_path.stem}__path_{path_hash}"
    return (
        cache_dir
        / "xclip_features"
        / xclip_cache_config_key(model_name, num_frames)
        / f"{stem}.pt"
    )


def is_xclip_cache_compatible(
    payload: dict[str, object],
    sample: VideoSample,
    *,
    model_name: str,
    num_frames: int,
) -> bool:
    feature = payload.get("feature")
    metadata = payload.get("cache_metadata")
    return (
        isinstance(feature, torch.Tensor)
        and feature.ndim == 1
        and feature.numel() > 0
        and metadata
        == build_xclip_cache_metadata(
            sample,
            model_name=model_name,
            num_frames=num_frames,
        )
    )


def read_uniform_video_frames(
    video_path: Path,
    num_frames: int,
) -> np.ndarray:
    """Decode a deterministic fixed-size view across the complete video."""
    with av.open(str(video_path)) as container:
        total_frames = container.streams.video[0].frames
        if total_frames > 0:
            indices = uniform_frame_indices(num_frames, total_frames)
            return read_video_pyav(container, indices)

        decoded = [
            frame.to_ndarray(format="rgb24")
            for frame in container.decode(video=0)
        ]
    if not decoded:
        raise ValueError(f"Video contains no decodable frames: {video_path}")
    indices = uniform_frame_indices(num_frames, len(decoded))
    return np.stack([decoded[int(index)] for index in indices])


def extract_xclip_feature(
    sample: VideoSample,
    *,
    processor: XCLIPProcessor,
    model: XCLIPModel,
    device: torch.device,
    num_frames: int,
) -> torch.Tensor:
    frames = read_uniform_video_frames(sample.video_path, num_frames)
    inputs = processor(images=list(frames), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.inference_mode():
        output = model.get_video_features(pixel_values=pixel_values)
    feature = output.pooler_output
    if feature.ndim != 2 or feature.shape[0] != 1:
        raise ValueError(
            "XCLIP get_video_features returned an unexpected shape: "
            f"{tuple(feature.shape)}"
        )
    return feature[0].detach().cpu()


def write_xclip_cache_manifest(
    cache_dir: Path,
    cached: dict[str, list[CachedXCLIPFeature]],
    *,
    model_name: str,
    num_frames: int,
) -> None:
    records = [
        {
            "feature_path": str(sample.feature_path),
            "video_path": str(sample.video_path),
            "class_id": sample.class_id,
            "split": sample.split,
        }
        for samples in cached.values()
        for sample in samples
    ]
    payload = {
        "cache_properties": {
            "model_name": model_name,
            "frame_sampling_strategy": FRAME_SAMPLING_STRATEGY,
            "num_frames": num_frames,
            "feature_method": FEATURE_METHOD,
            "config_key": xclip_cache_config_key(
                model_name,
                num_frames,
            ),
        },
        "features": sorted(
            records,
            key=lambda record: str(record["video_path"]),
        ),
    }
    manifest_path = cache_dir / "xclip_feature_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def extract_or_load_xclip_features(
    split: dict[str, list[VideoSample]],
    *,
    cache_dir: Path,
    model_name: str,
    num_frames: int,
    device: torch.device,
    force_extract: bool,
) -> dict[str, list[CachedXCLIPFeature]]:
    """Load compatible frozen features and extract only missing entries."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    processor: XCLIPProcessor | None = None
    model: XCLIPModel | None = None
    cached = {split_name: [] for split_name in split}
    refreshed_cache_files = 0

    progress = tqdm(
        total=sum(len(samples) for samples in split.values()),
        desc="XCLIP feature cache",
        unit="video",
        leave=True,
    )
    try:
        for split_name, samples in split.items():
            for sample in samples:
                cache_path = get_xclip_feature_cache_path(
                    cache_dir,
                    sample,
                    model_name=model_name,
                    num_frames=num_frames,
                )
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                payload: dict[str, object] | None = None
                should_extract = force_extract or not cache_path.exists()
                if not should_extract:
                    payload = torch.load(cache_path, map_location="cpu")
                    should_extract = not is_xclip_cache_compatible(
                        payload,
                        sample,
                        model_name=model_name,
                        num_frames=num_frames,
                    )

                if should_extract:
                    if processor is None or model is None:
                        processor = XCLIPProcessor.from_pretrained(model_name)
                        model = XCLIPModel.from_pretrained(model_name).to(
                            device
                        )
                        native_frames = int(
                            model.config.vision_config.num_frames
                        )
                        if num_frames != native_frames:
                            raise ValueError(
                                f"{model_name} expects {native_frames} "
                                f"frames, but --frames is {num_frames}."
                            )
                        model.requires_grad_(False)
                        model.eval()
                    feature = extract_xclip_feature(
                        sample,
                        processor=processor,
                        model=model,
                        device=device,
                        num_frames=num_frames,
                    )
                    payload = {
                        "feature": feature,
                        "cache_metadata": build_xclip_cache_metadata(
                            sample,
                            model_name=model_name,
                            num_frames=num_frames,
                        ),
                    }
                    torch.save(payload, cache_path)
                    refreshed_cache_files += 1

                cached[split_name].append(
                    CachedXCLIPFeature(
                        feature_path=cache_path,
                        video_path=sample.video_path,
                        class_id=sample.class_id,
                        class_index=sample.class_index,
                        split=sample.split,
                    )
                )
                progress.update(1)
    finally:
        progress.close()

    write_xclip_cache_manifest(
        cache_dir,
        cached,
        model_name=model_name,
        num_frames=num_frames,
    )
    if refreshed_cache_files:
        print(f"Refreshed {refreshed_cache_files} XCLIP feature files.")
    return cached
