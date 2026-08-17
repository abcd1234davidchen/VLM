"""Frozen XCLIP video-feature extraction, caching, and datasets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
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


@dataclass(frozen=True)
class _PendingXCLIPFeature:
    sample: VideoSample
    feature_path: Path


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
    return extract_xclip_feature_batch(
        [frames],
        processor=processor,
        model=model,
        device=device,
    )[0]


def extract_xclip_feature_batch(
    frames_by_video: Sequence[np.ndarray],
    *,
    processor: XCLIPProcessor,
    model: XCLIPModel,
    device: torch.device,
) -> list[torch.Tensor]:
    """Encode multiple fixed-size video views in one model call."""
    if not frames_by_video:
        raise ValueError("At least one video is required for extraction")
    images: list[np.ndarray] | list[list[np.ndarray]]
    if len(frames_by_video) == 1:
        images = list(frames_by_video[0])
    else:
        images = [list(frames) for frames in frames_by_video]
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.inference_mode():
        output = model.get_video_features(pixel_values=pixel_values)
    features = output.pooler_output
    expected_batch_size = len(frames_by_video)
    if features.ndim != 2 or features.shape[0] != expected_batch_size:
        raise ValueError(
            "XCLIP get_video_features returned an unexpected shape: "
            f"{tuple(features.shape)}; expected batch size "
            f"{expected_batch_size}"
        )
    return [feature.detach().cpu() for feature in features]


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
    feature_batch_size: int = 1,
    decode_workers: int = 1,
) -> dict[str, list[CachedXCLIPFeature]]:
    """Load compatible frozen features and extract only missing entries."""
    if feature_batch_size < 1:
        raise ValueError("feature_batch_size must be at least 1")
    if decode_workers < 1:
        raise ValueError("decode_workers must be at least 1")
    cache_dir.mkdir(parents=True, exist_ok=True)
    processor: XCLIPProcessor | None = None
    model: XCLIPModel | None = None
    planned: dict[
        str,
        list[CachedXCLIPFeature | _PendingXCLIPFeature],
    ] = {split_name: [] for split_name in split}
    pending: list[_PendingXCLIPFeature] = []
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
                    task = _PendingXCLIPFeature(
                        sample=sample,
                        feature_path=cache_path,
                    )
                    planned[split_name].append(task)
                    pending.append(task)
                else:
                    planned[split_name].append(
                        CachedXCLIPFeature(
                            feature_path=cache_path,
                            video_path=sample.video_path,
                            class_id=sample.class_id,
                            class_index=sample.class_index,
                            split=sample.split,
                        )
                    )
                    progress.update(1)

        resolved: dict[Path, CachedXCLIPFeature] = {}
        if pending:
            processor = XCLIPProcessor.from_pretrained(model_name)
            model = XCLIPModel.from_pretrained(model_name).to(device)
            native_frames = int(model.config.vision_config.num_frames)
            if num_frames != native_frames:
                raise ValueError(
                    f"{model_name} expects {native_frames} frames, but "
                    f"--frames is {num_frames}."
                )
            model.requires_grad_(False)
            model.eval()

            batches = [
                pending[start : start + feature_batch_size]
                for start in range(0, len(pending), feature_batch_size)
            ]
            with ThreadPoolExecutor(max_workers=decode_workers) as executor:
                current: list[
                    tuple[_PendingXCLIPFeature, Future[np.ndarray]]
                ] = [
                    (
                        task,
                        executor.submit(
                            read_uniform_video_frames,
                            task.sample.video_path,
                            num_frames,
                        ),
                    )
                    for task in batches[0]
                ]
                for batch_index, batch in enumerate(batches):
                    frames_by_video = [future.result() for _, future in current]
                    next_futures = (
                        [
                            (
                                task,
                                executor.submit(
                                    read_uniform_video_frames,
                                    task.sample.video_path,
                                    num_frames,
                                ),
                            )
                            for task in batches[batch_index + 1]
                        ]
                        if batch_index + 1 < len(batches)
                        else []
                    )
                    features = extract_xclip_feature_batch(
                        frames_by_video,
                        processor=processor,
                        model=model,
                        device=device,
                    )
                    for task, feature in zip(batch, features, strict=True):
                        payload = {
                            "feature": feature,
                            "cache_metadata": build_xclip_cache_metadata(
                                task.sample,
                                model_name=model_name,
                                num_frames=num_frames,
                            ),
                        }
                        torch.save(payload, task.feature_path)
                        resolved[task.feature_path] = CachedXCLIPFeature(
                            feature_path=task.feature_path,
                            video_path=task.sample.video_path,
                            class_id=task.sample.class_id,
                            class_index=task.sample.class_index,
                            split=task.sample.split,
                        )
                        refreshed_cache_files += 1
                        progress.update(1)
                    current = next_futures
    finally:
        progress.close()

    cached = {
        split_name: [
            (
                resolved[item.feature_path]
                if isinstance(item, _PendingXCLIPFeature)
                else item
            )
            for item in items
        ]
        for split_name, items in planned.items()
    }

    write_xclip_cache_manifest(
        cache_dir,
        cached,
        model_name=model_name,
        num_frames=num_frames,
    )
    if refreshed_cache_files:
        print(f"Refreshed {refreshed_cache_files} XCLIP feature files.")
    return cached
