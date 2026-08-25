"""Pre-MIT XCLIP projected-frame feature caching and datasets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import XCLIPModel, XCLIPProcessor
from transformers import __version__ as transformers_version

from vlm.data import VideoSample
from vlm.xclip_features import (
    FRAME_SAMPLING_STRATEGY,
    read_uniform_video_frames,
)


FEATURE_STAGE = "pre_mit_projected_frames"
MANIFEST_FILENAME = "xclip_pre_mit_feature_manifest.json"


@dataclass(frozen=True)
class CachedXCLIPPreMITFeature:
    feature_path: Path
    video_path: Path
    class_id: int
    class_index: int
    split: str


@dataclass(frozen=True)
class _PendingXCLIPPreMITFeature:
    sample: VideoSample
    feature_path: Path


class CachedXCLIPPreMITDataset(Dataset):
    """Load fixed-length projected frame features from the pre-MIT cache."""

    def __init__(
        self,
        samples: list[CachedXCLIPPreMITFeature],
        *,
        preload: bool = False,
    ) -> None:
        self.samples = samples
        self._features = (
            [self._load_feature(sample) for sample in samples]
            if preload
            else None
        )
        self._labels = [
            torch.tensor(sample.class_index, dtype=torch.long)
            for sample in samples
        ]

    @staticmethod
    def _load_feature(sample: CachedXCLIPPreMITFeature) -> torch.Tensor:
        payload = torch.load(sample.feature_path, map_location="cpu")
        return payload["feature"].float()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        feature = (
            self._features[index]
            if self._features is not None
            else self._load_feature(sample)
        )
        return feature, self._labels[index]


def xclip_pre_mit_cache_config_key(
    model_name: str,
    num_frames: int,
) -> str:
    safe_model_name = "".join(
        character
        if character.isalnum() or character in "._-"
        else "_"
        for character in model_name
    )
    return (
        f"{safe_model_name}__{FRAME_SAMPLING_STRATEGY}"
        f"__frames_{num_frames}__{FEATURE_STAGE}"
    )


def build_xclip_pre_mit_cache_metadata(
    sample: VideoSample,
    *,
    model_name: str,
    num_frames: int,
    feature_dim: int,
    dtype: str,
) -> dict[str, object]:
    return {
        "model_name": model_name,
        "transformers_version": transformers_version,
        "frame_sampling_strategy": FRAME_SAMPLING_STRATEGY,
        "num_frames": num_frames,
        "feature_stage": FEATURE_STAGE,
        "tensor_shape": [num_frames, feature_dim],
        "dtype": dtype,
        "video_path": str(sample.video_path),
    }


def get_xclip_pre_mit_feature_cache_path(
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
        / "xclip_pre_mit_features"
        / xclip_pre_mit_cache_config_key(model_name, num_frames)
        / f"{stem}.pt"
    )


def is_xclip_pre_mit_cache_compatible(
    payload: dict[str, object],
    sample: VideoSample,
    *,
    model_name: str,
    num_frames: int,
    feature_dim: int,
    dtype: str,
) -> bool:
    feature = payload.get("feature")
    metadata = payload.get("cache_metadata")
    return (
        isinstance(feature, torch.Tensor)
        and tuple(feature.shape) == (num_frames, feature_dim)
        and str(feature.dtype).removeprefix("torch.") == dtype
        and metadata
        == build_xclip_pre_mit_cache_metadata(
            sample,
            model_name=model_name,
            num_frames=num_frames,
            feature_dim=feature_dim,
            dtype=dtype,
        )
    )


def extract_xclip_pre_mit_feature_batch(
    frames_by_video: Sequence[np.ndarray],
    *,
    processor: XCLIPProcessor,
    model: XCLIPModel,
    device: torch.device,
) -> list[torch.Tensor]:
    """Return projected per-frame features immediately before XCLIP MIT."""
    if not frames_by_video:
        raise ValueError("At least one video is required for extraction")
    images: list[np.ndarray] | list[list[np.ndarray]]
    if len(frames_by_video) == 1:
        images = list(frames_by_video[0])
    else:
        images = [list(frames) for frames in frames_by_video]
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    batch_size, num_frames = pixel_values.shape[:2]
    flattened = pixel_values.reshape(-1, *pixel_values.shape[2:])
    with torch.inference_mode():
        vision_outputs = model.vision_model(
            pixel_values=flattened,
            return_dict=True,
        )
        projected = model.visual_projection(vision_outputs.pooler_output)
        projected = projected.reshape(batch_size, num_frames, -1)
    return [feature.detach().cpu() for feature in projected]


def write_xclip_pre_mit_cache_manifest(
    cache_dir: Path,
    cached: dict[str, list[CachedXCLIPPreMITFeature]],
    *,
    model_name: str,
    num_frames: int,
    feature_dim: int,
    dtype: str,
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
            "transformers_version": transformers_version,
            "frame_sampling_strategy": FRAME_SAMPLING_STRATEGY,
            "num_frames": num_frames,
            "feature_stage": FEATURE_STAGE,
            "tensor_shape": [num_frames, feature_dim],
            "dtype": dtype,
            "config_key": xclip_pre_mit_cache_config_key(
                model_name,
                num_frames,
            ),
        },
        "features": sorted(
            records,
            key=lambda record: str(record["video_path"]),
        ),
    }
    manifest_path = cache_dir / MANIFEST_FILENAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def extract_or_load_xclip_pre_mit_features(
    split: dict[str, list[VideoSample]],
    *,
    cache_dir: Path,
    model_name: str,
    num_frames: int,
    device: torch.device,
    force_extract: bool,
    feature_batch_size: int = 1,
    decode_workers: int = 1,
    cache_dtype: torch.dtype = torch.float16,
    read_only: bool = False,
) -> dict[str, list[CachedXCLIPPreMITFeature]]:
    """Load compatible pre-MIT features and extract only missing entries."""
    if feature_batch_size < 1:
        raise ValueError("feature_batch_size must be at least 1")
    if decode_workers < 1:
        raise ValueError("decode_workers must be at least 1")
    dtype = str(cache_dtype).removeprefix("torch.")
    cache_dir.mkdir(parents=True, exist_ok=True)
    planned: dict[
        str,
        list[CachedXCLIPPreMITFeature | _PendingXCLIPPreMITFeature],
    ] = {split_name: [] for split_name in split}
    pending: list[_PendingXCLIPPreMITFeature] = []

    progress = tqdm(
        total=sum(len(samples) for samples in split.values()),
        desc="XCLIP pre-MIT feature cache",
        unit="video",
        leave=True,
    )
    try:
        for split_name, samples in split.items():
            for sample in samples:
                cache_path = get_xclip_pre_mit_feature_cache_path(
                    cache_dir,
                    sample,
                    model_name=model_name,
                    num_frames=num_frames,
                )
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                task = _PendingXCLIPPreMITFeature(sample, cache_path)
                if force_extract or not cache_path.exists():
                    planned[split_name].append(task)
                    pending.append(task)
                    continue
                payload = torch.load(cache_path, map_location="cpu")
                feature = payload.get("feature")
                feature_dim = (
                    int(feature.shape[-1])
                    if isinstance(feature, torch.Tensor) and feature.ndim == 2
                    else 0
                )
                if not is_xclip_pre_mit_cache_compatible(
                    payload,
                    sample,
                    model_name=model_name,
                    num_frames=num_frames,
                    feature_dim=feature_dim,
                    dtype=dtype,
                ):
                    planned[split_name].append(task)
                    pending.append(task)
                    continue
                planned[split_name].append(
                    CachedXCLIPPreMITFeature(
                        cache_path,
                        sample.video_path,
                        sample.class_id,
                        sample.class_index,
                        sample.split,
                    )
                )
                progress.update(1)

        resolved: dict[Path, CachedXCLIPPreMITFeature] = {}
        feature_dim: int | None = None
        if read_only and pending:
            raise RuntimeError(
                "The read-only XCLIP pre-MIT cache has "
                f"{len(pending)} missing or incompatible entries"
            )
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
                    tuple[_PendingXCLIPPreMITFeature, Future[np.ndarray]]
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
                    frames = [future.result() for _, future in current]
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
                    features = extract_xclip_pre_mit_feature_batch(
                        frames,
                        processor=processor,
                        model=model,
                        device=device,
                    )
                    for task, feature in zip(batch, features, strict=True):
                        feature = feature.to(dtype=cache_dtype)
                        feature_dim = int(feature.shape[-1])
                        metadata = build_xclip_pre_mit_cache_metadata(
                            task.sample,
                            model_name=model_name,
                            num_frames=num_frames,
                            feature_dim=feature_dim,
                            dtype=dtype,
                        )
                        torch.save(
                            {
                                "feature": feature,
                                "cache_metadata": metadata,
                            },
                            task.feature_path,
                        )
                        cached_sample = CachedXCLIPPreMITFeature(
                            task.feature_path,
                            task.sample.video_path,
                            task.sample.class_id,
                            task.sample.class_index,
                            task.sample.split,
                        )
                        resolved[task.feature_path] = cached_sample
                        progress.update(1)
                    current = next_futures

        cached = {
            split_name: [
                resolved[item.feature_path]
                if isinstance(item, _PendingXCLIPPreMITFeature)
                else item
                for item in items
            ]
            for split_name, items in planned.items()
        }
        first = next(
            (sample for samples in cached.values() for sample in samples),
            None,
        )
        if first is not None:
            payload = torch.load(first.feature_path, map_location="cpu")
            feature_dim = int(payload["feature"].shape[-1])
        if feature_dim is None:
            raise ValueError("Cannot create an XCLIP cache for an empty split")
        if not read_only:
            write_xclip_pre_mit_cache_manifest(
                cache_dir,
                cached,
                model_name=model_name,
                num_frames=num_frames,
                feature_dim=feature_dim,
                dtype=dtype,
            )
        return cached
    finally:
        progress.close()
