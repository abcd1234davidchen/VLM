"""Shared CLIP video embedding cache.

The cache is keyed only by inputs that change the embedding tensor:
model name, frame stride, max sequence length, video path, and optional frame
augmentation. Train/val/test membership, labels, and split fractions live
outside the cache.
"""
from __future__ import annotations

import json
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from PIL import Image, ImageEnhance, ImageOps

from video_dataset import VideoSample, resize_embedding_sequence, sample_video_frames


@dataclass
class CachedSample:
    embedding_path: Path
    video_path: Path
    class_id: int
    class_index: int
    split: str
    # Non-padding rows in the embeddings tensor; see resize_embedding_sequence.
    valid_length: int
    augmentation_name: str | None = None

    def to_record(self) -> dict[str, object]:
        return {
            "embedding_path": str(self.embedding_path),
            "video_path": str(self.video_path),
            "class_id": self.class_id,
            "class_index": self.class_index,
            "split": self.split,
            "valid_length": self.valid_length,
            "augmentation_name": self.augmentation_name,
        }


@dataclass(frozen=True)
class FrameAugmentation:
    """A deterministic per-video frame augmentation applied before CLIP embedding."""

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
            character if character.isalnum() or character in "._-" else "_"
            for character in self.name
        )
        return f"{safe_name}_{digest}"


class CachedEmbeddingDataset(Dataset):
    """Loads all .pt embedding files into memory once (small prototype datasets)."""

    def __init__(self, samples: list[CachedSample]) -> None:
        self.samples = samples
        self.items = [torch.load(
            sample.embedding_path, map_location="cpu") for sample in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.items[index]
        embeddings = item["embeddings"].float()
        valid_length = torch.tensor(item["valid_length"], dtype=torch.long)
        label = torch.tensor(self.samples[index].class_index, dtype=torch.long)
        return embeddings, valid_length, label


def cache_config_key(model_name: str, frame_stride: int, max_seq_len: int) -> str:
    safe_model_name = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in model_name
    )
    return f"{safe_model_name}__stride_{frame_stride}__len_{max_seq_len}"


def stable_augmentation_seed(
    base_seed: int,
    split_name: str,
    sample: VideoSample,
    augmentation_name: str,
) -> int:
    payload = f"{base_seed}|{split_name}|{sample.video_path}|{sample.class_index}|{augmentation_name}"
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def apply_frame_augmentation(
    frames: list[Image.Image],
    augmentation: FrameAugmentation,
) -> list[Image.Image]:
    """Apply one deterministic video-level augmentation to all sampled frames."""
    rng = random.Random(augmentation.seed)

    if augmentation.name == "horizontal_flip":
        return [ImageOps.mirror(frame) for frame in frames]

    if augmentation.name == "small_rotation":
        angle = rng.uniform(-augmentation.rotation_degrees, augmentation.rotation_degrees)
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
            frame.crop(crop_box).resize((width, height), Image.Resampling.BICUBIC)
            for frame in frames
        ]

    raise ValueError(f"Unsupported frame augmentation: {augmentation.name}")


def build_cache_metadata(
    sample: VideoSample,
    *,
    model_name: str,
    frame_stride: int,
    max_seq_len: int,
    augmentation: FrameAugmentation | None = None,
) -> dict[str, object]:
    return {
        "model_name": model_name,
        "frame_stride": frame_stride,
        "max_seq_len": max_seq_len,
        "video_path": str(sample.video_path),
        "augmentation": augmentation.to_metadata() if augmentation is not None else None,
    }


def get_embedding_cache_path(
    cache_dir: Path,
    sample: VideoSample,
    *,
    model_name: str,
    frame_stride: int,
    max_seq_len: int,
    augmentation: FrameAugmentation | None = None,
) -> Path:
    """One cache file per video per embedding config."""
    stem = sample.video_path.stem
    if augmentation is not None:
        stem = f"{stem}__aug_{augmentation.cache_key()}"
    return (
        cache_dir
        / "embeddings"
        / cache_config_key(model_name, frame_stride, max_seq_len)
        / f"{stem}.pt"
    )


def is_cache_compatible(
    payload: dict[str, object],
    sample: VideoSample,
    *,
    model_name: str,
    frame_stride: int,
    max_seq_len: int,
    augmentation: FrameAugmentation | None = None,
) -> bool:
    embeddings = payload.get("embeddings")
    valid_length = payload.get("valid_length")
    metadata = payload.get("cache_metadata")

    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        return False
    if embeddings.shape[0] != max_seq_len:
        return False
    if not isinstance(valid_length, int):
        return False
    if valid_length < 1 or valid_length > max_seq_len:
        return False
    if not isinstance(metadata, dict):
        return False

    expected_metadata = build_cache_metadata(
        sample,
        model_name=model_name,
        frame_stride=frame_stride,
        max_seq_len=max_seq_len,
        augmentation=augmentation,
    )
    if augmentation is None:
        legacy_metadata = dict(expected_metadata)
        legacy_metadata.pop("augmentation")
        if metadata == legacy_metadata:
            return True
    return metadata == expected_metadata


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
    """Run CLIP on strided RGB frames, L2-normalize per frame, then fix sequence length."""
    frames = sample_video_frames(sample.video_path, frame_stride)
    if augmentation is not None:
        frames = apply_frame_augmentation(frames, augmentation)
    features: list[torch.Tensor] = []

    for start in range(0, len(frames), batch_size):
        frame_batch = frames[start: start + batch_size]
        inputs = processor(images=frame_batch, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.inference_mode():
            batch_features = model.get_image_features(
                pixel_values=pixel_values)
        if hasattr(batch_features, "pooler_output"):
            batch_features = batch_features.pooler_output
        batch_features = torch.nn.functional.normalize(batch_features, dim=-1)
        features.append(batch_features.cpu())

    stacked = torch.cat(features, dim=0)
    return resize_embedding_sequence(stacked, max_seq_len)


def write_cache_manifest(
    cache_dir: Path,
    cached: dict[str, list[CachedSample]],
    *,
    model_name: str,
    frame_stride: int,
    max_seq_len: int,
) -> None:
    records = []
    seen_paths: set[Path] = set()
    for samples in cached.values():
        for sample in samples:
            if sample.embedding_path in seen_paths:
                continue
            seen_paths.add(sample.embedding_path)
            records.append(
                {
                    "embedding_path": str(sample.embedding_path),
                    "video_path": str(sample.video_path),
                    "valid_length": sample.valid_length,
                    "augmentation_name": sample.augmentation_name,
                }
            )

    payload = {
        "cache_properties": {
            "model_name": model_name,
            "frame_stride": frame_stride,
            "max_seq_len": max_seq_len,
            "config_key": cache_config_key(model_name, frame_stride, max_seq_len),
        },
        "embeddings": sorted(records, key=lambda record: str(record["video_path"])),
    }
    (cache_dir / "embedding_manifest.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _embedding_cache_task_count(
    split: dict[str, list[VideoSample]],
    augmentations_by_split_class: dict[str, dict[int, list[str]]] | None,
) -> int:
    total = 0
    for split_name, samples in split.items():
        for sample in samples:
            n_variants = 1
            if augmentations_by_split_class is not None:
                n_variants += len(
                    augmentations_by_split_class.get(split_name, {}).get(
                        sample.class_index, []
                    )
                )
            total += n_variants
    return total


def extract_or_load_embeddings(
    split: dict[str, list[VideoSample]],
    *,
    cache_dir: Path,
    model_name: str,
    frame_stride: int,
    max_seq_len: int,
    device: torch.device,
    batch_size: int,
    force_extract: bool,
    augmentations_by_split_class: dict[str, dict[int, list[str]]] | None = None,
    augmentation_seed: int = 0,
) -> dict[str, list[CachedSample]]:
    """Fill cache only for missing/incompatible embedding tensors."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    processor: CLIPProcessor | None = None
    model: CLIPModel | None = None
    cached: dict[str, list[CachedSample]] = {
        split_name: [] for split_name in split}
    refreshed_cache_files = 0
    total_tasks = _embedding_cache_task_count(split, augmentations_by_split_class)

    progress = tqdm(
        total=total_tasks,
        desc="Embedding cache",
        unit="item",
        leave=True,
    )
    try:
        for split_name, samples in split.items():
            for sample in samples:
                augmentation_names: list[str | None] = [None]
                if augmentations_by_split_class is not None:
                    augmentation_names.extend(
                        augmentations_by_split_class
                        .get(split_name, {})
                        .get(sample.class_index, [])
                    )

                for augmentation_name in augmentation_names:
                    augmentation = None
                    if augmentation_name is not None:
                        augmentation = FrameAugmentation(
                            name=augmentation_name,
                            seed=stable_augmentation_seed(
                                augmentation_seed,
                                split_name,
                                sample,
                                augmentation_name,
                            ),
                        )

                    cache_path = get_embedding_cache_path(
                        cache_dir,
                        sample,
                        model_name=model_name,
                        frame_stride=frame_stride,
                        max_seq_len=max_seq_len,
                        augmentation=augmentation,
                    )

                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_payload: dict[str, object] | None = None
                    should_extract = force_extract or not cache_path.exists()
                    if not should_extract:
                        cache_payload = torch.load(cache_path, map_location="cpu")
                        should_extract = not is_cache_compatible(
                            cache_payload,
                            sample,
                            model_name=model_name,
                            frame_stride=frame_stride,
                            max_seq_len=max_seq_len,
                            augmentation=augmentation,
                        )

                    if should_extract:
                        if processor is None or model is None:
                            processor = CLIPProcessor.from_pretrained(model_name)
                            model = CLIPModel.from_pretrained(model_name).to(device)
                            model.eval()
                        embeddings, valid_length = embed_video(
                            sample,
                            processor=processor,
                            model=model,
                            device=device,
                            frame_stride=frame_stride,
                            max_seq_len=max_seq_len,
                            batch_size=batch_size,
                            augmentation=augmentation,
                        )
                        cache_payload = {
                            "embeddings": embeddings,
                            "valid_length": valid_length,
                            "cache_metadata": build_cache_metadata(
                                sample,
                                model_name=model_name,
                                frame_stride=frame_stride,
                                max_seq_len=max_seq_len,
                                augmentation=augmentation,
                            ),
                        }
                        torch.save(cache_payload, cache_path)
                        refreshed_cache_files += 1
                    else:
                        assert cache_payload is not None
                        valid_length = int(cache_payload["valid_length"])

                    cached[split_name].append(
                        CachedSample(
                            embedding_path=cache_path,
                            video_path=sample.video_path,
                            class_id=sample.class_id,
                            class_index=sample.class_index,
                            split=sample.split,
                            valid_length=valid_length,
                            augmentation_name=augmentation.name if augmentation is not None else None,
                        )
                    )
                    progress.update(1)
    finally:
        progress.close()

    write_cache_manifest(
        cache_dir,
        cached,
        model_name=model_name,
        frame_stride=frame_stride,
        max_seq_len=max_seq_len,
    )
    if refreshed_cache_files:
        print(f"Refreshed {refreshed_cache_files} cached embedding files.")
    return cached
