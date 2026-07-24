"""Configuration-aware CLIP embedding cache orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from vlm.data import VideoSample

from .augmentation import FrameAugmentation, stable_augmentation_seed
from .dataset import CachedSample
from .extractor import embed_video


def cache_config_key(
    model_name: str,
    frame_stride: int,
    max_seq_len: int,
) -> str:
    safe_model_name = "".join(
        character
        if character.isalnum() or character in "._-"
        else "_"
        for character in model_name
    )
    return f"{safe_model_name}__stride_{frame_stride}__len_{max_seq_len}"


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
        "augmentation": (
            augmentation.to_metadata()
            if augmentation is not None
            else None
        ),
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


def write_cache_manifest(
    cache_dir: Path,
    cached: dict[str, list[CachedSample]],
    *,
    model_name: str,
    frame_stride: int,
    max_seq_len: int,
) -> None:
    records: list[dict[str, object]] = []
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
            "config_key": cache_config_key(
                model_name,
                frame_stride,
                max_seq_len,
            ),
        },
        "embeddings": sorted(
            records,
            key=lambda record: str(record["video_path"]),
        ),
    }
    (cache_dir / "embedding_manifest.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _embedding_cache_task_count(
    split: dict[str, list[VideoSample]],
    augmentations_by_split_class: (
        dict[str, dict[int, list[str]]] | None
    ),
) -> int:
    total = 0
    for split_name, samples in split.items():
        for sample in samples:
            total += 1
            if augmentations_by_split_class is not None:
                total += len(
                    augmentations_by_split_class.get(split_name, {}).get(
                        sample.class_index,
                        [],
                    )
                )
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
    augmentations_by_split_class: (
        dict[str, dict[int, list[str]]] | None
    ) = None,
    augmentation_seed: int = 0,
    base_seed: int | None = None,
) -> dict[str, list[CachedSample]]:
    """Load compatible tensors and generate only missing cache entries.

    ``base_seed`` is accepted as a compatibility alias for the unfinished local
    prototype refactor. New callers should use ``augmentation_seed``.
    """
    if base_seed is not None:
        augmentation_seed = base_seed
    cache_dir.mkdir(parents=True, exist_ok=True)

    processor: CLIPProcessor | None = None
    model: CLIPModel | None = None
    cached = {split_name: [] for split_name in split}
    refreshed_cache_files = 0

    progress = tqdm(
        total=_embedding_cache_task_count(
            split,
            augmentations_by_split_class,
        ),
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
                        cache_payload = torch.load(
                            cache_path,
                            map_location="cpu",
                        )
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
                            processor = CLIPProcessor.from_pretrained(
                                model_name
                            )
                            model = CLIPModel.from_pretrained(model_name).to(
                                device
                            )
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
                            augmentation_name=(
                                augmentation.name
                                if augmentation is not None
                                else None
                            ),
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
