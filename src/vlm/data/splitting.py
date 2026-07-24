"""Dataset sample records, deterministic splits, and split manifests."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .discovery import discover_videos


@dataclass(frozen=True)
class VideoSample:
    video_path: Path
    class_id: int
    class_index: int
    split: str

    def to_manifest_record(self) -> dict[str, object]:
        record = asdict(self)
        record["video_path"] = str(self.video_path)
        return record


def build_split(
    videos_dir: Path,
    class_ids: Iterable[int],
    videos_per_split: int,
    seed: int,
) -> dict[str, list[VideoSample]]:
    """Select 6N clips per class and use a deterministic 5:1 train/test split."""
    if videos_per_split < 1:
        raise ValueError("videos_per_split must be at least 1")
    class_ids = list(class_ids)
    if not class_ids:
        raise ValueError("At least one class ID is required")

    rng = random.Random(seed)
    videos_by_class = discover_videos(videos_dir, class_ids)
    split: dict[str, list[VideoSample]] = {"train": [], "test": []}

    for class_index, class_id in enumerate(class_ids):
        candidates = list(videos_by_class[class_id])
        required = videos_per_split * 6
        if len(candidates) < required:
            raise ValueError(
                f"Class {class_id:03d} only has {len(candidates)} videos; "
                f"need at least {required}."
            )

        rng.shuffle(candidates)
        selected = candidates[:required]
        train_paths = sorted(
            selected[: 5 * videos_per_split],
            key=lambda path: path.name,
        )
        test_paths = sorted(
            selected[5 * videos_per_split :],
            key=lambda path: path.name,
        )
        for split_name, paths in (("train", train_paths), ("test", test_paths)):
            split[split_name].extend(
                VideoSample(
                    video_path=video_path,
                    class_id=class_id,
                    class_index=class_index,
                    split=split_name,
                )
                for video_path in paths
            )
    return split


def save_split_manifest(
    split: dict[str, list[VideoSample]],
    manifest_path: Path,
    *,
    seed: int,
    frame_stride: int,
    max_seq_len: int,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "frame_stride": frame_stride,
        "max_seq_len": max_seq_len,
        **{
            split_name: [sample.to_manifest_record() for sample in samples]
            for split_name, samples in split.items()
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_split_manifest(
    manifest_path: Path,
) -> dict[str, list[VideoSample]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata_keys = {"seed", "frame_stride", "max_seq_len"}
    return {
        split_name: [
            VideoSample(
                video_path=Path(item["video_path"]),
                class_id=int(item["class_id"]),
                class_index=int(item["class_index"]),
                split=str(item["split"]),
            )
            for item in records
        ]
        for split_name, records in payload.items()
        if split_name not in metadata_keys
    }
