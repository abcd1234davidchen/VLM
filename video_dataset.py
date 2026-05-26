"""Video discovery, train/test splitting, frame sampling, and embedding sequence shaping.

Filenames should end with A###_rgb.mp4 (### = three-digit class) so labels need no sidecar file.
Split manifests record train/test membership plus preprocessing knobs (stride, max sequence length).
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import torch
from PIL import Image


# e.g. foo_A001_rgb.mp4 -> class_id 1
VIDEO_NAME_PATTERN = re.compile(r".*A(?P<class_id>\d{3})_rgb\.mp4$")


@dataclass(frozen=True)
class VideoSample:
    video_path: Path
    class_id: int  # Parsed from filename (may be non-contiguous, e.g. 1, 3, 7).
    class_index: int  # 0..num_classes-1 index into the training label space for this run.
    split: str

    def to_manifest_record(self) -> dict[str, object]:
        record = asdict(self)
        record["video_path"] = str(self.video_path)
        return record


def parse_class_id(video_path: Path) -> int:
    match = VIDEO_NAME_PATTERN.fullmatch(video_path.name)
    if match is None:
        raise ValueError(f"Unsupported video filename: {video_path.name}")
    return int(match.group("class_id"))


def scan_all_classes(videos_dir: Path) -> list[int]:
    """Scan the directory for all unique class IDs based on filenames."""
    class_ids = set()
    for video_path in videos_dir.rglob("*.mp4"):
        match = VIDEO_NAME_PATTERN.fullmatch(video_path.name)
        if match is not None:
            class_ids.add(int(match.group("class_id")))
    return sorted(list(class_ids))


def discover_videos(videos_dir: Path, class_ids: Iterable[int]) -> dict[int, list[Path]]:
    """Collect .mp4 paths per requested class id; each class list is sorted by path."""
    requested_ids = list(class_ids)
    requested_set = set(requested_ids)
    discovered: dict[int, list[Path]] = {class_id: [] for class_id in requested_ids}

    for video_path in sorted(videos_dir.rglob("*.mp4")):
        class_id = parse_class_id(video_path)
        if class_id in requested_set:
            discovered[class_id].append(video_path)

    return discovered


def build_split(
    videos_dir: Path,
    class_ids: Iterable[int],
    videos_per_split: int,
    seed: int,
) -> dict[str, list[VideoSample]]:
    """Per class: shuffle, take 6 * videos_per_split clips, split 5/6 train/test, and build VideoSample objects."""
    class_ids = list(class_ids)
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
        # Deterministic ordering within each split after random selection (stable manifests / debugging).
        train_paths = sorted(selected[:5*videos_per_split], key=lambda path: path.name)
        test_paths = sorted(selected[5*videos_per_split:], key=lambda path: path.name)

        split["train"].extend(
            VideoSample(
                video_path=video_path,
                class_id=class_id,
                class_index=class_index,
                split="train",
            )
            for video_path in train_paths
        )
        split["test"].extend(
            VideoSample(
                video_path=video_path,
                class_id=class_id,
                class_index=class_index,
                split="test",
            )
            for video_path in test_paths
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
    """Write sample lists plus metadata needed to reproduce preprocessing."""
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


def load_split_manifest(manifest_path: Path) -> dict[str, list[VideoSample]]:
    """Rebuild VideoSample lists from JSON written by save_split_manifest."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata_keys = {"seed", "frame_stride", "max_seq_len"}
    split: dict[str, list[VideoSample]] = {}
    for split_name, records in payload.items():
        if split_name in metadata_keys:
            continue
        split[split_name] = []
        for item in records:
            split[split_name].append(
                VideoSample(
                    video_path=Path(item["video_path"]),
                    class_id=int(item["class_id"]),
                    class_index=int(item["class_index"]),
                    split=str(item["split"]),
                )
            )
    return split


def sample_video_frames(video_path: Path, frame_stride: int) -> list[Image.Image]:
    """Decode video and keep every frame_stride-th frame (indices 0, stride, 2*stride, ...)."""
    capture = cv2.VideoCapture(str(video_path))
    frames: list[Image.Image] = []
    frame_index = 0

    try:
        while True:
            has_frame, frame = capture.read()
            if not has_frame:
                break
            if frame_index % frame_stride == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb_frame))
            frame_index += 1
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"No frames sampled from {video_path}")

    return frames


def resize_embedding_sequence(sequence: torch.Tensor, max_seq_len: int) -> tuple[torch.Tensor, int]:
    """Produce max_seq_len rows: subsample evenly if longer, zero-pad if shorter.

    Returns (tensor, valid_length): valid_length is the real (pre-pad) length capped at
    max_seq_len, for GRU masking.
    """
    if sequence.ndim != 2:
        raise ValueError(f"Expected 2D embedding sequence, got shape {tuple(sequence.shape)}")

    seq_len, feature_dim = sequence.shape
    if seq_len == 0:
        raise ValueError("Cannot resize an empty embedding sequence")

    if seq_len > max_seq_len:
        # Evenly spaced indices along the time axis (approximate uniform subsampling).
        indices = torch.linspace(0, seq_len - 1, steps=max_seq_len).round().long()
        return sequence.index_select(0, indices), max_seq_len

    if seq_len == max_seq_len:
        return sequence, seq_len

    padded = torch.zeros((max_seq_len, feature_dim), dtype=sequence.dtype)
    padded[:seq_len] = sequence
    return padded, seq_len
