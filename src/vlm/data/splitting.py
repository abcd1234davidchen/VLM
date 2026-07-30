"""Dataset sample records, deterministic splits, and split manifests."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .discovery import discover_videos, parse_participant_id


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
    videos_per_split: int | None = None,
    seed: int = 42,
    *,
    max_videos_per_class: int | None = None,
    val_fraction: float = 1 / 6,
    test_fraction: float = 1 / 6,
) -> dict[str, list[VideoSample]]:
    """Build a deterministic participant-disjoint train/val/test split.

    By default every discovered video is retained. ``max_videos_per_class``
    applies an optional per-class cap. ``videos_per_split`` is retained as a
    compatibility alias for the historical ``6N`` per-class sample budget.
    """
    if videos_per_split is not None and videos_per_split < 1:
        raise ValueError("videos_per_split must be at least 1")
    if max_videos_per_class is not None and max_videos_per_class < 3:
        raise ValueError("max_videos_per_class must be at least 3")
    if (
        videos_per_split is not None
        and max_videos_per_class is not None
    ):
        raise ValueError(
            "Use either videos_per_split or max_videos_per_class, not both."
        )
    if videos_per_split is not None:
        max_videos_per_class = videos_per_split * 6
    class_ids = list(class_ids)
    if not class_ids:
        raise ValueError("At least one class ID is required")

    videos_by_class = discover_videos(videos_dir, class_ids)
    all_paths = [
        path
        for class_paths in videos_by_class.values()
        for path in class_paths
    ]
    participant_split = assign_participants_to_splits(
        (parse_participant_id(path) for path in all_paths),
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    participant_to_split = {
        participant_id: split_name
        for split_name, participant_ids in participant_split.items()
        for participant_id in participant_ids
    }
    split: dict[str, list[VideoSample]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for class_index, class_id in enumerate(class_ids):
        candidates_by_split: dict[str, list[Path]] = {
            "train": [],
            "val": [],
            "test": [],
        }
        for path in videos_by_class[class_id]:
            candidates_by_split[
                participant_to_split[parse_participant_id(path)]
            ].append(path)

        selected_counts = (
            split_counts_for_total(
                max_videos_per_class,
                val_fraction=val_fraction,
                test_fraction=test_fraction,
            )
            if max_videos_per_class is not None
            else {
                name: len(candidates)
                for name, candidates in candidates_by_split.items()
            }
        )
        empty_splits = [
            split_name
            for split_name, candidates in candidates_by_split.items()
            if not candidates
        ]
        if empty_splits:
            raise ValueError(
                f"Class {class_id:03d} has no videos in "
                f"participant-disjoint split(s): {', '.join(empty_splits)}. "
                "Provide more participants for this class or choose a "
                "different seed."
            )

        for split_name, target_count in selected_counts.items():
            candidates = candidates_by_split[split_name]
            if len(candidates) < target_count:
                raise ValueError(
                    f"Class {class_id:03d} has only {len(candidates)} "
                    f"{split_name} videos after participant-disjoint "
                    f"splitting; need {target_count}. Use a lower "
                    "--max-videos-per-class, remove the cap, or provide "
                    "more participants."
                )
            rng = random.Random(f"{seed}:{class_id}:{split_name}")
            rng.shuffle(candidates)
            paths = sorted(
                candidates[:target_count],
                key=lambda path: path.name,
            )
            split[split_name].extend(
                VideoSample(
                    video_path=video_path,
                    class_id=class_id,
                    class_index=class_index,
                    split=split_name,
                )
                for video_path in paths
            )
    for samples in split.values():
        samples.sort(
            key=lambda sample: (
                sample.class_index,
                sample.video_path.name,
            )
        )
    return split


def summarize_split_samples(
    samples: Iterable[VideoSample],
) -> dict[str, object]:
    """Summarize video, class, and participant counts for one split."""
    sample_list = list(samples)
    class_counts = Counter(sample.class_id for sample in sample_list)
    participants = {
        parse_participant_id(sample.video_path)
        for sample in sample_list
    }
    return {
        "videos": len(sample_list),
        "participants": len(participants),
        "class_counts": {
            f"{class_id:03d}": count
            for class_id, count in sorted(class_counts.items())
        },
    }


def validate_split_fractions(
    *,
    val_fraction: float,
    test_fraction: float,
) -> None:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1.")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError(
            "val_fraction plus test_fraction must be less than 1."
        )


def split_counts_for_total(
    total: int,
    *,
    val_fraction: float,
    test_fraction: float,
) -> dict[str, int]:
    """Allocate a total sample budget while keeping every split non-empty."""
    validate_split_fractions(
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    if total < 3:
        raise ValueError("At least 3 items are required for train/val/test.")
    test_count = min(max(1, round(total * test_fraction)), total - 2)
    val_count = min(
        max(1, round(total * val_fraction)),
        total - test_count - 1,
    )
    return {
        "train": total - val_count - test_count,
        "val": val_count,
        "test": test_count,
    }


def assign_participants_to_splits(
    participant_ids: Iterable[int],
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, set[int]]:
    """Assign each participant to exactly one deterministic dataset split."""
    participants = sorted(set(participant_ids))
    counts = split_counts_for_total(
        len(participants),
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    random.Random(seed).shuffle(participants)
    test_end = counts["test"]
    val_end = test_end + counts["val"]
    return {
        "train": set(participants[val_end:]),
        "val": set(participants[test_end:val_end]),
        "test": set(participants[:test_end]),
    }


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
        "split_strategy": "participant_disjoint",
        "participants": {
            split_name: sorted(
                {
                    parse_participant_id(sample.video_path)
                    for sample in samples
                }
            )
            for split_name, samples in split.items()
        },
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
    metadata_keys = {
        "seed",
        "frame_stride",
        "max_seq_len",
        "split_strategy",
        "participants",
    }
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
