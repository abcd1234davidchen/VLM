"""Discover supported videos and derive labels from their filenames."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


VIDEO_NAME_PATTERN = re.compile(
    r".*A(?P<class_id>\d{3})_rgb\.(?:avi|mp4)$",
    re.IGNORECASE,
)
PARTICIPANT_PATTERN = re.compile(r"P(?P<participant_id>\d{3})", re.IGNORECASE)
VIDEO_FILE_PATTERNS = ("*.avi", "*.mp4")


def parse_class_id(video_path: Path) -> int:
    match = VIDEO_NAME_PATTERN.fullmatch(video_path.name)
    if match is None:
        raise ValueError(f"Unsupported video filename: {video_path}")
    return int(match.group("class_id"))


def parse_participant_id(video_path: Path) -> int:
    """Return the NTU participant ID encoded as ``P###`` in a filename."""
    match = PARTICIPANT_PATTERN.search(video_path.name)
    if match is None:
        raise ValueError(
            f"Video filename has no NTU participant ID (P###): {video_path}"
        )
    return int(match.group("participant_id"))


def iter_video_paths(videos_dir: Path) -> list[Path]:
    """Return supported video files, sorted and de-duplicated."""
    if not videos_dir.exists():
        raise FileNotFoundError(f"Video directory does not exist: {videos_dir}")
    return sorted(
        {
            video_path
            for pattern in VIDEO_FILE_PATTERNS
            for video_path in videos_dir.rglob(pattern)
        }
    )


def scan_all_classes(videos_dir: Path) -> list[int]:
    class_ids = {
        parse_class_id(video_path)
        for video_path in iter_video_paths(videos_dir)
        if VIDEO_NAME_PATTERN.fullmatch(video_path.name)
    }
    return sorted(class_ids)


def discover_videos(
    videos_dir: Path,
    class_ids: Iterable[int],
) -> dict[int, list[Path]]:
    requested_ids = list(class_ids)
    requested_set = set(requested_ids)
    discovered = {class_id: [] for class_id in requested_ids}
    for video_path in iter_video_paths(videos_dir):
        class_id = parse_class_id(video_path)
        if class_id in requested_set:
            discovered[class_id].append(video_path)
    return discovered
