"""Video discovery, splitting, decoding, and sample models."""

from .discovery import (
    VIDEO_FILE_PATTERNS,
    VIDEO_NAME_PATTERN,
    discover_videos,
    iter_video_paths,
    parse_class_id,
    scan_all_classes,
)
from .splitting import (
    VideoSample,
    build_split,
    load_split_manifest,
    save_split_manifest,
)
from .video_io import resize_embedding_sequence, sample_video_frames

__all__ = [
    "VIDEO_FILE_PATTERNS",
    "VIDEO_NAME_PATTERN",
    "VideoSample",
    "build_split",
    "discover_videos",
    "iter_video_paths",
    "load_split_manifest",
    "parse_class_id",
    "resize_embedding_sequence",
    "sample_video_frames",
    "save_split_manifest",
    "scan_all_classes",
]
