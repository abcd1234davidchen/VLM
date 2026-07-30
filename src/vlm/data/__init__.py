"""Video discovery, splitting, decoding, and sample models."""

from .discovery import (
    VIDEO_FILE_PATTERNS,
    VIDEO_NAME_PATTERN,
    PARTICIPANT_PATTERN,
    discover_videos,
    iter_video_paths,
    parse_class_id,
    parse_participant_id,
    scan_all_classes,
)
from .splitting import (
    VideoSample,
    assign_participants_to_splits,
    build_split,
    load_split_manifest,
    load_split_manifest_metadata,
    save_split_manifest,
    select_split_classes,
    split_counts_for_total,
    summarize_split_samples,
    validate_loaded_split,
    validate_split_fractions,
)
from .video_io import resize_embedding_sequence, sample_video_frames

__all__ = [
    "VIDEO_FILE_PATTERNS",
    "VIDEO_NAME_PATTERN",
    "PARTICIPANT_PATTERN",
    "VideoSample",
    "assign_participants_to_splits",
    "build_split",
    "discover_videos",
    "iter_video_paths",
    "load_split_manifest",
    "load_split_manifest_metadata",
    "parse_class_id",
    "parse_participant_id",
    "resize_embedding_sequence",
    "sample_video_frames",
    "save_split_manifest",
    "select_split_classes",
    "scan_all_classes",
    "split_counts_for_total",
    "summarize_split_samples",
    "validate_loaded_split",
    "validate_split_fractions",
]
