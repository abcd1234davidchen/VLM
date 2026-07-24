"""Compatibility wrapper for the XCLIP video/text experiment."""

from _compat import ensure_src_path

ensure_src_path()

from vlm.experiments.video_sampling import (
    read_video_pyav,
    sample_frame_indices,
)
from vlm.experiments.xclip import build_parser, main

__all__ = [
    "build_parser",
    "main",
    "read_video_pyav",
    "sample_frame_indices",
]


if __name__ == "__main__":
    main()
