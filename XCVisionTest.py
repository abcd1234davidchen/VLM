"""Compatibility wrapper for the XCLIP vision experiment."""

from _compat import ensure_src_path

ensure_src_path()

from vlm.experiments.video_sampling import (
    read_video_pyav,
    sample_frame_indices,
)
from vlm.experiments.xclip_vision import build_parser, main

FRAME_NUM = 32

__all__ = [
    "FRAME_NUM",
    "build_parser",
    "main",
    "read_video_pyav",
    "sample_frame_indices",
]


if __name__ == "__main__":
    main()
