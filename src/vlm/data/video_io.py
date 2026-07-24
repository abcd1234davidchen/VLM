"""Decode sampled video frames and shape embedding sequences."""

from __future__ import annotations

from pathlib import Path

import cv2
import torch
from PIL import Image


def sample_video_frames(
    video_path: Path,
    frame_stride: int,
) -> list[Image.Image]:
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

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


def resize_embedding_sequence(
    sequence: torch.Tensor,
    max_seq_len: int,
) -> tuple[torch.Tensor, int]:
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be at least 1")
    if sequence.ndim != 2:
        raise ValueError(
            f"Expected 2D embedding sequence, got shape {tuple(sequence.shape)}"
        )

    seq_len, feature_dim = sequence.shape
    if seq_len == 0:
        raise ValueError("Cannot resize an empty embedding sequence")
    if seq_len > max_seq_len:
        indices = (
            torch.linspace(0, seq_len - 1, steps=max_seq_len)
            .round()
            .long()
        )
        return sequence.index_select(0, indices), max_seq_len
    if seq_len == max_seq_len:
        return sequence, seq_len

    padded = torch.zeros((max_seq_len, feature_dim), dtype=sequence.dtype)
    padded[:seq_len] = sequence
    return padded, seq_len
