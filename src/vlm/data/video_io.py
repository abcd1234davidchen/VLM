"""Decode time-sampled video frames and cap embedding sequences."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import torch
from PIL import Image


def time_based_frame_indices(
    *,
    total_frames: int,
    source_fps: float,
    target_fps: float,
    max_frames: int,
) -> list[int]:
    """Return deterministic time-based indices capped across the full video."""
    if total_frames < 1:
        raise ValueError("total_frames must be at least 1")
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("source_fps must be a positive finite number")
    if not math.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("target_fps must be a positive finite number")
    if max_frames < 1:
        raise ValueError("max_frames must be at least 1")

    frame_interval = max(source_fps / target_fps, 1.0)
    sampled_indices: list[int] = []
    sample_number = 0
    while True:
        frame_index = round(sample_number * frame_interval)
        if frame_index >= total_frames:
            break
        if not sampled_indices or frame_index != sampled_indices[-1]:
            sampled_indices.append(frame_index)
        sample_number += 1

    if len(sampled_indices) <= max_frames:
        return sampled_indices
    if max_frames == 1:
        return [sampled_indices[len(sampled_indices) // 2]]
    return [
        sampled_indices[
            round(position * (len(sampled_indices) - 1) / (max_frames - 1))
        ]
        for position in range(max_frames)
    ]


def sample_video_frames(
    video_path: Path,
    *,
    target_fps: float,
    max_frames: int,
) -> list[Image.Image]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    selected_indices = time_based_frame_indices(
        total_frames=total_frames,
        source_fps=source_fps,
        target_fps=target_fps,
        max_frames=max_frames,
    )
    selected_set = set(selected_indices)
    last_selected_index = selected_indices[-1]
    frames: list[Image.Image] = []
    frame_index = 0
    try:
        while frame_index <= last_selected_index:
            has_frame, frame = capture.read()
            if not has_frame:
                break
            if frame_index in selected_set:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb_frame))
            frame_index += 1
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"No frames sampled from {video_path}")
    if len(frames) != len(selected_indices):
        raise ValueError(
            f"Decoded {len(frames)} of {len(selected_indices)} selected "
            f"frames from {video_path}"
        )
    return frames


def cap_embedding_sequence(
    sequence: torch.Tensor,
    max_seq_len: int,
) -> tuple[torch.Tensor, int]:
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be at least 1")
    if sequence.ndim != 2:
        raise ValueError(
            f"Expected 2D embedding sequence, got shape {tuple(sequence.shape)}"
        )

    seq_len = sequence.shape[0]
    if seq_len == 0:
        raise ValueError("Cannot resize an empty embedding sequence")
    if seq_len > max_seq_len:
        indices = (
            torch.linspace(0, seq_len - 1, steps=max_seq_len)
            .round()
            .long()
        )
        return sequence.index_select(0, indices), max_seq_len
    return sequence, seq_len
