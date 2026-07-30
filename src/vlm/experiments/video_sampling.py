"""PyAV frame sampling shared by interactive experiments."""

from __future__ import annotations

import av
import numpy as np


def read_video_pyav(
    container: av.container.InputContainer,
    indices: np.ndarray,
) -> np.ndarray:
    if indices.size == 0:
        raise ValueError("At least one frame index is required.")
    decoded_frames: dict[int, np.ndarray] = {}
    container.seek(0)
    requested_indices = {int(index) for index in indices}
    end_index = max(requested_indices)
    for index, frame in enumerate(container.decode(video=0)):
        if index > end_index:
            break
        if index in requested_indices:
            decoded_frames[index] = frame.to_ndarray(format="rgb24")
    missing = requested_indices - decoded_frames.keys()
    if missing:
        raise ValueError(f"Could not decode requested frame indices: {missing}")
    return np.stack([decoded_frames[int(index)] for index in indices])


def uniform_frame_indices(
    clip_len: int,
    seg_len: int,
) -> np.ndarray:
    """Sample a deterministic fixed-size view across the complete video."""
    if clip_len < 1:
        raise ValueError("clip_len must be at least 1.")
    if seg_len < 1:
        raise ValueError("seg_len must be at least 1.")
    return np.linspace(0, seg_len - 1, num=clip_len).round().astype(np.int64)


def sample_frame_indices(
    clip_len: int,
    frame_sample_rate: int,
    seg_len: int,
) -> np.ndarray:
    converted_len = int(clip_len * frame_sample_rate)
    if seg_len <= converted_len:
        raise ValueError(
            f"Video has {seg_len} frames; need more than {converted_len} "
            "for this sampling configuration."
        )
    end_index = np.random.randint(converted_len, seg_len)
    start_index = end_index - converted_len
    indices = np.linspace(start_index, end_index, num=clip_len)
    return np.clip(
        indices,
        start_index,
        end_index - 1,
    ).astype(np.int64)
