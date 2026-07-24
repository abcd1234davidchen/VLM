"""PyAV frame sampling shared by interactive experiments."""

from __future__ import annotations

import av
import numpy as np


def read_video_pyav(
    container: av.container.InputContainer,
    indices: np.ndarray,
) -> np.ndarray:
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for index, frame in enumerate(container.decode(video=0)):
        if index > end_index:
            break
        if index >= start_index and index in indices:
            frames.append(frame)
    return np.stack(
        [frame.to_ndarray(format="rgb24") for frame in frames]
    )


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
