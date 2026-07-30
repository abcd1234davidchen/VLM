"""Extract XCLIP vision features from a sampled video."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

import av
import torch
from transformers import (
    AutoProcessor,
    XCLIPVisionConfig,
    XCLIPVisionModel,
)

from .video_sampling import read_video_pyav, uniform_frame_indices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract XCLIP vision features from a sampled video."
    )
    parser.add_argument(
        "--video",
        default=(
            "dataset/subset264/S19/"
            "S019C001P008R001A103_rgb.mp4"
        ),
    )
    parser.add_argument(
        "--model-name",
        default="microsoft/xclip-base-patch32",
    )
    parser.add_argument("--frames", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    container = av.open(args.video)
    total_frames = container.streams.video[0].frames
    indices = uniform_frame_indices(args.frames, total_frames)
    video = read_video_pyav(container, indices)
    print(video.shape)

    processor = AutoProcessor.from_pretrained(args.model_name)
    configuration = XCLIPVisionConfig.from_pretrained(args.model_name)
    configuration.num_frames = args.frames
    model = XCLIPVisionModel.from_pretrained(
        args.model_name,
        config=configuration,
    )
    started_at = time.perf_counter()
    pixel_values = processor(
        images=list(video),
        return_tensors="pt",
    ).pixel_values
    (
        _,
        _,
        num_channels,
        height,
        width,
    ) = pixel_values.shape
    pixel_values = pixel_values.reshape(
        -1,
        num_channels,
        height,
        width,
    )
    with torch.no_grad():
        outputs = model(pixel_values)
    elapsed = time.perf_counter() - started_at
    print(f"Processor + model time: {elapsed:.4f}s")
    print(outputs.last_hidden_state.shape)


if __name__ == "__main__":
    main()
