"""Run XCLIP video/text similarity on one sampled clip."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

import av
import numpy as np
import torch
from transformers import XCLIPModel, XCLIPProcessor

from .video_sampling import read_video_pyav, sample_frame_indices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare a sampled video with text using XCLIP."
    )
    parser.add_argument("--video", default="dataset/crop.mov")
    parser.add_argument(
        "--model-name",
        default="microsoft/xclip-base-patch32",
    )
    parser.add_argument(
        "--texts",
        nargs="+",
        default=["yawning", "fishing", "walking"],
    )
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    np.random.seed(args.seed)
    container = av.open(args.video)
    total_frames = container.streams.video[0].frames
    indices = sample_frame_indices(args.frames, 1, total_frames)
    video = read_video_pyav(container, indices)
    print(video.shape)

    processor = XCLIPProcessor.from_pretrained(args.model_name)
    model = XCLIPModel.from_pretrained(args.model_name)
    started_at = time.perf_counter()
    inputs = processor(
        text=args.texts,
        images=list(video),
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)
    elapsed = time.perf_counter() - started_at
    print(f"Processor + model time: {elapsed:.4f}s")
    print(outputs.logits_per_video.softmax(dim=1))


if __name__ == "__main__":
    main()
