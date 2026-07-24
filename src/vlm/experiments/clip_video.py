"""Run per-frame CLIP image/text similarity on a sampled video clip."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

import av
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .video_sampling import read_video_pyav, sample_frame_indices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare sampled video frames with text using CLIP."
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
        default="openai/clip-vit-base-patch32",
    )
    parser.add_argument(
        "--texts",
        nargs="+",
        default=["yawning", "sleeping", "writing"],
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
    print(f"Sampled frames shape: {video.shape}")

    model = CLIPModel.from_pretrained(args.model_name)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    pil_frames = [
        Image.fromarray(frame.astype("uint8"))
        for frame in video
    ]
    for frame_index, frame in enumerate(pil_frames):
        started_at = time.perf_counter()
        inputs = processor(
            text=args.texts,
            images=frame,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            outputs = model(**inputs)
        elapsed = time.perf_counter() - started_at
        probabilities = outputs.logits_per_image.softmax(dim=1)
        print(
            f"frame {frame_index} CLIP processor + model time: "
            f"{elapsed:.4f}s"
        )
        print(probabilities.squeeze(0))


if __name__ == "__main__":
    main()
