"""Shared command-line arguments for training pipelines."""

from __future__ import annotations

import argparse


def add_common_training_arguments(
    parser: argparse.ArgumentParser,
    *,
    artifacts_dir: str,
) -> None:
    parser.add_argument("--videos-dir", default="dataset/subset264")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--artifacts-dir", default=artifacts_dir)
    parser.add_argument(
        "--model-name",
        default="openai/clip-vit-base-patch32",
    )
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--max-seq-len", type=int, default=32)
    parser.add_argument("--clip-batch-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--force-extract", action="store_true")


def add_multiclass_arguments(
    parser: argparse.ArgumentParser,
    *,
    artifacts_dir: str,
) -> None:
    add_common_training_arguments(
        parser,
        artifacts_dir=artifacts_dir,
    )
    parser.add_argument(
        "--class-ids",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Class IDs to train. If omitted, discover all classes in "
            "the video directory."
        ),
    )
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument(
        "--max-videos-per-class",
        type=int,
        default=None,
        help=(
            "Optional total video cap per class across train/validation/test. "
            "By default all discovered videos are used."
        ),
    )
    limits.add_argument(
        "--videos-per-split",
        type=int,
        default=None,
        help=(
            "Deprecated compatibility option: use a total per-class budget "
            "of 6x this value. Prefer --max-videos-per-class."
        ),
    )
    parser.add_argument("--test-fraction", type=float, default=1 / 6)
    parser.add_argument("--val-fraction", type=float, default=1 / 6)
