"""Typed configuration shared by training pipelines."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    videos_dir: Path
    cache_dir: Path
    artifacts_dir: Path
    model_name: str
    frame_stride: int
    max_seq_len: int
    clip_batch_size: int
    batch_size: int
    epochs: int
    hidden_size: int
    learning_rate: float
    dropout: float
    weight_decay: float
    label_smoothing: float
    max_grad_norm: float
    seed: int
    device: str
    extract_only: bool
    force_extract: bool

    @classmethod
    def from_namespace(
        cls,
        args: argparse.Namespace,
    ) -> "TrainingConfig":
        return cls(
            videos_dir=Path(args.videos_dir),
            cache_dir=Path(args.cache_dir),
            artifacts_dir=Path(args.artifacts_dir),
            model_name=args.model_name,
            frame_stride=args.frame_stride,
            max_seq_len=args.max_seq_len,
            clip_batch_size=args.clip_batch_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            hidden_size=args.hidden_size,
            learning_rate=args.learning_rate,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            max_grad_norm=args.max_grad_norm,
            seed=args.seed,
            device=args.device,
            extract_only=args.extract_only,
            force_extract=args.force_extract,
        )
