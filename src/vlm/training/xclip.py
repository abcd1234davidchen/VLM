"""Frozen XCLIP video-feature multiclass training pipeline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from vlm.data import (
    build_split,
    load_split_manifest,
    load_split_manifest_metadata,
    save_split_manifest,
    scan_all_classes,
    select_split_classes,
    summarize_split_samples,
)
from vlm.evaluation import evaluate_features
from vlm.models import XCLIPFeatureClassifier
from vlm.reporting import (
    save_accuracy_plot,
    save_checkpoint,
    write_json,
)
from vlm.xclip_features import (
    FRAME_SAMPLING_STRATEGY,
    CachedXCLIPFeatureDataset,
    extract_or_load_xclip_features,
)

from .engine import (
    ValidationCheckpoint,
    resolve_device,
    set_seed,
    train_feature_epoch,
)


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    videos_dir = Path(args.videos_dir)
    cache_dir = Path(args.cache_dir)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = (
        Path(args.split_manifest)
        if args.split_manifest
        else None
    )
    loaded_split = (
        load_split_manifest(source_manifest)
        if source_manifest is not None
        else None
    )
    source_metadata = (
        load_split_manifest_metadata(source_manifest)
        if source_manifest is not None
        else {}
    )
    if loaded_split is not None and (
        args.max_videos_per_class is not None
        or args.videos_per_split is not None
    ):
        raise ValueError(
            "A loaded --split-manifest already fixes sample membership; "
            "do not combine it with a video cap."
        )

    discovered_class_ids = (
        sorted(
            {
                sample.class_id
                for samples in loaded_split.values()
                for sample in samples
            }
        )
        if loaded_split is not None
        else scan_all_classes(videos_dir)
    )
    class_ids = (
        list(args.class_ids)
        if args.class_ids
        else discovered_class_ids
    )
    if not class_ids:
        raise ValueError(
            f"No supported class-labelled videos found in {videos_dir}"
        )
    if not args.class_ids:
        print(
            f"Auto-discovered {len(class_ids)} classes in "
            f"{videos_dir}"
        )
    device = resolve_device(args.device)

    if args.videos_per_split is not None:
        print(
            "Warning: --videos-per-split is deprecated; prefer "
            "--max-videos-per-class."
        )
    split = (
        select_split_classes(loaded_split, class_ids)
        if loaded_split is not None
        else build_split(
            videos_dir=videos_dir,
            class_ids=class_ids,
            videos_per_split=args.videos_per_split,
            seed=args.seed,
            max_videos_per_class=args.max_videos_per_class,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
        )
    )
    split_summary = {
        name: summarize_split_samples(samples)
        for name, samples in split.items()
    }
    print(json.dumps({"split_summary": split_summary}, indent=2))
    save_split_manifest(
        split,
        artifacts_dir / "selected_split.json",
        seed=int(source_metadata.get("seed", args.seed)),
        sample_fps=None,
        max_seq_len=args.frames,
        split_strategy=str(
            source_metadata.get(
                "split_strategy",
                "participant_disjoint",
            )
        ),
        frame_sampling_strategy=FRAME_SAMPLING_STRATEGY,
        source_manifest=source_manifest,
    )
    cached = extract_or_load_xclip_features(
        split,
        cache_dir=cache_dir,
        model_name=args.model_name,
        num_frames=args.frames,
        device=device,
        force_extract=args.force_extract,
    )

    if args.extract_only:
        summary = {
            "device": str(device),
            "train_samples": len(cached["train"]),
            "val_samples": len(cached["val"]),
            "test_samples": len(cached["test"]),
            "model_name": args.model_name,
            "frame_sampling_strategy": FRAME_SAMPLING_STRATEGY,
            "frames": args.frames,
            "cache_dir": str(cache_dir),
            "split_summary": split_summary,
        }
        print(json.dumps(summary, indent=2))
        return summary

    train_dataset = CachedXCLIPFeatureDataset(cached["train"])
    val_dataset = CachedXCLIPFeatureDataset(cached["val"])
    test_dataset = CachedXCLIPFeatureDataset(cached["test"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    feature_dim = int(train_dataset[0][0].numel())
    model = XCLIPFeatureClassifier(
        feature_dim=feature_dim,
        num_classes=len(class_ids),
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )
    criterion = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing
    )

    history: list[dict[str, float]] = []
    best_checkpoint = ValidationCheckpoint()
    for epoch in range(1, args.epochs + 1):
        loss = train_feature_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            args.max_grad_norm,
            epoch=epoch,
        )
        train_accuracy, _ = evaluate_features(
            model,
            train_eval_loader,
            cached["train"],
            device,
            class_ids,
        )
        val_accuracy, _ = evaluate_features(
            model,
            val_loader,
            cached["val"],
            device,
            class_ids,
        )
        scheduler.step()
        best_checkpoint.consider(
            model,
            epoch=epoch,
            score=val_accuracy,
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
            }
        )
        print(
            f"epoch={epoch:02d} "
            f"train_loss={loss:.4f} "
            f"train_accuracy={train_accuracy:.4f} "
            f"val_accuracy={val_accuracy:.4f}"
        )

    best_checkpoint.restore(model)
    validation_accuracy, _ = evaluate_features(
        model,
        val_loader,
        cached["val"],
        device,
        class_ids,
    )
    final_accuracy, predictions = evaluate_features(
        model,
        test_loader,
        cached["test"],
        device,
        class_ids,
    )
    checkpoint_path = artifacts_dir / "xclip_classifier.pt"
    results_path = artifacts_dir / "evaluation_results.json"
    graph_path = artifacts_dir / "accuracy_graph.png"
    save_checkpoint(
        checkpoint_path,
        {
            "model_state_dict": model.state_dict(),
            "class_ids": class_ids,
            "config": vars(args),
            "feature_dim": feature_dim,
            "best_epoch": best_checkpoint.epoch,
            "validation_accuracy": validation_accuracy,
            "encoder_frozen": True,
        },
    )
    results = {
        "device": str(device),
        "model_name": args.model_name,
        "encoder_frozen": True,
        "feature_method": "get_video_features",
        "frame_sampling_strategy": FRAME_SAMPLING_STRATEGY,
        "frames": args.frames,
        "split_summary": split_summary,
        "history": history,
        "best_epoch": best_checkpoint.epoch,
        "validation_accuracy": validation_accuracy,
        "final_test_accuracy": final_accuracy,
        "checkpoint_path": str(checkpoint_path),
        "predictions": predictions,
    }
    write_json(results_path, results)
    save_accuracy_plot(
        history,
        graph_path,
        train_key="train_accuracy",
        evaluation_key="val_accuracy",
        evaluation_label="Validation Accuracy",
        title="Frozen XCLIP Training and Validation Accuracy",
    )
    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved evaluation results to {results_path}")
    print(f"Saved accuracy graph to {graph_path}")
    print(f"Best validation epoch: {best_checkpoint.epoch}")
    print(f"Validation accuracy: {validation_accuracy:.4f}")
    print(f"Final test accuracy: {final_accuracy:.4f}")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a multiclass head over frozen XCLIP video features."
        )
    )
    parser.add_argument("--videos-dir", default="dataset/subset264")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--artifacts-dir", default="artifacts/xclip")
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument(
        "--model-name",
        default="microsoft/xclip-base-patch32",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=8,
        help="Uniform whole-video frames; must match the checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument(
        "--class-ids",
        type=int,
        nargs="+",
        default=None,
    )
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument("--max-videos-per-class", type=int, default=None)
    limits.add_argument("--videos-per-split", type=int, default=None)
    parser.add_argument("--test-fraction", type=float, default=1 / 6)
    parser.add_argument("--val-fraction", type=float, default=1 / 6)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    run_pipeline(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
