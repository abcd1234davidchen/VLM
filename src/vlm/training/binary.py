"""Binary safe/danger video-classification pipeline."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from vlm.config import TrainingConfig
from vlm.data import (
    VideoSample,
    iter_video_paths,
    parse_class_id,
    save_split_manifest,
)
from vlm.embeddings import (
    CachedEmbeddingDataset,
    extract_or_load_embeddings,
)
from vlm.evaluation.binary import (
    BINARY_CLASS_NAMES,
    binary_class_name,
    compute_metrics,
    evaluate_binary,
    safe_divide,
    validation_selection_score,
)
from vlm.models import GRUVideoClassifier
from vlm.reporting import (
    save_accuracy_plot,
    save_checkpoint,
    write_json,
)

from .arguments import add_common_training_arguments
from .engine import resolve_device, set_seed, train_epoch


DANGER_CLASS_IDS = frozenset({42, 43, 48})
DEFAULT_FRAME_AUGMENTATIONS = (
    "horizontal_flip",
    "small_rotation",
    "color_jitter",
    "random_crop",
)


def binary_class_index(class_id: int) -> int:
    return 1 if class_id in DANGER_CLASS_IDS else 0


def discover_binary_videos(
    videos_dir: Path,
) -> dict[int, list[Path]]:
    grouped: dict[int, list[Path]] = {0: [], 1: []}
    for video_path in iter_video_paths(videos_dir):
        class_id = parse_class_id(video_path)
        grouped[binary_class_index(class_id)].append(video_path)
    return grouped


def build_binary_split(
    videos_dir: Path,
    *,
    test_fraction: float,
    val_fraction: float,
    seed: int,
    max_videos_per_category: int | None,
) -> dict[str, list[VideoSample]]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("--test-fraction must be between 0 and 1.")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1.")
    if test_fraction + val_fraction >= 1.0:
        raise ValueError(
            "--test-fraction plus --val-fraction must be less than 1."
        )

    grouped = discover_binary_videos(videos_dir)
    if not grouped[0] or not grouped[1]:
        counts = {
            binary_class_name(index): len(paths)
            for index, paths in grouped.items()
        }
        raise ValueError(
            f"Need at least one safe and one danger video; found {counts}."
        )

    rng = random.Random(seed)
    split: dict[str, list[VideoSample]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for class_index, paths in grouped.items():
        candidates = list(paths)
        rng.shuffle(candidates)
        if max_videos_per_category is not None:
            candidates = candidates[:max_videos_per_category]
        if len(candidates) < 3:
            raise ValueError(
                f"{binary_class_name(class_index)} has only "
                f"{len(candidates)} videos after selection; need at least "
                "3 for a train/val/test split."
            )

        test_count = max(1, round(len(candidates) * test_fraction))
        test_count = min(test_count, len(candidates) - 2)
        val_count = max(1, round(len(candidates) * val_fraction))
        val_count = min(
            val_count,
            len(candidates) - test_count - 1,
        )
        test_paths = sorted(
            candidates[:test_count],
            key=lambda path: path.name,
        )
        val_paths = sorted(
            candidates[test_count : test_count + val_count],
            key=lambda path: path.name,
        )
        train_paths = sorted(
            candidates[test_count + val_count :],
            key=lambda path: path.name,
        )
        for split_name, split_paths in (
            ("train", train_paths),
            ("val", val_paths),
            ("test", test_paths),
        ):
            split[split_name].extend(
                VideoSample(
                    video_path=video_path,
                    class_id=parse_class_id(video_path),
                    class_index=class_index,
                    split=split_name,
                )
                for video_path in split_paths
            )

    for samples in split.values():
        samples.sort(
            key=lambda sample: (
                sample.class_index,
                sample.video_path.name,
            )
        )
    return split


def summarize_samples(
    samples: Iterable[VideoSample],
) -> dict[str, object]:
    samples = list(samples)
    binary_counts = Counter(sample.class_index for sample in samples)
    original_counts = Counter(sample.class_id for sample in samples)
    return {
        "binary_counts": {
            binary_class_name(index): binary_counts.get(index, 0)
            for index in range(len(BINARY_CLASS_NAMES))
        },
        "original_class_counts": {
            f"{class_id:03d}": count
            for class_id, count in sorted(original_counts.items())
        },
    }


def compute_class_weights(
    samples: Iterable[VideoSample],
    device: torch.device,
) -> torch.Tensor:
    counts = Counter(sample.class_index for sample in samples)
    total = sum(counts.values())
    weights = [
        total
        / (
            len(BINARY_CLASS_NAMES)
            * max(counts.get(index, 0), 1)
        )
        for index in range(len(BINARY_CLASS_NAMES))
    ]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_train_augmentation_plan(
    samples: list[VideoSample],
    augmentation_names: list[str],
) -> dict[str, dict[int, list[str]]]:
    counts = Counter(sample.class_index for sample in samples)
    if not counts or not augmentation_names:
        return {}
    minority_class = min(
        counts,
        key=lambda class_index: counts[class_index],
    )
    return {"train": {minority_class: augmentation_names}}


def summarize_augmentation_plan(
    plan: dict[str, dict[int, list[str]]],
) -> dict[str, object]:
    train_plan = plan.get("train", {})
    if not train_plan:
        return {
            "enabled": False,
            "train_augmented_classes": {},
        }
    return {
        "enabled": True,
        "train_augmented_classes": {
            binary_class_name(class_index): {
                "augmentation_types": augmentation_names,
                "variants_per_original_video": len(augmentation_names),
            }
            for class_index, augmentation_names in sorted(
                train_plan.items()
            )
        },
    }


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    config = TrainingConfig.from_namespace(args)
    set_seed(config.seed)
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    split = build_binary_split(
        config.videos_dir,
        test_fraction=args.test_fraction,
        val_fraction=args.val_fraction,
        seed=config.seed,
        max_videos_per_category=args.max_videos_per_category,
    )
    split_summary = {
        name: summarize_samples(samples)
        for name, samples in split.items()
    }
    augmentation_names = (
        list(args.augmentation_types)
        if args.use_augmentation
        else []
    )
    augmentation_plan = build_train_augmentation_plan(
        split["train"],
        augmentation_names,
    )
    augmentation_summary = summarize_augmentation_plan(
        augmentation_plan
    )
    save_split_manifest(
        split,
        config.artifacts_dir / "selected_split.json",
        seed=config.seed,
        frame_stride=config.frame_stride,
        max_seq_len=config.max_seq_len,
    )
    print(
        json.dumps(
            {
                "split_summary": split_summary,
                "augmentation_summary": augmentation_summary,
            },
            indent=2,
        )
    )

    cached = extract_or_load_embeddings(
        split,
        cache_dir=config.cache_dir,
        model_name=config.model_name,
        frame_stride=config.frame_stride,
        max_seq_len=config.max_seq_len,
        device=device,
        batch_size=config.clip_batch_size,
        force_extract=config.force_extract,
        augmentations_by_split_class=augmentation_plan,
        augmentation_seed=config.seed,
    )
    effective_split_summary = {
        name: summarize_samples(samples)
        for name, samples in cached.items()
    }
    if config.extract_only:
        summary = {
            "device": str(device),
            "train_samples": len(cached["train"]),
            "val_samples": len(cached["val"]),
            "test_samples": len(cached["test"]),
            "frame_stride": config.frame_stride,
            "max_seq_len": config.max_seq_len,
            "cache_dir": str(config.cache_dir),
            "split_summary": split_summary,
            "effective_split_summary": effective_split_summary,
            "augmentation_summary": augmentation_summary,
        }
        print(json.dumps(summary, indent=2))
        return summary

    train_dataset = CachedEmbeddingDataset(cached["train"])
    val_dataset = CachedEmbeddingDataset(cached["val"])
    test_dataset = CachedEmbeddingDataset(cached["test"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    input_dim = train_dataset[0][0].shape[-1]
    model = GRUVideoClassifier(
        input_dim=input_dim,
        hidden_size=config.hidden_size,
        num_classes=len(BINARY_CLASS_NAMES),
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
    )
    class_weights = (
        compute_class_weights(cached["train"], device)
        if args.use_class_weights
        else None
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=config.label_smoothing,
    )
    if class_weights is not None:
        print(
            "Using class weights: "
            + ", ".join(
                f"{name}={weight:.4f}"
                for name, weight in zip(
                    BINARY_CLASS_NAMES,
                    class_weights.cpu().tolist(),
                )
            )
        )

    history: list[dict[str, float]] = []
    best_epoch = 0
    best_score: tuple[float, ...] | None = None
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_validation_metrics: dict[str, object] | None = None
    for epoch in range(1, config.epochs + 1):
        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            config.max_grad_norm,
        )
        train_metrics, _ = evaluate_binary(
            model,
            train_eval_loader,
            cached["train"],
            device,
        )
        val_metrics, _ = evaluate_binary(
            model,
            val_loader,
            cached["val"],
            device,
        )
        scheduler.step()

        score = validation_selection_score(val_metrics)
        if best_score is None or score > best_score:
            best_epoch = epoch
            best_score = score
            best_state_dict = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
            best_validation_metrics = val_metrics

        epoch_record = {
            "epoch": float(epoch),
            "train_loss": loss,
            "train_accuracy": float(train_metrics["accuracy"]),
            "train_balanced_accuracy": float(
                train_metrics["balanced_accuracy"]
            ),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_balanced_accuracy": float(
                val_metrics["balanced_accuracy"]
            ),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_danger_precision": float(
                val_metrics["per_class"]["danger"]["precision"]
            ),
            "val_danger_recall": float(
                val_metrics["per_class"]["danger"]["recall"]
            ),
        }
        history.append(epoch_record)
        print(
            f"epoch={epoch:02d} "
            f"train_loss={loss:.4f} "
            f"train_accuracy={epoch_record['train_accuracy']:.4f} "
            f"train_balanced_accuracy="
            f"{epoch_record['train_balanced_accuracy']:.4f} "
            f"val_accuracy={epoch_record['val_accuracy']:.4f} "
            f"val_balanced_accuracy="
            f"{epoch_record['val_balanced_accuracy']:.4f} "
            f"val_macro_f1={epoch_record['val_macro_f1']:.4f} "
            f"val_danger_recall="
            f"{epoch_record['val_danger_recall']:.4f}"
        )

    if best_state_dict is None or best_validation_metrics is None:
        raise RuntimeError(
            "Training did not produce a validation checkpoint."
        )
    model.load_state_dict(best_state_dict)
    validation_metrics, _ = evaluate_binary(
        model,
        val_loader,
        cached["val"],
        device,
    )
    final_metrics, predictions = evaluate_binary(
        model,
        test_loader,
        cached["test"],
        device,
    )

    checkpoint_path = (
        config.artifacts_dir / "binary_video_classifier.pt"
    )
    results_path = (
        config.artifacts_dir / "binary_evaluation_results.json"
    )
    graph_path = config.artifacts_dir / "accuracy_graph.png"
    save_checkpoint(
        checkpoint_path,
        {
            "model_state_dict": model.state_dict(),
            "class_names": list(BINARY_CLASS_NAMES),
            "danger_class_ids": sorted(DANGER_CLASS_IDS),
            "config": vars(args),
            "input_dim": input_dim,
            "best_epoch": best_epoch,
            "danger_threshold": 0.5,
            "validation_metrics": validation_metrics,
        },
    )
    results = {
        "device": str(device),
        "class_names": list(BINARY_CLASS_NAMES),
        "danger_class_ids": sorted(DANGER_CLASS_IDS),
        "split_summary": split_summary,
        "effective_split_summary": effective_split_summary,
        "augmentation_summary": augmentation_summary,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_metrics": best_validation_metrics,
        "danger_threshold": 0.5,
        "validation_metrics": validation_metrics,
        "final_test_metrics": final_metrics,
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
        title="Training and Validation Accuracy (Binary Classifier)",
    )

    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved evaluation results to {results_path}")
    print(f"Saved accuracy graph to {graph_path}")
    print(f"Best validation epoch: {best_epoch}")
    print("Danger threshold: 0.500000")
    print("Validation metrics:")
    print(json.dumps(validation_metrics, indent=2))
    print("Final test metrics:")
    print(json.dumps(final_metrics, indent=2))
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a binary safe/danger video classifier from CLIP "
            "frame embeddings."
        )
    )
    add_common_training_arguments(
        parser,
        artifacts_dir="artifacts/binary",
    )
    parser.add_argument("--test-fraction", type=float, default=1 / 6)
    parser.add_argument("--val-fraction", type=float, default=1 / 6)
    parser.add_argument(
        "--max-videos-per-category",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--augmentation-types",
        nargs="+",
        default=list(DEFAULT_FRAME_AUGMENTATIONS),
        choices=list(DEFAULT_FRAME_AUGMENTATIONS),
        help=(
            "Frame augmentations to apply to each minority-class "
            "training video before CLIP embedding extraction."
        ),
    )
    parser.add_argument(
        "--no-augmentation",
        dest="use_augmentation",
        action="store_false",
        help="Disable train-only minority-class frame augmentation.",
    )
    parser.add_argument(
        "--no-class-weights",
        dest="use_class_weights",
        action="store_false",
    )
    parser.set_defaults(
        use_class_weights=True,
        use_augmentation=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    run_pipeline(build_parser().parse_args(argv))


__all__ = [
    "BINARY_CLASS_NAMES",
    "DANGER_CLASS_IDS",
    "DEFAULT_FRAME_AUGMENTATIONS",
    "binary_class_index",
    "binary_class_name",
    "build_binary_split",
    "build_parser",
    "build_train_augmentation_plan",
    "compute_class_weights",
    "compute_metrics",
    "discover_binary_videos",
    "evaluate_binary",
    "main",
    "run_pipeline",
    "safe_divide",
    "summarize_augmentation_plan",
    "summarize_samples",
    "validation_selection_score",
]


if __name__ == "__main__":
    main()
