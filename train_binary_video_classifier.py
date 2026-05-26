"""Binary safe/danger video classifier using the existing CLIP->BiGRU architecture.

Danger contains classes in DANGER_CLASS_IDS.
Safe contains every other class except those in DANGER_CLASS_IDS.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader

from train_video_classifier import (
    GRUVideoClassifier,
    resolve_device,
    set_seed,
    train_epoch,
)
from video_dataset import VideoSample, parse_class_id, save_split_manifest
from video_embedding_cache import CachedEmbeddingDataset, extract_or_load_embeddings


DANGER_CLASS_IDS = frozenset({42, 43, 48})
BINARY_CLASS_NAMES = ("safe", "danger")
DEFAULT_FRAME_AUGMENTATIONS = (
    "horizontal_flip",
    "small_rotation",
    "color_jitter",
    "random_crop",
)


def binary_class_index(class_id: int) -> int:
    return 1 if class_id in DANGER_CLASS_IDS else 0


def binary_class_name(class_index: int) -> str:
    return BINARY_CLASS_NAMES[class_index]


def discover_binary_videos(videos_dir: Path) -> dict[int, list[Path]]:
    """Collect all supported .avi videos, grouped by binary safe/danger label."""
    grouped: dict[int, list[Path]] = {0: [], 1: []}
    for video_path in sorted(videos_dir.rglob("*.avi")):
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
    """Stratified binary split over all discovered videos."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("--test-fraction must be between 0 and 1.")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1.")
    if test_fraction + val_fraction >= 1.0:
        raise ValueError(
            "--test-fraction plus --val-fraction must be less than 1.")

    grouped = discover_binary_videos(videos_dir)
    if not grouped[0] or not grouped[1]:
        counts = {binary_class_name(index): len(paths)
                  for index, paths in grouped.items()}
        raise ValueError(
            f"Need at least one safe and one danger video; found {counts}.")

    rng = random.Random(seed)
    split: dict[str, list[VideoSample]] = {"train": [], "val": [], "test": []}

    for class_index, paths in grouped.items():
        candidates = list(paths)
        rng.shuffle(candidates)
        if max_videos_per_category is not None:
            candidates = candidates[:max_videos_per_category]
        if len(candidates) < 3:
            raise ValueError(
                f"{binary_class_name(class_index)} has only {len(candidates)} videos after "
                "selection; need at least 3 for a train/val/test split."
            )

        test_count = max(1, round(len(candidates) * test_fraction))
        test_count = min(test_count, len(candidates) - 2)
        val_count = max(1, round(len(candidates) * val_fraction))
        val_count = min(val_count, len(candidates) - test_count - 1)
        test_paths = sorted(
            candidates[:test_count], key=lambda path: path.name)
        val_paths = sorted(
            candidates[test_count: test_count + val_count], key=lambda path: path.name)
        train_paths = sorted(
            candidates[test_count + val_count:], key=lambda path: path.name)

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

    split["train"].sort(key=lambda sample: (
        sample.class_index, sample.video_path.name))
    split["val"].sort(key=lambda sample: (
        sample.class_index, sample.video_path.name))
    split["test"].sort(key=lambda sample: (
        sample.class_index, sample.video_path.name))
    return split


def summarize_samples(samples: Iterable[VideoSample]) -> dict[str, object]:
    binary_counts = Counter(sample.class_index for sample in samples)
    original_counts = Counter(sample.class_id for sample in samples)
    return {
        "binary_counts": {
            binary_class_name(index): binary_counts.get(index, 0)
            for index in range(len(BINARY_CLASS_NAMES))
        },
        "original_class_counts": {
            f"{class_id:03d}": count for class_id, count in sorted(original_counts.items())
        },
    }


def compute_class_weights(samples: Iterable[VideoSample], device: torch.device) -> torch.Tensor:
    counts = Counter(sample.class_index for sample in samples)
    total = sum(counts.values())
    weights = [
        total / (len(BINARY_CLASS_NAMES) * max(counts.get(index, 0), 1))
        for index in range(len(BINARY_CLASS_NAMES))
    ]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_train_augmentation_plan(
    samples: list[VideoSample],
    augmentation_names: list[str],
) -> dict[str, dict[int, list[str]]]:
    """Augment only the minority training class with one variant per augmentation name."""
    counts = Counter(sample.class_index for sample in samples)
    if not counts or not augmentation_names:
        return {}

    minority_class = min(counts, key=lambda class_index: counts[class_index])
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
            for class_index, augmentation_names in sorted(train_plan.items())
        },
    }


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
) -> dict[str, object]:
    confusion = [[0 for _ in BINARY_CLASS_NAMES] for _ in BINARY_CLASS_NAMES]
    for truth, predicted in zip(y_true, y_pred):
        confusion[truth][predicted] += 1

    total = len(y_true)
    correct = sum(confusion[index][index]
                  for index in range(len(BINARY_CLASS_NAMES)))
    per_class: dict[str, dict[str, float | int]] = {}

    for index, name in enumerate(BINARY_CLASS_NAMES):
        true_positive = confusion[index][index]
        false_positive = sum(confusion[row][index] for row in range(
            len(BINARY_CLASS_NAMES))) - true_positive
        false_negative = sum(confusion[index]) - true_positive
        support = sum(confusion[index])
        precision = safe_divide(true_positive, true_positive + false_positive)
        recall = safe_divide(true_positive, true_positive + false_negative)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    macro_f1 = sum(metrics["f1"]
                   for metrics in per_class.values()) / len(BINARY_CLASS_NAMES)
    weighted_f1 = safe_divide(
        sum(metrics["f1"] * metrics["support"]
            for metrics in per_class.values()),
        total,
    )
    balanced_accuracy = sum(
        metrics["recall"] for metrics in per_class.values()) / len(BINARY_CLASS_NAMES)

    return {
        "accuracy": safe_divide(correct, total),
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": confusion,
        "confusion_matrix_labels": list(BINARY_CLASS_NAMES),
        "per_class": per_class,
    }


def validation_selection_score(metrics: dict[str, object]) -> tuple[float, ...]:
    """Choose the cleanest validation checkpoint using the fixed 0.5 threshold."""
    return (
        float(metrics["macro_f1"]),
        float(metrics["balanced_accuracy"]),
        float(metrics["accuracy"]),
    )


def evaluate_binary(
    model: nn.Module,
    loader: DataLoader,
    samples: list,
    device: torch.device,
    danger_threshold: float = 0.5,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Binary metrics plus per-sample predictions in loader order."""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    predictions: list[dict[str, object]] = []
    original_class_correct: defaultdict[int, int] = defaultdict(int)
    original_class_total: defaultdict[int, int] = defaultdict(int)
    sample_index = 0

    with torch.inference_mode():
        for embeddings, lengths, labels in loader:
            embeddings = embeddings.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            logits = model(embeddings, lengths)
            probabilities = logits.softmax(dim=1)
            predicted = (probabilities[:, 1] >= danger_threshold).long()

            for truth, pred, probs in zip(
                labels.cpu().tolist(),
                predicted.cpu().tolist(),
                probabilities.cpu().tolist(),
            ):
                sample = samples[sample_index]
                y_true.append(truth)
                y_pred.append(pred)
                original_class_total[sample.class_id] += 1
                if truth == pred:
                    original_class_correct[sample.class_id] += 1
                predictions.append(
                    {
                        "video": sample.video_path.name,
                        "path": str(sample.video_path),
                        "original_class_id": sample.class_id,
                        "true_label": binary_class_name(truth),
                        "predicted_label": binary_class_name(pred),
                        "safe_probability": probs[0],
                        "danger_probability": probs[1],
                    }
                )
                sample_index += 1

    metrics = compute_metrics(y_true, y_pred)
    metrics["original_class_accuracy"] = {
        f"{class_id:03d}": {
            "accuracy": safe_divide(original_class_correct[class_id], total),
            "correct": original_class_correct[class_id],
            "total": total,
        }
        for class_id, total in sorted(original_class_total.items())
    }
    return metrics, predictions


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    videos_dir = Path(args.videos_dir)
    cache_dir = Path(args.cache_dir)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    split = build_binary_split(
        videos_dir,
        test_fraction=args.test_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
        max_videos_per_category=args.max_videos_per_category,
    )
    split_summary = {
        "train": summarize_samples(split["train"]),
        "val": summarize_samples(split["val"]),
        "test": summarize_samples(split["test"]),
    }
    augmentation_names = list(args.augmentation_types) if args.use_augmentation else []
    augmentation_plan = build_train_augmentation_plan(
        split["train"],
        augmentation_names,
    )
    augmentation_summary = summarize_augmentation_plan(augmentation_plan)
    save_split_manifest(
        split,
        artifacts_dir / "selected_split.json",
        seed=args.seed,
        frame_stride=args.frame_stride,
        max_seq_len=args.max_seq_len,
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
        cache_dir=cache_dir,
        model_name=args.model_name,
        frame_stride=args.frame_stride,
        max_seq_len=args.max_seq_len,
        device=device,
        batch_size=args.clip_batch_size,
        force_extract=args.force_extract,
        augmentations_by_split_class=augmentation_plan,
        augmentation_seed=args.seed,
    )
    effective_split_summary = {
        "train": summarize_samples(cached["train"]),
        "val": summarize_samples(cached["val"]),
        "test": summarize_samples(cached["test"]),
    }

    if args.extract_only:
        summary = {
            "device": str(device),
            "train_samples": len(cached["train"]),
            "val_samples": len(cached["val"]),
            "test_samples": len(cached["test"]),
            "frame_stride": args.frame_stride,
            "max_seq_len": args.max_seq_len,
            "cache_dir": str(cache_dir),
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
        train_dataset, batch_size=args.batch_size, shuffle=True)
    train_eval_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False)

    input_dim = train_dataset[0][0].shape[-1]
    model = GRUVideoClassifier(
        input_dim=input_dim,
        hidden_size=args.hidden_size,
        num_classes=len(BINARY_CLASS_NAMES),
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    class_weights = compute_class_weights(
        cached["train"], device) if args.use_class_weights else None
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )

    if class_weights is not None:
        print(
            "Using class weights: "
            + ", ".join(
                f"{name}={weight:.4f}"
                for name, weight in zip(BINARY_CLASS_NAMES, class_weights.cpu().tolist())
            )
        )

    history: list[dict[str, float]] = []
    best_epoch = 0
    best_score: tuple[float, ...] | None = None
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_validation_metrics: dict[str, object] | None = None
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            args.max_grad_norm,
        )
        train_metrics, _ = evaluate_binary(
            model, train_eval_loader, cached["train"], device)
        val_metrics, _ = evaluate_binary(
            model, val_loader, cached["val"], device)
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
            "train_balanced_accuracy": float(train_metrics["balanced_accuracy"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_danger_precision": float(val_metrics["per_class"]["danger"]["precision"]),
            "val_danger_recall": float(val_metrics["per_class"]["danger"]["recall"]),
        }
        history.append(epoch_record)
        print(
            f"epoch={epoch:02d} "
            f"train_loss={loss:.4f} "
            f"train_accuracy={epoch_record['train_accuracy']:.4f} "
            f"train_balanced_accuracy={epoch_record['train_balanced_accuracy']:.4f} "
            f"val_accuracy={epoch_record['val_accuracy']:.4f} "
            f"val_balanced_accuracy={epoch_record['val_balanced_accuracy']:.4f} "
            f"val_macro_f1={epoch_record['val_macro_f1']:.4f} "
            f"val_danger_recall={epoch_record['val_danger_recall']:.4f}"
        )

    if best_state_dict is None or best_validation_metrics is None:
        raise RuntimeError("Training did not produce a validation checkpoint.")

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
    checkpoint_path = artifacts_dir / "binary_video_classifier.pt"
    results_path = artifacts_dir / "binary_evaluation_results.json"

    torch.save(
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
        checkpoint_path,
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
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved evaluation results to {results_path}")
    print(f"Best validation epoch: {best_epoch}")
    print("Danger threshold: 0.500000")
    print("Validation metrics:")
    print(json.dumps(validation_metrics, indent=2))
    print("Final test metrics:")
    print(json.dumps(final_metrics, indent=2))
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a binary safe/danger video classifier from CLIP frame embeddings.",
    )
    parser.add_argument("--videos-dir", default="dataset/prototype")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--artifacts-dir", default="artifacts/binary")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--test-fraction", type=float, default=1 / 6)
    parser.add_argument("--val-fraction", type=float, default=1 / 6)
    parser.add_argument("--max-videos-per-category", type=int, default=None)
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
    parser.add_argument(
        "--augmentation-types",
        nargs="+",
        default=list(DEFAULT_FRAME_AUGMENTATIONS),
        choices=list(DEFAULT_FRAME_AUGMENTATIONS),
        help=(
            "Frame augmentations to apply to each minority-class training video "
            "before CLIP embedding extraction."
        ),
    )
    parser.add_argument(
        "--no-augmentation",
        dest="use_augmentation",
        action="store_false",
        help="Disable train-only minority-class frame augmentation.",
    )
    parser.add_argument("--no-class-weights",
                        dest="use_class_weights", action="store_false")
    parser.set_defaults(use_class_weights=True, use_augmentation=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
