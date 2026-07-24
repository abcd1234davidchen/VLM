"""Multiclass CLIP-embedding GRU training pipeline."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from vlm.config import TrainingConfig
from vlm.data import build_split, save_split_manifest, scan_all_classes
from vlm.embeddings import (
    CachedEmbeddingDataset,
    extract_or_load_embeddings,
)
from vlm.evaluation import evaluate
from vlm.models import GRUVideoClassifier
from vlm.reporting import (
    save_accuracy_plot,
    save_checkpoint,
    write_json,
)

from .arguments import add_multiclass_arguments
from .engine import resolve_device, set_seed, train_epoch


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    config = TrainingConfig.from_namespace(args)
    set_seed(config.seed)
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    class_ids = (
        list(args.class_ids)
        if args.class_ids
        else scan_all_classes(config.videos_dir)
    )
    if not class_ids:
        raise ValueError(
            f"No supported class-labelled videos found in "
            f"{config.videos_dir}"
        )
    if not args.class_ids:
        print(
            f"Auto-discovered {len(class_ids)} classes in "
            f"{config.videos_dir}"
        )
    device = resolve_device(config.device)

    split = build_split(
        videos_dir=config.videos_dir,
        class_ids=class_ids,
        videos_per_split=args.videos_per_split,
        seed=config.seed,
    )
    save_split_manifest(
        split,
        config.artifacts_dir / "selected_split.json",
        seed=config.seed,
        frame_stride=config.frame_stride,
        max_seq_len=config.max_seq_len,
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
    )

    if config.extract_only:
        summary = {
            "device": str(device),
            "train_samples": len(cached["train"]),
            "test_samples": len(cached["test"]),
            "frame_stride": config.frame_stride,
            "max_seq_len": config.max_seq_len,
            "cache_dir": str(config.cache_dir),
        }
        print(json.dumps(summary, indent=2))
        return summary

    train_dataset = CachedEmbeddingDataset(cached["train"])
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
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    input_dim = train_dataset[0][0].shape[-1]
    model = GRUVideoClassifier(
        input_dim=input_dim,
        hidden_size=config.hidden_size,
        num_classes=len(class_ids),
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
    criterion = nn.CrossEntropyLoss(
        label_smoothing=config.label_smoothing
    )

    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            config.max_grad_norm,
        )
        train_accuracy, _ = evaluate(
            model,
            train_eval_loader,
            cached["train"],
            device,
            class_ids,
        )
        started_at = time.time()
        test_accuracy, _ = evaluate(
            model,
            test_loader,
            cached["test"],
            device,
            class_ids,
        )
        test_time = time.time() - started_at
        scheduler.step()
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": loss,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "test_time": test_time,
            }
        )
        print(
            f"epoch={epoch:02d} "
            f"train_loss={loss:.4f} "
            f"train_accuracy={train_accuracy:.4f} "
            f"test_accuracy={test_accuracy:.4f} "
            f"test_time={test_time:.4f}s"
        )

    final_accuracy, predictions = evaluate(
        model,
        test_loader,
        cached["test"],
        device,
        class_ids,
    )
    checkpoint_path = config.artifacts_dir / "video_classifier.pt"
    results_path = config.artifacts_dir / "evaluation_results.json"
    graph_path = config.artifacts_dir / "accuracy_graph.png"
    save_checkpoint(
        checkpoint_path,
        {
            "model_state_dict": model.state_dict(),
            "class_ids": class_ids,
            "config": vars(args),
            "input_dim": input_dim,
        },
    )
    results = {
        "device": str(device),
        "history": history,
        "final_test_accuracy": final_accuracy,
        "checkpoint_path": str(checkpoint_path),
        "predictions": predictions,
    }
    write_json(results_path, results)
    save_accuracy_plot(
        history,
        graph_path,
        train_key="train_accuracy",
        evaluation_key="test_accuracy",
        evaluation_label="Test Accuracy",
        title="Training and Testing Accuracy",
    )
    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved evaluation results to {results_path}")
    print(f"Saved accuracy graph to {graph_path}")
    print(f"Final test accuracy: {final_accuracy:.4f}")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a prototype video classifier from CLIP frame embeddings."
        )
    )
    add_multiclass_arguments(parser, artifacts_dir="artifacts")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    run_pipeline(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
