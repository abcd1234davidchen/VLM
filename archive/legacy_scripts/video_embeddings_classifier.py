"""End-to-end prototype: CLIP image embeddings per video frame -> bidirectional GRU -> class logits.

The training script builds or loads a fixed train/test split, caches per-video embedding tensors
on disk, then trains a small sequence model with masked pooling over variable (padded) lengths.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor

from video_dataset import (
    build_split,
    scan_all_classes,
)
from video_embedding_cache import (
    CachedEmbeddingDataset,
    CachedSample,
    extract_or_load_embeddings,
)


# If --class-ids is omitted, the script will scan the videos directory for all available classes.


class PrototypeVideoClassifier(nn.Module):
    """Layer-normed frame embeddings, packed BiGRU, then prototype similarity logits."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.gru = nn.GRU(
            input_dim,
            hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        # BiGRU output is 2*hidden_size; concatenating mean and max across time -> 4*hidden_size.
        pooled_dim = hidden_size * 4
        self.feature_norm = nn.LayerNorm(pooled_dim)
        self.feature_dropout = nn.Dropout(dropout)
        self.prototypes = nn.Parameter(torch.empty(num_classes, pooled_dim))
        self.logit_scale = nn.Parameter(torch.tensor(2.0))
        torch.nn.init.normal_(self.prototypes, mean=0.0, std=0.02)

    def forward(self, embeddings: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # Pack ignores padding; pooling uses lengths so padded tail frames do not affect mean/max.
        embeddings = self.input_norm(embeddings)
        packed = pack_padded_sequence(
            embeddings,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.gru(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        time_steps = torch.arange(output.size(1), device=lengths.device).unsqueeze(0)
        mask = time_steps < lengths.unsqueeze(1)
        # Max pool: masked positions -> -inf so max is taken over real frames only.
        masked_output = output.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        max_pool = masked_output.max(dim=1).values
        mean_pool = output.masked_fill(~mask.unsqueeze(-1), 0.0).sum(dim=1)
        mean_pool = mean_pool / lengths.unsqueeze(1)
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        pooled = self.feature_norm(pooled)
        pooled = self.feature_dropout(pooled)
        pooled = torch.nn.functional.normalize(pooled, dim=-1)
        prototypes = torch.nn.functional.normalize(self.prototypes, dim=-1)
        return self.logit_scale.exp().clamp(max=100.0) * pooled @ prototypes.t()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float,
    epoch: int = 1,
) -> float:
    """Average cross-entropy loss over the training set (weighted by batch size)."""
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch} Training", leave=False)
    for embeddings, lengths, labels in pbar:
        embeddings = embeddings.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(embeddings, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        batch_loss = loss.item()
        total_loss += batch_loss * embeddings.size(0)
        pbar.set_postfix(loss=f"{batch_loss:.4f}")

    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    samples: list[CachedSample],
    device: torch.device,
    class_ids: list[int],
) -> tuple[float, list[dict[str, object]]]:
    """Accuracy plus per-sample rows in the same order as samples (loader must not shuffle)."""
    model.eval()
    predictions: list[dict[str, object]] = []
    correct = 0
    sample_index = 0

    with torch.inference_mode():
        for embeddings, lengths, labels in loader:
            embeddings = embeddings.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            logits = model(embeddings, lengths)
            predicted = logits.argmax(dim=1)
            correct += (predicted == labels).sum().item()

            for truth, pred in zip(labels.cpu().tolist(), predicted.cpu().tolist()):
                sample = samples[sample_index]
                predictions.append(
                    {
                        "video": sample.video_path.name,
                        "true_class_id": sample.class_id,
                        "predicted_class_id": class_ids[pred],
                        "true_class_index": truth,
                        "predicted_class_index": pred,
                    }
                )
                sample_index += 1

    accuracy = correct / len(loader.dataset)
    return accuracy, predictions


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    """Split -> cache embeddings -> (optional) train GRU -> checkpoint + JSON metrics."""
    set_seed(args.seed)
    videos_dir = Path(args.videos_dir)
    cache_dir = Path(args.cache_dir)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    if args.class_ids:
        class_ids = list(args.class_ids)
    else:
        class_ids = scan_all_classes(videos_dir)
        print(f"Auto-discovered {len(class_ids)} classes in {videos_dir}")
        
    device = resolve_device(args.device)

    split = build_split(
        videos_dir=videos_dir,
        class_ids=class_ids,
        videos_per_split=args.videos_per_split,
        seed=args.seed,
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
        base_seed=args.seed,
    )

    if args.extract_only:
        # Useful for precomputing caches on one machine before training elsewhere.
        summary = {
            "device": str(device),
            "train_samples": len(cached["train"]),
            "test_samples": len(cached["test"]),
            "frame_stride": args.frame_stride,
            "max_seq_len": args.max_seq_len,
            "cache_dir": str(cache_dir),
        }
        print(json.dumps(summary, indent=2))
        return summary

    train_dataset = CachedEmbeddingDataset(cached["train"])
    test_dataset = CachedEmbeddingDataset(cached["test"])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # Separate loader for train accuracy: fixed order matches evaluate() sample indexing.
    train_eval_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    sample_item = train_dataset[0]
    input_dim = sample_item[0].shape[-1]
    model = PrototypeVideoClassifier(
        input_dim=input_dim,
        hidden_size=args.hidden_size,
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
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            args.max_grad_norm,
            epoch,
        )
        train_accuracy, _ = evaluate(
            model,
            train_eval_loader,
            cached["train"],
            device,
            class_ids,
        )
        
        t0 = time.time()
        test_accuracy, _ = evaluate(model, test_loader, cached["test"], device, class_ids)
        test_time = time.time() - t0
        
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
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
    checkpoint_path = artifacts_dir / "video_classifier.pt"
    results_path = artifacts_dir / "evaluation_results.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_ids": class_ids,
            "config": vars(args),
            "input_dim": input_dim,
        },
        checkpoint_path,
    )
    results = {
        "device": str(device),
        "history": history,
        "final_test_accuracy": final_accuracy,
        "checkpoint_path": str(checkpoint_path),
        "predictions": predictions,
    }
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved evaluation results to {results_path}")

    # 繪製並儲存準確度圖表
    epochs_list = [h["epoch"] for h in history]
    train_acc = [h["train_accuracy"] for h in history]
    test_acc = [h["test_accuracy"] for h in history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, train_acc, label="Train Accuracy", marker="o")
    plt.plot(epochs_list, test_acc, label="Test Accuracy", marker="o")
    plt.title("Training and Testing Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    
    graph_path = artifacts_dir / "accuracy_graph.png"
    plt.savefig(graph_path)
    plt.close()
    
    print(f"Saved accuracy graph to {graph_path}")
    print(f"Final test accuracy: {final_accuracy:.4f}")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a prototype video classifier from CLIP frame embeddings.",
    )
    parser.add_argument("--videos-dir", default="dataset/subset264")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--class-ids", type=int, nargs="+", default=None, help="List of class IDs. If omitted, uses all discovered classes in the dataset folder.")
    parser.add_argument("--videos-per-split", type=int, default=10)
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
