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

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor

from video_dataset import (
    VideoSample,
    build_split,
    resize_embedding_sequence,
    sample_video_frames,
    save_split_manifest,
    scan_all_classes,
)


# If --class-ids is omitted, the script will scan the videos directory for all available classes.


@dataclass
class CachedSample:
    embedding_path: Path
    video_path: Path
    class_id: int
    class_index: int
    split: str
    valid_length: int  # Non-padding rows in the embeddings tensor; see resize_embedding_sequence.

    def to_record(self) -> dict[str, object]:
        return {
            "embedding_path": str(self.embedding_path),
            "video_path": str(self.video_path),
            "class_id": self.class_id,
            "class_index": self.class_index,
            "split": self.split,
            "valid_length": self.valid_length,
        }


class CachedEmbeddingDataset(Dataset):
    """Loads all .pt embedding files into memory once (small prototype datasets)."""

    def __init__(self, samples: list[CachedSample]) -> None:
        self.samples = samples
        self.items = [torch.load(sample.embedding_path, map_location="cpu") for sample in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.items[index]
        embeddings = item["embeddings"].float()
        valid_length = torch.tensor(item["valid_length"], dtype=torch.long)
        label = torch.tensor(item["class_index"], dtype=torch.long)
        return embeddings, valid_length, label


class GRUVideoClassifier(nn.Module):
    """Layer-normed frame embeddings, packed BiGRU, then mean+max pool over valid timesteps."""

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
        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Dropout(dropout),
            nn.Linear(pooled_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_size, num_classes),
        )

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
        return self.classifier(pooled)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(requested_device: str) -> torch.device:
    if requested_device != "auto":
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_embedding_cache_path(cache_dir: Path, sample: VideoSample) -> Path:
    """One .pt file per video stem, under a split subfolder to avoid accidental overlap."""
    return cache_dir / "embeddings" / sample.split / f"{sample.video_path.stem}.pt"


def build_cache_metadata(
    sample: VideoSample,
    *,
    model_name: str,
    frame_stride: int,
    max_seq_len: int,
) -> dict[str, object]:
    return {
        "model_name": model_name,
        "frame_stride": frame_stride,
        "max_seq_len": max_seq_len,
        "video_path": str(sample.video_path),
        "split": sample.split,
        "class_id": sample.class_id,
        "class_index": sample.class_index,
    }


def is_cache_compatible(
    payload: dict[str, object],
    sample: VideoSample,
    *,
    model_name: str,
    frame_stride: int,
    max_seq_len: int,
) -> bool:
    embeddings = payload.get("embeddings")
    valid_length = payload.get("valid_length")
    metadata = payload.get("cache_metadata")

    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        return False
    if embeddings.shape[0] != max_seq_len:
        return False
    if not isinstance(valid_length, int):
        return False
    if valid_length < 1 or valid_length > max_seq_len:
        return False
    if not isinstance(metadata, dict):
        return False

    expected_metadata = build_cache_metadata(
        sample,
        model_name=model_name,
        frame_stride=frame_stride,
        max_seq_len=max_seq_len,
    )
    return metadata == expected_metadata


def embed_video(
    sample: VideoSample,
    *,
    processor: CLIPProcessor,
    model: CLIPModel,
    device: torch.device,
    frame_stride: int,
    max_seq_len: int,
    batch_size: int,
) -> tuple[torch.Tensor, int]:
    """Run CLIP on strided RGB frames, L2-normalize per frame, then fix sequence length."""
    frames = sample_video_frames(sample.video_path, frame_stride)
    features: list[torch.Tensor] = []

    for start in range(0, len(frames), batch_size):
        frame_batch = frames[start : start + batch_size]
        inputs = processor(images=frame_batch, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.inference_mode():
            batch_features = model.get_image_features(pixel_values=pixel_values)
        if hasattr(batch_features, "pooler_output"):
            batch_features = batch_features.pooler_output
        batch_features = torch.nn.functional.normalize(batch_features, dim=-1)
        features.append(batch_features.cpu())

    stacked = torch.cat(features, dim=0)
    return resize_embedding_sequence(stacked, max_seq_len)


def extract_or_load_embeddings(
    split: dict[str, list[VideoSample]],
    *,
    cache_dir: Path,
    model_name: str,
    frame_stride: int,
    max_seq_len: int,
    device: torch.device,
    batch_size: int,
    force_extract: bool,
    seed: int,
) -> dict[str, list[CachedSample]]:
    """Write selected_split.json; fill cache/embeddings/<split>/*.pt when missing or forced."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "selected_split.json"
    save_split_manifest(
        split,
        manifest_path,
        seed=seed,
        frame_stride=frame_stride,
        max_seq_len=max_seq_len,
    )

    processor: CLIPProcessor | None = None
    model: CLIPModel | None = None

    cached: dict[str, list[CachedSample]] = {"train": [], "test": []}
    refreshed_cache_files = 0

    for split_name in ("train", "test"):
        for sample in split[split_name]:
            cache_path = get_embedding_cache_path(cache_dir, sample)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_payload: dict[str, object] | None = None
            should_extract = force_extract or not cache_path.exists()
            if not should_extract:
                cache_payload = torch.load(cache_path, map_location="cpu")
                should_extract = not is_cache_compatible(
                    cache_payload,
                    sample,
                    model_name=model_name,
                    frame_stride=frame_stride,
                    max_seq_len=max_seq_len,
                )

            if should_extract:
                # Lazy-load CLIP only if at least one video needs extraction.
                if processor is None or model is None:
                    processor = CLIPProcessor.from_pretrained(model_name)
                    model = CLIPModel.from_pretrained(model_name).to(device)
                    model.eval()
                embeddings, valid_length = embed_video(
                    sample,
                    processor=processor,
                    model=model,
                    device=device,
                    frame_stride=frame_stride,
                    max_seq_len=max_seq_len,
                    batch_size=batch_size,
                )
                torch.save(
                    {
                        "embeddings": embeddings,
                        "valid_length": valid_length,
                        "class_id": sample.class_id,
                        "class_index": sample.class_index,
                        "video_path": str(sample.video_path),
                        "split": sample.split,
                        "cache_metadata": build_cache_metadata(
                            sample,
                            model_name=model_name,
                            frame_stride=frame_stride,
                            max_seq_len=max_seq_len,
                        ),
                    },
                    cache_path,
                )
                refreshed_cache_files += 1
            else:
                assert cache_payload is not None
                valid_length = int(cache_payload["valid_length"])

            cached[split_name].append(
                CachedSample(
                    embedding_path=cache_path,
                    video_path=sample.video_path,
                    class_id=sample.class_id,
                    class_index=sample.class_index,
                    split=sample.split,
                    valid_length=valid_length,
                )
            )

    # Human-readable index of all cached tensors (paths, labels, valid lengths).
    records_path = cache_dir / "embedding_manifest.json"
    payload = {
        split_name: [sample.to_record() for sample in samples]
        for split_name, samples in cached.items()
    }
    records_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if refreshed_cache_files:
        print(f"Refreshed {refreshed_cache_files} cached embedding files.")
    return cached


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float,
) -> float:
    """Average cross-entropy loss over the training set (weighted by batch size)."""
    model.train()
    total_loss = 0.0

    for embeddings, lengths, labels in loader:
        embeddings = embeddings.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(embeddings, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item() * embeddings.size(0)

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
        seed=args.seed,
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
    model = GRUVideoClassifier(
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
