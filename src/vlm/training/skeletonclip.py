"""Gate 2 controlled bake-off for SkeletonCLIP skeleton backbones."""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import statistics
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from vlm.data.skeleton import PHASE1_ACTION_IDS, SKELETON_TENSOR_PREPROCESSING_ID, load_skeleton_tensor
from vlm.models.skeletonclip import (
    BLOCKGCN_SOURCE_REVISION,
    HYPERFORMER_SOURCE_REVISION,
    SkeletonCheckpointIdentity,
    SkeletonLinearTeacher,
)


SOURCE_REVISIONS = {"blockgcn": BLOCKGCN_SOURCE_REVISION, "hyperformer": HYPERFORMER_SOURCE_REVISION}
GATE2_SEEDS = (42, 43)


class PairedSkeletonDataset(Dataset[tuple[Tensor, int, str]]):
    def __init__(self, records: list[dict[str, object]], *, frames: int) -> None:
        self.records = records
        self.frames = frames

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, int, str]:
        record = self.records[index]
        return (
            load_skeleton_tensor(Path(str(record["skeleton_path"])), frames=self.frames),
            int(record["class_index"]),
            str(record["sample_id"]),
        )


class CachedSkeletonDataset(Dataset[tuple[Tensor, int, str]]):
    def __init__(self, inputs: Tensor, labels: Tensor, sample_ids: list[str]) -> None:
        self.inputs, self.labels, self.sample_ids = inputs, labels, sample_ids

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> tuple[Tensor, int, str]:
        return self.inputs[index], int(self.labels[index]), self.sample_ids[index]


def _cached_dataset(records: list[dict[str, object]], *, frames: int, workers: int, path: Path) -> CachedSkeletonDataset:
    expected_ids = [str(record["sample_id"]) for record in records]
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["sample_ids"] != expected_ids or payload["preprocessing_id"] != SKELETON_TENSOR_PREPROCESSING_ID:
            raise ValueError(f"Stale skeleton tensor cache: {path}")
        return CachedSkeletonDataset(payload["inputs"], payload["labels"], payload["sample_ids"])
    loader = DataLoader(PairedSkeletonDataset(records, frames=frames), batch_size=64, shuffle=False, num_workers=workers, persistent_workers=workers > 0)
    inputs = torch.empty(len(records), 3, frames, 25, 2)
    labels = torch.empty(len(records), dtype=torch.long)
    sample_ids: list[str] = []
    offset = 0
    for batch_index, (batch_inputs, batch_labels, batch_ids) in enumerate(loader, start=1):
        batch_end = offset + len(batch_ids)
        inputs[offset:batch_end].copy_(batch_inputs)
        labels[offset:batch_end].copy_(batch_labels)
        sample_ids.extend(batch_ids)
        offset = batch_end
        if batch_index == 1 or batch_index % 100 == 0 or offset == len(records):
            print(f"skeleton cache {path.name}: {offset}/{len(records)}", flush=True)
    payload = {"inputs": inputs, "labels": labels, "sample_ids": sample_ids, "preprocessing_id": SKELETON_TENSOR_PREPROCESSING_ID}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return CachedSkeletonDataset(payload["inputs"], payload["labels"], sample_ids)


def _balanced_limit(records: list[dict[str, object]], limit: int | None) -> list[dict[str, object]]:
    if limit is None or limit >= len(records):
        return records
    grouped: dict[int, list[dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(int(record["class_index"]), []).append(record)
    selected: list[dict[str, object]] = []
    offset = 0
    while len(selected) < limit:
        added = False
        for class_index in sorted(grouped):
            if offset < len(grouped[class_index]) and len(selected) < limit:
                selected.append(grouped[class_index][offset])
                added = True
        if not added:
            break
        offset += 1
    return selected


def _metrics(logits: Tensor, labels: Tensor) -> dict[str, object]:
    probabilities = logits.float().softmax(dim=1)
    predictions = probabilities.argmax(dim=1)
    accuracy = predictions.eq(labels).float().mean().item()
    f1_values = []
    for class_index in range(59):
        predicted = predictions.eq(class_index)
        actual = labels.eq(class_index)
        tp = (predicted & actual).sum().item()
        fp = (predicted & ~actual).sum().item()
        fn = (~predicted & actual).sum().item()
        denominator = 2 * tp + fp + fn
        f1_values.append(2 * tp / denominator if denominator else 0.0)
    confidence, _ = probabilities.max(dim=1)
    correct = predictions.eq(labels).float()
    ece = 0.0
    for lower in torch.linspace(0, 0.9, 10):
        mask = (confidence > lower) & (confidence <= lower + 0.1)
        if mask.any():
            ece += mask.float().mean().item() * abs(correct[mask].mean().item() - confidence[mask].mean().item())
    return {
        "accuracy": accuracy,
        "macro_f1": sum(f1_values) / len(f1_values),
        "nll": nn.functional.cross_entropy(logits.float(), labels).item(),
        "ece": ece,
        "per_class_f1": {f"A{index + 1:03d}": value for index, value in enumerate(f1_values)},
    }


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, object], list[dict[str, object]]]:
    model.eval()
    all_logits, all_labels, rows = [], [], []
    with torch.no_grad():
        for inputs, labels, sample_ids in loader:
            logits = model(inputs.to(device)).float().cpu()
            all_logits.append(logits)
            all_labels.append(labels)
            for sample_id, label, prediction in zip(sample_ids, labels.tolist(), logits.argmax(dim=1).tolist()):
                rows.append({"sample_id": sample_id, "label": label, "prediction": prediction})
    return _metrics(torch.cat(all_logits), torch.cat(all_labels)), rows


def _identity(backbone: str) -> SkeletonCheckpointIdentity:
    return SkeletonCheckpointIdentity(
        backbone=backbone,  # type: ignore[arg-type]
        ordered_class_ids=tuple(f"A{action:03d}" for action in PHASE1_ACTION_IDS),
        preprocessing_id=SKELETON_TENSOR_PREPROCESSING_ID,
        source_revision=SOURCE_REVISIONS[backbone],
    )


def train_run(*, backbone: str, seed: int, datasets: dict[str, CachedSkeletonDataset], output_dir: Path, epochs: int, batch_size: int, workers: int, frames: int, resume: bool = False) -> dict[str, object]:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, generator=generator, num_workers=workers, persistent_workers=workers > 0, pin_memory=True)
    val_loader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=workers, persistent_workers=workers > 0, pin_memory=True)
    model = SkeletonLinearTeacher(backbone).to(device)  # type: ignore[arg-type]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.025, momentum=0.9, nesterov=True, weight_decay=4e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    output_dir.mkdir(parents=True, exist_ok=True)
    history, best_key, best_epoch = [], (-1.0, -1.0), 0
    start_epoch = 1
    last_checkpoint = output_dir / "last_checkpoint.pt"
    if resume and last_checkpoint.exists():
        checkpoint = torch.load(last_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history = checkpoint["history"]
        start_epoch = int(checkpoint["epoch"]) + 1
        if history:
            best_row = max(history, key=lambda row: (float(row["validation"]["macro_f1"]), float(row["validation"]["accuracy"])))
            best_key = (float(best_row["validation"]["macro_f1"]), float(best_row["validation"]["accuracy"]))
            best_epoch = int(best_row["epoch"])
        scheduler.step()
        print(f"{backbone} seed={seed} resuming at epoch={start_epoch}/{epochs}", flush=True)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        loss_total, correct, seen = 0.0, 0, 0
        for inputs, labels, _sample_ids in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = nn.functional.cross_entropy(logits.float(), labels, label_smoothing=0.1)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss in {backbone} seed {seed} epoch {epoch}")
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * labels.numel()
            correct += logits.argmax(1).eq(labels).sum().item()
            seen += labels.numel()
        validation, predictions = _evaluate(model, val_loader, device)
        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], "train_loss": loss_total / seen, "train_accuracy": correct / seen, "validation": validation}
        history.append(row)
        key = (float(validation["macro_f1"]), float(validation["accuracy"]))
        payload = {"identity": _identity(backbone).__dict__, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch, "history": history}
        torch.save(payload, output_dir / "last_checkpoint.pt")
        if key > best_key:
            best_key, best_epoch = key, epoch
            torch.save(payload, output_dir / "best_checkpoint.pt")
            (output_dir / "validation_predictions.json").write_text(json.dumps(predictions, indent=2) + "\n")
        scheduler.step()
        print(f"{backbone} seed={seed} epoch={epoch}/{epochs} macro_f1={key[0]:.4f} accuracy={key[1]:.4f}", flush=True)
    elapsed = time.perf_counter() - started
    result = {"status": "complete", "backbone": backbone, "seed": seed, "epochs": epochs, "best_epoch": best_epoch, "best_macro_f1": best_key[0], "best_accuracy": best_key[1], "train_samples": len(datasets["train"]), "validation_samples": len(datasets["val"]), "elapsed_seconds": elapsed, "examples_per_second": epochs * len(datasets["train"]) / elapsed, "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(), "history": history}
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _write_bakeoff_report(*, artifacts_dir: Path, manifest: Path) -> dict[str, object]:
    results = [json.loads(path.read_text()) for path in sorted(artifacts_dir.glob("*/result.json"))]
    expected_runs = {(backbone, seed) for backbone in SOURCE_REVISIONS for seed in GATE2_SEEDS}
    completed_runs = {(str(result["backbone"]), int(result["seed"])) for result in results if result.get("status") == "complete" and int(result["epochs"]) == 20}
    aggregates: dict[str, dict[str, object]] = {}
    for backbone in SOURCE_REVISIONS:
        backbone_results = sorted((result for result in results if result["backbone"] == backbone), key=lambda result: int(result["seed"]))
        if not backbone_results:
            continue
        selected_rows = []
        for result in backbone_results:
            selected_rows.append(next(row for row in result["history"] if int(row["epoch"]) == int(result["best_epoch"])))
        def aggregate(values: list[float]) -> dict[str, float]:
            return {"mean": statistics.fmean(values), "population_stddev": statistics.pstdev(values)}
        aggregates[backbone] = {
            "seeds": [int(result["seed"]) for result in backbone_results],
            "macro_f1": aggregate([float(result["best_macro_f1"]) for result in backbone_results]),
            "accuracy": aggregate([float(result["best_accuracy"]) for result in backbone_results]),
            "nll": aggregate([float(row["validation"]["nll"]) for row in selected_rows]),
            "ece": aggregate([float(row["validation"]["ece"]) for row in selected_rows]),
            "examples_per_second": aggregate([float(result["examples_per_second"]) for result in backbone_results]),
            "peak_memory_allocated_bytes": max(int(result["peak_memory_allocated_bytes"]) for result in backbone_results),
            "best_macro_f1_by_seed": {str(result["seed"]): float(result["best_macro_f1"]) for result in backbone_results},
            "best_epoch_by_seed": {str(result["seed"]): int(result["best_epoch"]) for result in backbone_results},
        }
    finalists = []
    for backbone, aggregate_row in aggregates.items():
        scores = list(aggregate_row["best_macro_f1_by_seed"].values())
        dominated = any(
            other != backbone
            and min(aggregates[other]["best_macro_f1_by_seed"].values()) > max(scores)
            and float(aggregates[other]["examples_per_second"]["mean"]) >= float(aggregate_row["examples_per_second"]["mean"])
            for other in aggregates
        )
        if not dominated:
            finalists.append(backbone)
    finalists.sort(key=lambda backbone: float(aggregates[backbone]["macro_f1"]["mean"]), reverse=True)
    status = "complete" if completed_runs == expected_runs else "partial"
    report = {
        "schema_version": 1,
        "status": status,
        "manifest": str(manifest),
        "preprocessing_id": SKELETON_TENSOR_PREPROCESSING_ID,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "expected_runs": [{"backbone": backbone, "seed": seed} for backbone, seed in sorted(expected_runs)],
        "results": results,
        "aggregates": aggregates,
        "finalists": finalists[:2],
        "selection_rule": "Retain at most two backbones; eliminate a backbone when every seed is below every seed of another backbone and the dominating backbone is at least as fast on average.",
    }
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (artifacts_dir / "bakeoff_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run SkeletonCLIP Gate 2 bake-off.")
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/skeleton_data/paired_manifest.json"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/skeletonclip_gate2"))
    parser.add_argument("--backbones", nargs="+", choices=("blockgcn", "hyperformer"), default=("blockgcn", "hyperformer"))
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 43))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/skeletonclip_gate2"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)
    if args.summarize_only:
        report = _write_bakeoff_report(artifacts_dir=args.artifacts_dir, manifest=args.manifest)
        print(f"Gate 2 report status={report['status']} finalists={report['finalists']}", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Gate 2 requires CUDA")
    payload = json.loads(args.manifest.read_text())
    records = {"train": _balanced_limit(payload["train"], args.max_train_samples), "val": _balanced_limit(payload["val"], args.max_val_samples)}
    cache_suffix = f"{len(records['train'])}_{len(records['val'])}_{args.frames}"
    datasets = {
        split: _cached_dataset(split_records, frames=args.frames, workers=args.workers, path=args.cache_dir / f"{split}_{cache_suffix}.pt")
        for split, split_records in records.items()
    }
    results = []
    for backbone in args.backbones:
        for seed in args.seeds:
            results.append(train_run(backbone=backbone, seed=seed, datasets=datasets, output_dir=args.artifacts_dir / f"{backbone}_seed{seed}", epochs=args.epochs, batch_size=args.batch_size, workers=args.workers, frames=args.frames, resume=args.resume))
    _write_bakeoff_report(artifacts_dir=args.artifacts_dir, manifest=args.manifest)


if __name__ == "__main__":
    main()
