"""Gate 3 full BlockGCN teacher comparison and selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from vlm.data.skeleton import PHASE1_ACTION_IDS, SKELETON_TENSOR_PREPROCESSING_ID
from vlm.models.skeletonclip import BLOCKGCN_SOURCE_REVISION, SkeletonLinearTeacher, SkeletonPrototypeTeacher
from vlm.skeleton_actions import PHASE1_ACTION_LABELS, PHASE1_PROMPT_TEMPLATES, phase1_prompts
from vlm.training.skeletonclip import CachedSkeletonDataset, _balanced_limit, _cached_dataset


XCLIP_MODEL = "microsoft/xclip-base-patch32"
XCLIP_REVISION = "a2e27a78a2b5d802e894b8a1ef14f3a8ce490963"
OPENAI_CLIP_MODEL = "openai/clip-vit-base-patch16"
OPENAI_CLIP_REVISION = "57c216476eefef5ab752ec549e440a49ae4ae5f3"
GATE3_SEEDS = (42, 43)
HEADS = ("linear", "xclip", "openai_clip")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() or None


def build_text_prototypes(*, model_name: str, revision: str, cache_path: Path, device: torch.device) -> tuple[Tensor, dict[str, object]]:
    expected = {
        "schema_version": 1,
        "model_name": model_name,
        "model_revision": revision,
        "ordered_class_ids": [f"A{value:03d}" for value in PHASE1_ACTION_IDS],
        "action_labels": list(PHASE1_ACTION_LABELS),
        "prompt_templates": list(PHASE1_PROMPT_TEMPLATES),
    }
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if payload.get("identity") != expected:
            raise ValueError(f"Stale text prototype cache: {cache_path}")
        return payload["prototypes"].float(), expected

    from transformers import AutoTokenizer, CLIPModel, XCLIPModel

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model_class = XCLIPModel if model_name == XCLIP_MODEL else CLIPModel
    text_model = model_class.from_pretrained(model_name, revision=revision).to(device).eval()
    prototypes = []
    with torch.inference_mode():
        for prompts in phase1_prompts():
            tokens = tokenizer(list(prompts), padding=True, return_tensors="pt")
            outputs = text_model.get_text_features(**{key: value.to(device) for key, value in tokens.items()})
            features = nn.functional.normalize(outputs.pooler_output.float(), dim=1)
            prototypes.append(nn.functional.normalize(features.mean(dim=0), dim=0).cpu())
    result = torch.stack(prototypes)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"identity": expected, "prototypes": result}, cache_path)
    return result, expected


def _metrics(logits: Tensor, labels: Tensor) -> dict[str, object]:
    logits, labels = logits.float(), labels.long()
    probabilities = logits.softmax(dim=1)
    predictions = probabilities.argmax(dim=1)
    confusion = torch.zeros(59, 59, dtype=torch.long)
    for actual, predicted in zip(labels.tolist(), predictions.tolist()):
        confusion[actual, predicted] += 1
    f1_values, supports = [], confusion.sum(dim=1)
    for index in range(59):
        tp = confusion[index, index].item()
        fp = confusion[:, index].sum().item() - tp
        fn = confusion[index, :].sum().item() - tp
        denominator = 2 * tp + fp + fn
        f1_values.append(2 * tp / denominator if denominator else 0.0)
    confidence = probabilities.max(dim=1).values
    correct = predictions.eq(labels).float()
    ece = 0.0
    for lower in torch.arange(0, 1, 0.1):
        mask = (confidence > lower) & (confidence <= lower + 0.1)
        if mask.any():
            ece += mask.float().mean().item() * abs(correct[mask].mean().item() - confidence[mask].mean().item())
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
    return {
        "accuracy": correct.mean().item(),
        "macro_f1": sum(f1_values) / 59,
        "nll": nn.functional.cross_entropy(logits, labels).item(),
        "ece": ece,
        "mean_entropy": entropy.mean().item(),
        "per_class_support": {f"A{index + 1:03d}": int(supports[index]) for index in range(59)},
        "per_class_f1": {f"A{index + 1:03d}": value for index, value in enumerate(f1_values)},
        "confusion_matrix": confusion.tolist(),
    }


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, object], Tensor, Tensor, list[str]]:
    model.eval()
    logits_rows, label_rows, sample_ids = [], [], []
    with torch.inference_mode():
        for inputs, labels, identifiers in loader:
            logits_rows.append(model(inputs.to(device, non_blocking=True)).float().cpu())
            label_rows.append(labels)
            sample_ids.extend(identifiers)
    logits, labels = torch.cat(logits_rows), torch.cat(label_rows)
    return _metrics(logits, labels), logits, labels, sample_ids


def _checkpoint_identity(*, head: str, seed: int, epochs: int, frames: int, manifest_sha256: str, text_identity: dict[str, object] | None) -> dict[str, object]:
    return {
        "schema_version": 1,
        "phase": "phase1_gate3",
        "backbone": "blockgcn",
        "head": head,
        "seed": seed,
        "epochs": epochs,
        "ordered_class_ids": [f"A{value:03d}" for value in PHASE1_ACTION_IDS],
        "excluded_class_ids": ["A060"],
        "action_labels": list(PHASE1_ACTION_LABELS),
        "preprocessing_id": SKELETON_TENSOR_PREPROCESSING_ID,
        "frames": frames,
        "source_revision": BLOCKGCN_SOURCE_REVISION,
        "manifest_sha256": manifest_sha256,
        "text": text_identity,
    }


def train_run(*, head: str, seed: int, datasets: dict[str, CachedSkeletonDataset], output_dir: Path, epochs: int, batch_size: int, workers: int, frames: int, manifest_sha256: str, prototypes: Tensor | None = None, text_identity: dict[str, object] | None = None, resume: bool = False) -> dict[str, object]:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, generator=generator, num_workers=workers, persistent_workers=workers > 0, pin_memory=True),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=workers, persistent_workers=workers > 0, pin_memory=True),
    }
    model: nn.Module = SkeletonLinearTeacher("blockgcn") if head == "linear" else SkeletonPrototypeTeacher("blockgcn", prototypes if prototypes is not None else torch.empty(0))
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.025, momentum=0.9, nesterov=True, weight_decay=4e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    identity = _checkpoint_identity(head=head, seed=seed, epochs=epochs, frames=frames, manifest_sha256=manifest_sha256, text_identity=text_identity)
    output_dir.mkdir(parents=True, exist_ok=True)
    history, best_key, best_epoch = [], (-1.0, -1.0), 0
    start_epoch = 1
    last_checkpoint = output_dir / "last_checkpoint.pt"
    if resume and last_checkpoint.exists():
        checkpoint = torch.load(last_checkpoint, map_location=device, weights_only=False)
        if checkpoint.get("identity") != identity:
            raise ValueError(f"Cannot resume checkpoint with a different identity: {last_checkpoint}")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        generator.set_state(checkpoint["data_generator_state"].cpu())
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_states"]])
        history = checkpoint["history"]
        start_epoch = int(checkpoint["epoch"]) + 1
        if history:
            best_row = max(history, key=lambda row: (float(row["validation"]["macro_f1"]), float(row["validation"]["accuracy"])))
            best_key = (float(best_row["validation"]["macro_f1"]), float(best_row["validation"]["accuracy"]))
            best_epoch = int(best_row["epoch"])
        print(f"{head} seed={seed} resuming at epoch={start_epoch}/{epochs}", flush=True)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss, correct, seen = 0.0, 0, 0
        for inputs, labels, _ in loaders["train"]:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = nn.functional.cross_entropy(logits.float(), labels, label_smoothing=0.1)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss in {head} seed={seed} epoch={epoch}")
            loss.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                raise RuntimeError(f"Non-finite gradient in {head} seed={seed} epoch={epoch}")
            optimizer.step()
            total_loss += loss.item() * labels.numel()
            correct += logits.argmax(dim=1).eq(labels).sum().item()
            seen += labels.numel()
        validation, val_logits, val_labels, val_ids = _evaluate(model, loaders["val"], device)
        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], "train_loss": total_loss / seen, "train_accuracy": correct / seen, "validation": validation}
        history.append(row)
        key = (float(validation["macro_f1"]), float(validation["accuracy"]))
        scheduler.step()
        payload = {"identity": identity, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch, "history": history, "data_generator_state": generator.get_state(), "torch_rng_state": torch.get_rng_state(), "cuda_rng_states": torch.cuda.get_rng_state_all()}
        torch.save(payload, output_dir / "last_checkpoint.pt")
        if key > best_key:
            best_key, best_epoch = key, epoch
            torch.save(payload, output_dir / "best_checkpoint.pt")
            torch.save({"sample_ids": val_ids, "labels": val_labels, "logits": val_logits}, output_dir / "validation_logits.pt")
            predictions = [{"sample_id": sample_id, "label": int(label), "prediction": int(prediction)} for sample_id, label, prediction in zip(val_ids, val_labels, val_logits.argmax(dim=1))]
            (output_dir / "validation_predictions.json").write_text(json.dumps(predictions, indent=2) + "\n")
        print(f"{head} seed={seed} epoch={epoch}/{epochs} macro_f1={key[0]:.4f} accuracy={key[1]:.4f}", flush=True)
    elapsed = time.perf_counter() - started
    best_payload = torch.load(output_dir / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_payload["state_dict"], strict=True)
    selected, _, _, _ = _evaluate(model, loaders["val"], device)
    expected = next(row["validation"] for row in history if row["epoch"] == best_epoch)
    if selected != expected:
        raise RuntimeError("Strict checkpoint reload did not reproduce selected validation metrics")
    result = {
        "status": "complete", "head": head, "seed": seed, "epochs": epochs, "best_epoch": best_epoch,
        "best_validation": selected, "train_samples": len(datasets["train"]), "validation_samples": len(datasets["val"]),
        "elapsed_seconds": elapsed, "examples_per_second": max(0, epochs - start_epoch + 1) * len(datasets["train"]) / elapsed,
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(), "checkpoint_identity": identity,
        "trainable_parameters": [name for name, parameter in model.named_parameters() if parameter.requires_grad], "history": history,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def select_teacher(linear: dict[str, object], xclip: dict[str, object]) -> dict[str, object]:
    linear_metrics = linear["best_validation"]
    xclip_metrics = xclip["best_validation"]
    macro_gap = float(linear_metrics["macro_f1"]) - float(xclip_metrics["macro_f1"])
    nll_better = float(xclip_metrics["nll"]) < float(linear_metrics["nll"])
    ece_better = float(xclip_metrics["ece"]) < float(linear_metrics["ece"])
    nll_not_worse = float(xclip_metrics["nll"]) <= float(linear_metrics["nll"])
    ece_not_worse = float(xclip_metrics["ece"]) <= float(linear_metrics["ece"])
    passes = macro_gap <= 0.02 and ((nll_better and ece_not_worse) or (ece_better and nll_not_worse))
    return {
        "selected_head": "xclip" if passes else "linear",
        "xclip_passes_quality_gate": passes,
        "macro_f1_gap": macro_gap,
        "rule": "XCLIP macro-F1 may trail linear by at most 0.02; NLL or ECE must improve and the other must not worsen.",
    }


def _mean_result(results: list[dict[str, object]]) -> dict[str, object]:
    metrics = [result["best_validation"] for result in results]
    return {"best_validation": {name: sum(float(row[name]) for row in metrics) / len(metrics) for name in ("macro_f1", "accuracy", "nll", "ece", "mean_entropy")}}


def write_report(*, artifacts_dir: Path, manifest: Path, expected_heads: Sequence[str], expected_seeds: Sequence[int]) -> dict[str, object]:
    results = [json.loads(path.read_text()) for path in sorted(artifacts_dir.glob("*/result.json"))]
    completed = {(str(result["head"]), int(result["seed"])) for result in results if result.get("status") == "complete"}
    expected = {(head, seed) for head in expected_heads for seed in expected_seeds}
    by_head = {head: [result for result in results if result["head"] == head and int(result["seed"]) in expected_seeds] for head in expected_heads}
    aggregates = {head: _mean_result(rows)["best_validation"] for head, rows in by_head.items() if rows}
    report: dict[str, object] = {
        "schema_version": 1, "status": "complete" if expected <= completed else "partial", "git_revision": _git_revision(),
        "manifest": str(manifest), "manifest_sha256": _sha256(manifest),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "expected_runs": [{"head": head, "seed": seed} for head, seed in sorted(expected)], "runs": results, "aggregates": aggregates,
    }
    if by_head.get("linear") and by_head.get("xclip"):
        selection = select_teacher(_mean_result(by_head["linear"]), _mean_result(by_head["xclip"]))
        selected_runs = by_head[str(selection["selected_head"])]
        winner = max(selected_runs, key=lambda result: (float(result["best_validation"]["macro_f1"]), float(result["best_validation"]["accuracy"])))
        selection["selected_run"] = {"head": winner["head"], "seed": winner["seed"], "checkpoint": str(artifacts_dir / f"{winner['head']}_seed{winner['seed']}" / "best_checkpoint.pt")}
        report["selection"] = selection
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "teacher_comparison_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run SkeletonCLIP Phase 1 Gate 3 teacher comparison.")
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/skeleton_data/paired_manifest.json"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/skeletonclip_gate3"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/skeletonclip_gate3"))
    parser.add_argument("--heads", nargs="+", choices=HEADS, default=("linear",), help="Heads to train; defaults to the linear baseline so adaptive XCLIP/reference runs require explicit opt-in.")
    parser.add_argument("--seeds", nargs="+", type=int, default=GATE3_SEEDS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--openai-clip-revision", default=OPENAI_CLIP_REVISION)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)
    if args.summarize_only:
        report = write_report(artifacts_dir=args.artifacts_dir, manifest=args.manifest, expected_heads=args.heads, expected_seeds=args.seeds)
        print(f"Gate 3 report status={report['status']}", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Gate 3 requires CUDA")
    manifest_payload = json.loads(args.manifest.read_text())
    records = {"train": _balanced_limit(manifest_payload["train"], args.max_train_samples), "val": _balanced_limit(manifest_payload["val"], args.max_val_samples)}
    suffix = f"{len(records['train'])}_{len(records['val'])}_{args.frames}"
    datasets = {split: _cached_dataset(rows, frames=args.frames, workers=args.workers, path=args.cache_dir / f"{split}_{suffix}.pt") for split, rows in records.items()}
    text_contracts: dict[str, tuple[Tensor, dict[str, object]]] = {}
    if "xclip" in args.heads:
        text_contracts["xclip"] = build_text_prototypes(model_name=XCLIP_MODEL, revision=XCLIP_REVISION, cache_path=args.cache_dir / "xclip_text_prototypes.pt", device=torch.device("cuda"))
    if "openai_clip" in args.heads:
        revision = str(args.openai_clip_revision)
        text_contracts["openai_clip"] = build_text_prototypes(model_name=OPENAI_CLIP_MODEL, revision=revision, cache_path=args.cache_dir / "openai_clip_text_prototypes.pt", device=torch.device("cuda"))
    results = []
    for head in args.heads:
        prototypes, text_identity = text_contracts.get(head, (None, None))
        for seed in args.seeds:
            results.append(train_run(head=head, seed=seed, datasets=datasets, output_dir=args.artifacts_dir / f"{head}_seed{seed}", epochs=args.epochs, batch_size=args.batch_size, workers=args.workers, frames=args.frames, manifest_sha256=_sha256(args.manifest), prototypes=prototypes, text_identity=text_identity, resume=args.resume))
    write_report(artifacts_dir=args.artifacts_dir, manifest=args.manifest, expected_heads=args.heads, expected_seeds=args.seeds)


if __name__ == "__main__":
    main()
