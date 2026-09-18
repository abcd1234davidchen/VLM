"""Phase 3 Gate 1: matched RGB-path frozen/vision-LoRA CE controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from transformers import XCLIPModel, XCLIPProcessor

from vlm.models.lora import lora_state_dict, load_lora_state_dict, merge_lora, merged_base_state_dict
from vlm.models.xclip_vision_lora import XCLIPVisionClassifier, XCLIPVisionLoRAClassifier
from vlm.processed_rgb_cache import ProcessedRGBCache
from vlm.processed_uint8_cache import ProcessedUInt8Cache
from vlm.reporting import write_json
from vlm.skeleton_teacher_cache import sha256_file
from vlm.training.engine import set_seed
from vlm.xclip_features import FRAME_SAMPLING_STRATEGY, read_uniform_video_frames


class RGBVideoDataset(Dataset):
    def __init__(self, rows: list[dict[str, object]], processor: XCLIPProcessor, cache: ProcessedRGBCache | ProcessedUInt8Cache | None = None) -> None:
        self.rows = rows
        self.processor = processor
        self.cache = cache

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        video_path = Path(str(row["video_path"]))
        if self.cache is None:
            frames = read_uniform_video_frames(video_path, 8)
            pixels = self.processor(images=list(frames), return_tensors="pt")["pixel_values"].squeeze(0)
        else:
            pixels = self.cache.load(video_path)
        return pixels, int(row["class_id"]) - 1


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, payload)
    temporary.replace(path)


def _atomic_checkpoint(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _log(path: Path, payload: dict[str, object]) -> None:
    message = json.dumps(payload, sort_keys=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(message + "\n")
        stream.flush()
    print(message, flush=True)


def _model(args: argparse.Namespace, device: torch.device) -> XCLIPVisionLoRAClassifier:
    base = XCLIPModel.from_pretrained(args.model_name, revision=args.model_revision, local_files_only=True)
    model = XCLIPVisionLoRAClassifier(
        base, num_classes=120, head_dropout=args.head_dropout,
        upper_blocks=0 if args.arm == "frozen" else args.upper_blocks,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, feature_noise_std=args.feature_noise_std,
    )
    return model.to(device)


def _optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.AdamW:
    groups: dict[tuple[str, bool], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        family = "head" if name.startswith("classifier.") else "vision" if name.startswith("vision_model.") else "mit"
        no_decay = parameter.ndim == 1 or name.endswith(".bias")
        groups.setdefault((family, no_decay), []).append(parameter)
    rates = {"head": args.head_lr, "mit": args.mit_lr, "vision": args.vision_lr}
    return torch.optim.AdamW([
        {"params": parameters, "lr": rates[family], "weight_decay": 0.0 if no_decay else args.weight_decay}
        for (family, no_decay), parameters in groups.items()
    ])


def _snapshot(model: XCLIPVisionLoRAClassifier) -> dict[str, object]:
    return {
        "lora_state_dict": lora_state_dict(model),
        "classifier_state_dict": {key: value.detach().cpu().clone() for key, value in model.classifier.state_dict().items()},
        "target_modules": model.target_modules,
    }


def _restore(model: XCLIPVisionLoRAClassifier, payload: dict[str, object]) -> None:
    if payload["target_modules"] != model.target_modules:
        raise ValueError("Checkpoint LoRA targets differ from the current arm")
    load_lora_state_dict(model, payload["lora_state_dict"])
    model.classifier.load_state_dict(payload["classifier_state_dict"], strict=True)


def _same_experiment_except_cache(old: dict[str, object], new: dict[str, object]) -> bool:
    """Permit an explicit checkpoint handoff when only exact RGB transport changes."""
    old = dict(old)
    new = dict(new)
    for item in (old, new):
        item.pop("source_sha256", None)
        item.pop("continued_from_checkpoint_sha256", None)
        config = dict(item["config"])
        config.pop("uint8_cache_dir", None)
        item["config"] = config
    return old == new


def _load_ce_initialization(model: XCLIPVisionLoRAClassifier, path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("class_ids") != list(range(1, 121)):
        raise ValueError("CE initialization has a different 120-class order")
    source = {f"mit.{name}": value for name, value in checkpoint["lora_state_dict"].items()}
    expected = {
        name for name, _ in model.named_parameters()
        if name.startswith("mit.") and name.endswith(("lora_a", "lora_b"))
    }
    if set(source) != expected:
        raise ValueError("CE initialization MIT LoRA keys do not match")
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in source.items():
            parameters[name].copy_(value)
    model.classifier.load_state_dict(checkpoint["classifier_state_dict"], strict=True)


def _stratified_subset(rows: list[dict[str, object]], fraction: float) -> list[dict[str, object]]:
    by_class: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_class[int(row["class_id"])].append(row)
    if len(by_class) != 120:
        raise ValueError("Pilot train split must contain all 120 classes")
    selected: list[dict[str, object]] = []
    for class_id in sorted(by_class):
        ranked = sorted(by_class[class_id], key=lambda row: hashlib.sha256(str(row["video_path"]).encode()).hexdigest())
        selected.extend(ranked[:max(1, round(len(ranked) * fraction))])
    return sorted(selected, key=lambda row: str(row["video_path"]))


def _loader(dataset: Dataset, args: argparse.Namespace, *, train: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=train,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0, worker_init_fn=_seed_worker,
        generator=generator,
    )


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device, *, max_batches: int | None) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    model.eval()
    logits_rows: list[torch.Tensor] = []
    labels_rows: list[torch.Tensor] = []
    with torch.inference_mode():
        for index, (pixels, labels) in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            logits_rows.append(model(pixels.to(device, non_blocking=True)).float().cpu())
            labels_rows.append(labels)
    logits, labels = torch.cat(logits_rows), torch.cat(labels_rows)
    predicted = logits.argmax(1)
    confusion = torch.bincount(labels * 120 + predicted, minlength=120 * 120).reshape(120, 120)
    tp = confusion.diag().float()
    denominator = confusion.sum(0).float() + confusion.sum(1).float()
    macro_f1 = torch.where(denominator > 0, 2 * tp / denominator.clamp_min(1), 0).mean()
    metrics = {
        "accuracy": float((predicted == labels).float().mean()),
        "macro_f1": float(macro_f1),
        "nll": float(nn.functional.cross_entropy(logits, labels)),
        "samples": int(labels.numel()),
    }
    return metrics, logits, labels


def _train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, args: argparse.Namespace, device: torch.device, *, epoch: int, output: Path) -> tuple[float, int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    loss_sum = 0.0
    seen = 0
    started = time.perf_counter()
    _log(output / "train.log", {"arm": args.arm, "epoch": epoch, "event": "epoch_started", "expected_samples": len(loader.dataset)})
    for index, (pixels, labels) in enumerate(loader):
        if args.max_train_batches is not None and index >= args.max_train_batches:
            break
        pixels, labels = pixels.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=args.amp):
            logits = model(pixels)
            loss = criterion(logits.float(), labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite CE loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_((p for p in model.parameters() if p.requires_grad), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        count = labels.numel()
        seen += count
        loss_sum += float(loss.detach()) * count
        if (index + 1) % 25 == 0:
            elapsed = time.perf_counter() - started
            rate = seen / elapsed
            progress = {"state": "training", "arm": args.arm, "epoch": epoch, "completed_batches": index + 1, "samples_seen": seen, "expected_samples": len(loader.dataset), "examples_per_second": rate, "epoch_eta_seconds": max(0.0, (len(loader.dataset) - seen) / rate), "test_evaluated": False}
            _atomic_json(output / "progress.json", progress)
            _log(output / "train.log", {"event": "batch_progress", **progress})
    if seen == 0:
        raise RuntimeError("Training consumed no samples")
    _log(output / "train.log", {"arm": args.arm, "epoch": epoch, "event": "train_epoch_completed", "samples_seen": seen, "elapsed_seconds": time.perf_counter() - started})
    return loss_sum / seen, seen


def run_arm(args: argparse.Namespace, train_rows: list[dict[str, object]], val_rows: list[dict[str, object]], identity: dict[str, object]) -> dict[str, object]:
    arm_name = "frozen_vision" if args.arm == "frozen" else "vision_lora"
    output = Path(args.artifacts_dir) / f"{arm_name}_ce_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    set_seed(args.seed)
    processor = XCLIPProcessor.from_pretrained(args.model_name, revision=args.model_revision, local_files_only=True)
    cache = ProcessedRGBCache(Path(args.rgb_cache_dir), processor, model_name=args.model_name, model_revision=args.model_revision) if args.rgb_cache_dir else None
    if args.uint8_cache_dir:
        cache = ProcessedUInt8Cache(Path(args.uint8_cache_dir), processor, model_name=args.model_name, model_revision=args.model_revision)
    train_loader = _loader(RGBVideoDataset(train_rows, processor, cache), args, train=True)
    val_loader = _loader(RGBVideoDataset(val_rows, processor, cache), args, train=False)
    model = _model(args, device)
    if args.init_adapter:
        _load_ce_initialization(model, Path(args.init_adapter))
    model.eval()
    initial_pixels, _ = next(iter(val_loader))
    with torch.inference_mode():
        initial_logits = model(initial_pixels.to(device)).float().cpu()
    initial_logits_sha256 = hashlib.sha256(initial_logits.numpy().tobytes()).hexdigest()
    del initial_pixels, initial_logits
    audit = model.trainable_parameter_audit().as_dict()
    vision_names = [name for name in audit["names"] if name.startswith("vision_model.")]
    if bool(vision_names) != (args.arm == "vision"):
        raise RuntimeError("Vision trainable parameter audit does not match arm")
    optimizer = _optimizer(args, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    amp_identity = {"enabled": args.amp, "dtype": "float16" if args.amp else "float32"}
    run_identity = {**identity, "arm": args.arm, "amp": amp_identity}
    _atomic_json(output / "run_lock.json", run_identity)
    last_path = output / "last_checkpoint.pt"
    history: list[dict[str, object]] = []
    best_accuracy = -1.0
    best_epoch = 0
    start_epoch = 1
    resume_path = last_path if args.resume and last_path.exists() else Path(args.resume_from) if args.resume_from else None
    if resume_path is not None:
        previous = torch.load(resume_path, map_location="cpu", weights_only=False)
        migrating = resume_path != last_path
        if migrating:
            if not _same_experiment_except_cache(previous["identity"], run_identity):
                raise ValueError("Checkpoint handoff differs beyond RGB cache transport or source version")
        elif previous["identity"] != run_identity:
            raise ValueError("Resume identity differs from the locked run")
        _restore(model, previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        scheduler.load_state_dict(previous["scheduler"])
        scaler.load_state_dict(previous["scaler"])
        history = previous["history"]
        best_accuracy, best_epoch = previous["best_accuracy"], previous["best_epoch"]
        torch.set_rng_state(previous["torch_rng_state"])
        torch.cuda.set_rng_state_all(previous["cuda_rng_states"])
        train_loader.generator.set_state(previous["train_loader_rng_state"])
        start_epoch = previous["epoch"] + 1
        if migrating and best_accuracy >= 0:
            source_best = torch.load(resume_path.parent / "best_adapter.pt", map_location="cpu", weights_only=False)
            if source_best["identity"] != previous["identity"] or source_best["epoch"] != best_epoch:
                raise ValueError("Source best adapter does not match the handoff checkpoint")
            source_best["identity"] = run_identity
            source_best["continued_from_checkpoint_sha256"] = identity["continued_from_checkpoint_sha256"]
            _atomic_checkpoint(output / "best_adapter.pt", source_best)
            shutil.copyfile(resume_path.parent / "best_validation_logits.pt", output / "best_validation_logits.pt")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_count = _train_epoch(model, train_loader, optimizer, scaler, args, device, epoch=epoch, output=output)
        if args.max_train_batches is None and train_count != len(train_rows):
            raise RuntimeError(f"Expected {len(train_rows)} train samples, got {train_count}")
        evaluate_now = epoch % args.validate_every == 0 or epoch == args.epochs
        metrics = None
        if evaluate_now:
            metrics, logits, labels = _evaluate(model, val_loader, device, max_batches=args.max_val_batches)
            if args.max_val_batches is None and metrics["samples"] != 12960:
                raise RuntimeError(f"Expected 12,960 validation samples, got {metrics['samples']}")
        scheduler.step()
        row = {"epoch": epoch, "train_loss": train_loss, "train_samples": train_count, "validation": metrics, "elapsed_seconds": time.perf_counter() - started}
        history.append(row)
        if metrics is not None and metrics["accuracy"] > best_accuracy:
            best_accuracy, best_epoch = metrics["accuracy"], epoch
            _atomic_checkpoint(output / "best_adapter.pt", {"identity": run_identity, "epoch": epoch, "model": _snapshot(model), "validation": metrics})
            _atomic_checkpoint(output / "best_validation_logits.pt", {"logits": logits, "labels": labels, "sample_ids": [Path(str(item["video_path"])).stem.removesuffix("_rgb") for item in val_rows[:len(labels)]]})
        checkpoint = {
            "identity": run_identity, "epoch": epoch, "model": _snapshot(model),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "history": history,
            "best_accuracy": best_accuracy, "best_epoch": best_epoch,
            "torch_rng_state": torch.get_rng_state(), "cuda_rng_states": torch.cuda.get_rng_state_all(),
            "train_loader_rng_state": train_loader.generator.get_state(),
        }
        _atomic_checkpoint(last_path, checkpoint)
        _atomic_json(output / "progress.json", {"state": "running", "arm": args.arm, "completed_epoch": epoch, "total_epochs": args.epochs, "best_epoch": best_epoch, "best_accuracy": best_accuracy, "latest": row, "test_evaluated": False})
        _log(output / "train.log", {"arm": args.arm, "event": "validation_completed" if metrics is not None else "train_epoch_completed", **row})
    best = torch.load(output / "best_adapter.pt", map_location="cpu", weights_only=False)
    _restore(model, best["model"])
    model.eval()
    pixels, _ = next(iter(val_loader))
    pixels = pixels.to(device)
    with torch.inference_mode():
        reference = model(pixels).float().cpu()
        merge_lora(model)
        merged = model(pixels).float().cpu()
    parity = float((reference - merged).abs().max())
    if not torch.allclose(reference, merged, atol=1e-5, rtol=1e-5):
        raise RuntimeError(f"Merged checkpoint parity failed: {parity}")
    _atomic_checkpoint(output / "best_merged.pt", {"identity": run_identity, "epoch": best_epoch, "state_dict": merged_base_state_dict(model), "merge_max_abs_diff": parity})
    merged_payload = torch.load(output / "best_merged.pt", map_location="cpu", weights_only=False)
    clean = XCLIPVisionClassifier(
        XCLIPModel.from_pretrained(args.model_name, revision=args.model_revision, local_files_only=True),
        num_classes=120, head_dropout=args.head_dropout,
    ).to(device)
    clean.load_state_dict(merged_payload["state_dict"], strict=True)
    clean.eval()
    with torch.inference_mode():
        reloaded_logits = clean(pixels).float().cpu()
    reload_error = float((reloaded_logits - reference).abs().max())
    if not torch.allclose(reference, reloaded_logits, atol=1e-5, rtol=1e-5):
        raise RuntimeError(f"Merged RGB-only reload parity failed: {reload_error}")
    report = {"status": "completed", "arm": args.arm, "initial_logits_sha256": initial_logits_sha256, "best_epoch": best_epoch, "best_validation_accuracy": best_accuracy, "trainable_parameter_audit": audit, "history": history, "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device), "merge_max_abs_diff": parity, "merged_reload_max_abs_diff": reload_error, "test_evaluated": False}
    _atomic_json(output / "run_report.json", report)
    _atomic_json(output / "progress.json", {"state": "completed", "arm": args.arm, "completed_epoch": len(history), "total_epochs": args.epochs, "best_epoch": best_epoch, "best_accuracy": best_accuracy, "test_evaluated": False})
    return report


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Gate 1 requires a visible CUDA GPU")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive")
    if args.rgb_cache_dir and args.uint8_cache_dir:
        raise ValueError("Choose only one RGB cache format")
    if args.resume_from and (not args.uint8_cache_dir or not args.resume):
        raise ValueError("Checkpoint handoff requires --uint8-cache-dir and resume enabled")
    split_path = Path(args.split_manifest)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("frame_sampling_strategy") != FRAME_SAMPLING_STRATEGY or split.get("max_seq_len") != 8:
        raise ValueError("Split sampling contract differs from Gate 0")
    train_rows, val_rows = split["train"], split["val"]
    if len(train_rows) != 80640 or len(val_rows) != 12960:
        raise ValueError("Canonical train/validation counts differ from the locked split")
    if args.pilot_fraction is not None:
        if not 0 < args.pilot_fraction < 1 or not args.init_adapter:
            raise ValueError("Pilot needs a fraction in (0,1) and a CE initialization checkpoint")
        train_rows = _stratified_subset(train_rows, args.pilot_fraction)
    winning_path = Path(args.winning_config)
    winning = json.loads(winning_path.read_text(encoding="utf-8"))
    expected = {"lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha, "lora_dropout": args.lora_dropout, "dropout": args.head_dropout, "feature_noise_std": args.feature_noise_std, "lora_learning_rate": args.mit_lr, "head_learning_rate": args.head_lr, "weight_decay": args.weight_decay, "label_smoothing": args.label_smoothing, "epochs": args.epochs}
    if args.pilot_fraction is None and args.max_train_batches is None:
        if any(winning.get(key) != value for key, value in expected.items()):
            raise ValueError("Gate 1 full run differs from the Phase 2 winning MIT/head recipe")
        if winning.get("batch_size") != args.batch_size and not args.allow_batch_size_change:
            raise ValueError("Batch size differs from the Phase 2 winning recipe; pass --allow-batch-size-change to lock a new experiment")
    subset_ids = [Path(str(row["video_path"])).stem.removesuffix("_rgb") for row in train_rows]
    identity = {"schema_version": 1, "model_name": args.model_name, "model_revision": args.model_revision, "split_sha256": sha256_file(split_path), "winning_config_sha256": sha256_file(winning_path), "init_adapter_sha256": sha256_file(Path(args.init_adapter)) if args.init_adapter else None, "train_subset_sha256": hashlib.sha256("\n".join(subset_ids).encode()).hexdigest(), "train_samples": len(train_rows), "validation_samples": len(val_rows), "source_sha256": sha256_file(Path(__file__)), "continued_from_checkpoint_sha256": sha256_file(Path(args.resume_from)) if args.resume_from else None, "config": {key: value for key, value in vars(args).items() if key not in {"arm", "artifacts_dir", "resume", "resume_from"}}, "test_evaluated": False}
    arms = ("frozen", "vision") if args.arm == "both" else (args.arm,)
    results = {arm: run_arm(argparse.Namespace(**{**vars(args), "arm": arm}), train_rows, val_rows, identity) for arm in arms}
    if len(results) == 2:
        if results["frozen"]["initial_logits_sha256"] != results["vision"]["initial_logits_sha256"]:
            raise RuntimeError("Control arms did not start from identical RGB logits")
        comparison = {"frozen_accuracy": results["frozen"]["best_validation_accuracy"], "vision_accuracy": results["vision"]["best_validation_accuracy"], "delta_percentage_points": 100 * (results["vision"]["best_validation_accuracy"] - results["frozen"]["best_validation_accuracy"]), "test_evaluated": False}
        _atomic_json(Path(args.artifacts_dir) / "ce_comparison.json", comparison)
        return comparison
    return results[arms[0]]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run matched Phase 3 Gate 1 CE controls")
    parser.add_argument("--arm", choices=("frozen", "vision", "both"), default="both")
    parser.add_argument("--model-name", default="microsoft/xclip-base-patch32")
    parser.add_argument("--model-revision", default="a2e27a78a2b5d802e894b8a1ef14f3a8ce490963")
    parser.add_argument("--split-manifest", default="artifacts/xclip/selected_split.json")
    parser.add_argument("--winning-config", default="artifacts/xclip_lora_tuning/winning_config.json")
    parser.add_argument("--init-adapter", default=None)
    parser.add_argument("--pilot-fraction", type=float, default=None)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--artifacts-dir", default="artifacts/skeletonclip_phase3/gate1")
    parser.add_argument("--rgb-cache-dir", default=None, help="Optional lossless processed RGB cache; allow about 4.8 MB per video")
    parser.add_argument("--uint8-cache-dir", default=None, help="Exact processed RGB cache using about 1.2 MB per video")
    parser.add_argument("--resume-from", default=None, help="Handoff a completed epoch checkpoint when switching only to the uint8 RGB cache")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--allow-batch-size-change", action="store_true", help="Explicitly lock a new batch size with a different optimizer-step budget")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upper-blocks", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--head-dropout", type=float, default=0.25)
    parser.add_argument("--feature-noise-std", type=float, default=0.01)
    parser.add_argument("--mit-lr", type=float, default=2e-4)
    parser.add_argument("--vision-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    print(json.dumps(run(parser.parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
