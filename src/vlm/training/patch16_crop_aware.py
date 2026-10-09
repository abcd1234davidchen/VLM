"""Patch16 crop-aware training: the patch16 pilot recipe with per-sample whole/center-zoom70 views.

Every training sample sees either the whole view or the center_zoom70 view, chosen by a
deterministic (seed, epoch, sample_id) hash; labels are unchanged. Every epoch reports whole,
center_zoom70 and their probability mean on the full validation split. Test data is never
loaded. Contract and pre-registered decision rules: docs/patch16-crop-aware-plan.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, Sampler
from transformers import XCLIPModel, XCLIPProcessor

import vlm
from vlm.models.lora import merge_lora, merged_base_state_dict
from vlm.models.xclip_vision_lora import XCLIPVisionClassifier
from vlm.processed_uint8_cache import ProcessedUInt8Cache
from vlm.skeleton_teacher_cache import sha256_file
from vlm.training.engine import set_seed
from vlm.training.skeletonclip_phase3_gate1 import (
    _atomic_checkpoint, _atomic_json, _loader, _log, _model, _optimizer, _restore, _seed_worker, _snapshot, _train_epoch,
)
from vlm.xclip_features import FRAME_SAMPLING_STRATEGY, read_uniform_video_frames

VIEWS = ("whole", "center_zoom70")
ATTENTION = torch.arange(41, 48)  # A042–A048, zero-based labels
PILOT_LOCK = Path("artifacts/patch16_resolution/pilot_seed42/vision_lora_ce_seed42/run_lock.json")
PILOT_VIEWS = Path("artifacts/patch16_resolution/whole_zoom_validation/view_predictions.pt")
PILOT_INITIAL_LOGITS_SHA256 = "786a79b887085ef2de8a69a6bf32e27e24a32d9ce292c294ad6df18e78c0667b"
# Pre-registered rules (docs/patch16-crop-aware-plan.md section 5).
EARLY_EPOCH, EARLY_MIN_WHOLE = 5, 0.7068 - 0.01
PROMISING_MARGIN = 0.002


def center_zoom70(frames: np.ndarray) -> np.ndarray:
    """Same crop as rgb_input_audit.ViewDataset: central square of 70% of the short edge."""
    h, w = frames.shape[1:3]
    side = round(min(h, w) * .7)
    top, left = (h - side) // 2, (w - side) // 2
    return frames[:, top:top + side, left:left + side]


class CenterZoomUInt8Cache(ProcessedUInt8Cache):
    """Fill-on-miss uint8 cache of center_zoom70 crops with an identity separate from whole.

    The base module is left untouched because locked runs hash its source; `load` mirrors
    the base implementation with the crop applied between decoding and the processor.
    """

    def __init__(self, root: Path, processor: object, *, model_name: str, model_revision: str) -> None:
        super().__init__(root, processor, model_name=model_name, model_revision=model_revision)
        config = {"whole_config_sha256": self.config_hash, "view": "center_zoom70",
                  "crop": "central square, side round(0.7 * short edge), offsets floor((dim - side) / 2), applied before the processor"}
        self.config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()

    def load(self, video_path: Path) -> torch.Tensor:
        path = self.path_for(video_path)
        compact = None
        if path.exists():
            try:
                compact = np.load(path, mmap_mode="r", allow_pickle=False)
                if compact.dtype != np.uint8 or compact.shape != (8, 3, 224, 224):
                    compact = None
            except (OSError, ValueError, EOFError):
                compact = None
        if compact is None:
            frames = center_zoom70(read_uniform_video_frames(video_path, 8))
            compact = self.processor(images=list(frames), return_tensors="np", do_rescale=False, do_normalize=False)["pixel_values"][0]
            if compact.dtype != np.uint8 or compact.shape != (8, 3, 224, 224):
                raise ValueError(f"Unexpected uint8 crop format for {video_path}: {compact.shape}, {compact.dtype}")
            if self.path_for(video_path) != path:
                raise RuntimeError(f"Source video changed during extraction: {video_path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.stem}.{os.getpid()}.{uuid.uuid4().hex}.npy")
            try:
                with temporary.open("wb") as stream:
                    np.save(stream, compact, allow_pickle=False)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        pixels = self.processor(
            images=list(compact.transpose(0, 2, 3, 1)),
            return_tensors="pt", do_resize=False, do_center_crop=False,
        )["pixel_values"].squeeze(0)
        if not isinstance(pixels, torch.Tensor) or pixels.dtype != torch.float32 or pixels.shape != (8, 3, 224, 224):
            raise ValueError(f"Unexpected reconstructed RGB pixels for {video_path}")
        return pixels


def sample_id(row: dict[str, object]) -> str:
    return Path(str(row["video_path"])).stem.removesuffix("_rgb")


def view_for(seed: int, epoch: int, identifier: str) -> str:
    digest = hashlib.sha256(f"patch16-crop-aware:{seed}:{epoch}:{identifier}".encode()).digest()
    return VIEWS[digest[0] & 1]


class EpochSampler(Sampler):
    """The pilot's shuffle order, with each index tagged by the current epoch.

    Wrapping RandomSampler around the DataLoader's own generator reproduces the generator
    draws of DataLoader(shuffle=True), so only the per-sample view differs from the pilot.
    """

    def __init__(self, size: int, generator: torch.Generator) -> None:
        self.order = RandomSampler(range(size), generator=generator)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.order)

    def __iter__(self):
        epoch = self.epoch
        return ((index, epoch) for index in self.order)


class MixedViewDataset(Dataset):
    def __init__(self, rows: list[dict[str, object]], caches: dict[str, ProcessedUInt8Cache], seed: int) -> None:
        self.rows, self.caches, self.seed = rows, caches, seed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, key: tuple[int, int]) -> tuple[torch.Tensor, int]:
        index, epoch = key
        row = self.rows[index]
        view = view_for(self.seed, epoch, sample_id(row))
        return self.caches[view].load(Path(str(row["video_path"]))), int(row["class_id"]) - 1


class WholeZoomDataset(Dataset):
    def __init__(self, rows: list[dict[str, object]], caches: dict[str, ProcessedUInt8Cache]) -> None:
        self.rows, self.caches = rows, caches

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        path = Path(str(row["video_path"]))
        return torch.stack([self.caches[view].load(path) for view in VIEWS]), int(row["class_id"]) - 1


def train_loader(dataset: MixedViewDataset, args: argparse.Namespace) -> DataLoader:
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(
        dataset, batch_size=args.batch_size, sampler=EpochSampler(len(dataset), generator),
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
        worker_init_fn=_seed_worker, generator=generator,
    )


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, *, max_batches: int | None) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    model.eval()
    outputs: dict[str, list[torch.Tensor]] = {view: [] for view in VIEWS}
    labels_rows: list[torch.Tensor] = []
    with torch.inference_mode():
        for index, (pixels, labels) in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            pixels = pixels.to(device, non_blocking=True)
            for position, view in enumerate(VIEWS):
                outputs[view].append(model(pixels[:, position]).float().cpu())
            labels_rows.append(labels)
    return {view: torch.cat(parts) for view, parts in outputs.items()}, torch.cat(labels_rows)


def view_metrics(logits: dict[str, torch.Tensor], labels: torch.Tensor) -> dict[str, dict[str, float | int]]:
    probabilities = {view: logits[view].softmax(-1) for view in VIEWS}
    probabilities["whole_zoom_mean"] = (probabilities["whole"] + probabilities["center_zoom70"]) / 2
    attention = torch.isin(labels, ATTENTION)
    result = {}
    for name, probability in probabilities.items():
        if not torch.isfinite(probability).all():
            raise RuntimeError(f"Non-finite validation probabilities for {name}")
        predicted = probability.argmax(-1)
        ok = predicted.eq(labels)
        confusion = torch.bincount(labels * 120 + predicted, minlength=120 * 120).reshape(120, 120)
        tp = confusion.diag().float()
        denominator = confusion.sum(0).float() + confusion.sum(1).float()
        result[name] = {
            "accuracy": float(ok.float().mean()), "correct": int(ok.sum()), "samples": int(labels.numel()),
            "macro_f1": float(torch.where(denominator > 0, 2 * tp / denominator.clamp_min(1), 0).mean()),
            "nll": float(-probability[torch.arange(len(labels)), labels].clamp_min(1e-38).log().mean()),
            "attention_correct": int(ok[attention].sum()), "attention_samples": int(attention.sum()),
        }
    return result


def pilot_comparison(logits: dict[str, torch.Tensor], labels: torch.Tensor, ids: list[str], metrics: dict[str, dict]) -> dict[str, object]:
    pilot = torch.load(PILOT_VIEWS, map_location="cpu", weights_only=True)
    if pilot["sample_ids"] != ids or not torch.equal(pilot["labels"], labels) or pilot["test_evaluated"] is not False:
        raise ValueError("Pilot whole/zoom predictions do not match the validation order")
    reference = view_metrics(pilot["logits"], labels)
    pilot_probabilities = {view: pilot["logits"][view].softmax(-1) for view in VIEWS}
    pilot_probabilities["whole_zoom_mean"] = (pilot_probabilities["whole"] + pilot_probabilities["center_zoom70"]) / 2
    probabilities = {view: logits[view].softmax(-1) for view in VIEWS}
    probabilities["whole_zoom_mean"] = (probabilities["whole"] + probabilities["center_zoom70"]) / 2
    views = {}
    for name, probability in probabilities.items():
        ok, old = probability.argmax(-1).eq(labels), pilot_probabilities[name].argmax(-1).eq(labels)
        views[name] = {"pilot": reference[name], "crop_aware": metrics[name],
                       "delta_percentage_points": 100 * (metrics[name]["accuracy"] - reference[name]["accuracy"]),
                       "fixed": int((ok & ~old).sum()), "regressed": int((~ok & old).sum())}
    ensemble, pilot_ensemble = metrics["whole_zoom_mean"], reference["whole_zoom_mean"]
    if ensemble["accuracy"] >= pilot_ensemble["accuracy"] + PROMISING_MARGIN and ensemble["attention_correct"] >= pilot_ensemble["attention_correct"]:
        decision = "promising"
    elif ensemble["accuracy"] < pilot_ensemble["accuracy"]:
        decision = "no_go"
    else:
        decision = "inconclusive"
    return {"decision": decision, "views": views,
            "single_pass_whole_vs_pilot": {"crop_aware_whole": metrics["whole"]["accuracy"],
                                           "pilot_zoom70": reference["center_zoom70"]["accuracy"],
                                           "pilot_whole_zoom_mean": pilot_ensemble["accuracy"]},
            "rule": "promising: whole_zoom_mean >= pilot + 0.20 points and A042-A048 correct >= pilot; no_go: whole_zoom_mean below pilot; else inconclusive"}


def run(cli: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Crop-aware training requires a visible CUDA GPU")
    pilot_lock = json.loads(PILOT_LOCK.read_text(encoding="utf-8"))
    # The recipe comes verbatim from the pilot lock; only operational settings are taken from the CLI.
    args = argparse.Namespace(**{**pilot_lock["config"], "arm": "vision",
                                 "workers": cli.workers, "max_train_batches": cli.max_train_batches,
                                 "max_val_batches": cli.max_val_batches})
    if not 1 <= cli.stop_after_epoch <= args.epochs:
        raise ValueError("--stop-after-epoch must be within [1, epochs]")
    split_path, winning_path = Path(args.split_manifest), Path(args.winning_config)
    if sha256_file(split_path) != pilot_lock["split_sha256"] or sha256_file(winning_path) != pilot_lock["winning_config_sha256"]:
        raise ValueError("Split or winning config differs from the patch16 pilot")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("frame_sampling_strategy") != FRAME_SAMPLING_STRATEGY or split.get("max_seq_len") != 8:
        raise ValueError("Split sampling contract differs from the pilot")
    train_rows, val_rows = split["train"], split["val"]
    if len(train_rows) != 80640 or len(val_rows) != 12960:
        raise ValueError("Canonical train/validation counts differ from the locked split")
    output = Path(cli.artifacts_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    set_seed(args.seed)
    processor = XCLIPProcessor.from_pretrained(args.model_name, revision=args.model_revision, local_files_only=True)
    cache_processor = XCLIPProcessor.from_pretrained(args.cache_model_name, revision=args.cache_model_revision, local_files_only=True)
    if cache_processor.image_processor.to_dict() != processor.image_processor.to_dict():
        raise ValueError("Cache model preprocessing differs from the trained model")
    caches = {
        "whole": ProcessedUInt8Cache(Path(args.uint8_cache_dir), processor, model_name=args.cache_model_name, model_revision=args.cache_model_revision),
        "center_zoom70": CenterZoomUInt8Cache(Path(cli.zoom_cache_dir), processor, model_name=args.cache_model_name, model_revision=args.cache_model_revision),
    }
    if caches["whole"].config_hash == caches["center_zoom70"].config_hash:
        raise RuntimeError("Whole and zoom caches share an identity")
    train_data = train_loader(MixedViewDataset(train_rows, caches, args.seed), args)
    val_data = _loader(WholeZoomDataset(val_rows, caches), args, train=False)
    model = _model(args, device)
    model.eval()
    initial_pixels, _ = next(iter(val_data))
    with torch.inference_mode():
        initial_logits = model(initial_pixels[:, 0].to(device)).float().cpu()
    initial_logits_sha256 = hashlib.sha256(initial_logits.numpy().tobytes()).hexdigest()
    del initial_pixels, initial_logits
    audit = model.trainable_parameter_audit().as_dict()
    if not any(name.startswith("vision_model.") for name in audit["names"]):
        raise RuntimeError("Vision LoRA parameters are not trainable")
    optimizer = _optimizer(args, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    source_files = [Path(__file__), Path(__file__).with_name("skeletonclip_phase3_gate1.py"),
                    Path(vlm.__file__).parent / "processed_uint8_cache.py",
                    Path(vlm.__file__).parent / "models/xclip_vision_lora.py", Path(vlm.__file__).parent / "models/lora.py"]
    identity = {
        "schema_version": 1, "experiment": "patch16_crop_aware_v1",
        "pilot_run_lock_sha256": sha256_file(PILOT_LOCK), "recipe": pilot_lock["config"],
        "split_sha256": pilot_lock["split_sha256"], "winning_config_sha256": pilot_lock["winning_config_sha256"],
        "amp": {"enabled": args.amp, "dtype": "float16" if args.amp else "float32"},
        "train_view_policy": "per (seed, epoch, sample_id) sha256 bit: whole or center_zoom70, about 50/50; labels unchanged",
        "validation_views": [*VIEWS, "whole_zoom_mean"],
        "cache_config_sha256": {view: cache.config_hash for view, cache in caches.items()},
        "zoom_cache_dir": str(cli.zoom_cache_dir),
        "source_sha256": {str(path.relative_to(Path(vlm.__file__).parents[1])): sha256_file(path) for path in source_files},
        "bounded_smoke": {"max_train_batches": args.max_train_batches, "max_val_batches": args.max_val_batches},
        "test_evaluated": False,
    }
    _atomic_json(output / "run_lock.json", {**identity, "vlm_package_path": str(Path(vlm.__file__).parent),
                                            "initial_logits_sha256": initial_logits_sha256,
                                            "initial_logits_match_pilot": initial_logits_sha256 == PILOT_INITIAL_LOGITS_SHA256})
    last_path = output / "last_checkpoint.pt"
    history: list[dict[str, object]] = []
    start_epoch = 1
    if cli.resume and last_path.exists():
        previous = torch.load(last_path, map_location="cpu", weights_only=False)
        if previous["identity"] != identity:
            raise ValueError("Resume identity differs from the locked run")
        _restore(model, previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        scheduler.load_state_dict(previous["scheduler"])
        scaler.load_state_dict(previous["scaler"])
        history = previous["history"]
        torch.set_rng_state(previous["torch_rng_state"])
        torch.cuda.set_rng_state_all(previous["cuda_rng_states"])
        train_data.generator.set_state(previous["train_loader_rng_state"])
        start_epoch = previous["epoch"] + 1
        if previous.get("status") == "early_stopped":
            raise ValueError("Run already stopped at the pre-registered early check")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    train_ids = [sample_id(row) for row in train_rows]
    status = "completed"
    for epoch in range(start_epoch, cli.stop_after_epoch + 1):
        train_data.sampler.epoch = epoch
        zoom_count = sum(view_for(args.seed, epoch, identifier) == "center_zoom70" for identifier in train_ids)
        _log(output / "train.log", {"event": "view_assignment", "epoch": epoch, "whole": len(train_ids) - zoom_count, "center_zoom70": zoom_count})
        train_loss, train_count = _train_epoch(model, train_data, optimizer, scaler, args, device, epoch=epoch, output=output, class_weights=None)
        if args.max_train_batches is None and train_count != len(train_rows):
            raise RuntimeError(f"Expected {len(train_rows)} train samples, got {train_count}")
        _log(output / "train.log", {"event": "validation_started", "epoch": epoch})
        logits, labels = evaluate(model, val_data, device, max_batches=args.max_val_batches)
        if args.max_val_batches is None and len(labels) != 12960:
            raise RuntimeError(f"Expected 12,960 validation samples, got {len(labels)}")
        metrics = view_metrics(logits, labels)
        scheduler.step()
        zoom_files = sum(1 for _ in (Path(cli.zoom_cache_dir) / caches["center_zoom70"].config_hash).rglob("*.npy"))
        row = {"epoch": epoch, "train_loss": train_loss, "train_samples": train_count, "train_zoom_assignments": zoom_count,
               "validation": metrics, "zoom_cache_files": zoom_files, "elapsed_seconds": time.perf_counter() - started}
        history.append(row)
        if epoch == EARLY_EPOCH and args.max_val_batches is None and metrics["whole"]["accuracy"] < EARLY_MIN_WHOLE:
            status = "early_stopped"
        _atomic_checkpoint(last_path, {
            "identity": identity, "epoch": epoch, "status": status, "model": _snapshot(model),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "history": history, "torch_rng_state": torch.get_rng_state(), "cuda_rng_states": torch.cuda.get_rng_state_all(),
            "train_loader_rng_state": train_data.generator.get_state(),
        })
        _atomic_json(output / "progress.json", {"state": "running", "completed_epoch": epoch, "stop_after_epoch": cli.stop_after_epoch,
                                                "total_epochs": args.epochs, "latest": row, "test_evaluated": False})
        _log(output / "train.log", {"event": "validation_completed", **row})
        if status == "early_stopped":
            _log(output / "train.log", {"event": "early_stopped", "epoch": epoch, "whole_accuracy": metrics["whole"]["accuracy"], "minimum": EARLY_MIN_WHOLE})
            break
    if not history:
        raise RuntimeError("No epoch ran; the run is already at --stop-after-epoch")
    final_epoch = history[-1]["epoch"]
    if start_epoch > final_epoch:
        raise RuntimeError("Resumed run has no epoch left before --stop-after-epoch")
    ids = [sample_id(row) for row in val_rows[:len(labels)]]
    _atomic_checkpoint(output / "final_adapter.pt", {"identity": identity, "epoch": final_epoch, "model": _snapshot(model), "validation": history[-1]["validation"]})
    _atomic_checkpoint(output / "final_view_predictions.pt", {"logits": logits, "labels": labels, "sample_ids": ids, "epoch": final_epoch, "training": False, "test_evaluated": False})
    model.eval()
    pixels, _ = next(iter(val_data))
    pixels = pixels[:, 0].to(device)
    with torch.inference_mode():
        reference = model(pixels).float().cpu()
        merge_lora(model)
        merged = model(pixels).float().cpu()
    parity = float((reference - merged).abs().max())
    if not torch.allclose(reference, merged, atol=1e-5, rtol=1e-5):
        raise RuntimeError(f"Merged checkpoint parity failed: {parity}")
    _atomic_checkpoint(output / "final_merged.pt", {"identity": identity, "epoch": final_epoch, "state_dict": merged_base_state_dict(model), "merge_max_abs_diff": parity})
    clean = XCLIPVisionClassifier(
        XCLIPModel.from_pretrained(args.model_name, revision=args.model_revision, local_files_only=True),
        num_classes=args.num_classes, head_dropout=args.head_dropout,
    ).to(device)
    clean.load_state_dict(torch.load(output / "final_merged.pt", map_location="cpu", weights_only=False)["state_dict"], strict=True)
    clean.eval()
    with torch.inference_mode():
        reloaded = clean(pixels).float().cpu()
    reload_error = float((reloaded - reference).abs().max())
    if not torch.allclose(reference, reloaded, atol=1e-5, rtol=1e-5):
        raise RuntimeError(f"Merged RGB-only reload parity failed: {reload_error}")
    comparison = pilot_comparison(logits, labels, ids, history[-1]["validation"]) if args.max_val_batches is None else None
    report = {"status": status, "final_epoch": final_epoch, "stop_after_epoch": cli.stop_after_epoch,
              "decision": "no_go" if status == "early_stopped" else comparison["decision"] if comparison else None,
              "pilot_comparison": comparison, "initial_logits_sha256": initial_logits_sha256,
              "initial_logits_match_pilot": initial_logits_sha256 == PILOT_INITIAL_LOGITS_SHA256,
              "trainable_parameter_audit": audit, "history": history,
              "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
              "merge_max_abs_diff": parity, "merged_reload_max_abs_diff": reload_error, "test_evaluated": False}
    _atomic_json(output / "run_report.json", report)
    _atomic_json(output / "progress.json", {"state": status, "completed_epoch": final_epoch, "decision": report["decision"], "test_evaluated": False})
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", default="artifacts/patch16_crop_aware/seed42")
    parser.add_argument("--zoom-cache-dir", default="cache/patch16_crop_aware_zoom70_uint8")
    parser.add_argument("--stop-after-epoch", type=int, default=20, help="Stop early while keeping the pilot's 60-epoch cosine schedule")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-batches", type=int, default=None, help="Bounded smoke only")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Bounded smoke only")
    report = run(parser.parse_args(argv))
    print(json.dumps({key: value for key, value in report.items() if key not in {"history", "trainable_parameter_audit"}}, indent=2))


if __name__ == "__main__":
    main()
