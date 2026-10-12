"""Patch16 person crop training: the crop-aware recipe with person crops in place of center_zoom70.

Every training sample sees either the whole view or its RGB person crop, chosen by the same
(seed, epoch, sample_id) hash bit as patch16_crop_aware, so each sample gets the crop view in
exactly the epochs it got center_zoom70 there. Every epoch reports whole, person_crop and their
probability mean on the full validation split. Test data is never loaded. Contract and
pre-registered decision rules: docs/person-aware-crop-stage1-plan.md.
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
from torch.utils.data import Dataset
from transformers import XCLIPModel, XCLIPProcessor

import vlm
from vlm.models.lora import merge_lora, merged_base_state_dict
from vlm.models.xclip_vision_lora import XCLIPVisionClassifier
from vlm.processed_uint8_cache import ProcessedUInt8Cache
from vlm.skeleton_teacher_cache import sha256_file
from vlm.training.engine import set_seed
from vlm.training.patch16_crop_aware import PILOT_INITIAL_LOGITS_SHA256, PILOT_LOCK, sample_id, train_loader, view_for
from vlm.training.skeletonclip_phase3_gate1 import (
    _atomic_checkpoint, _atomic_json, _loader, _log, _model, _optimizer, _restore, _snapshot, _train_epoch,
)
from vlm.xclip_features import FRAME_SAMPLING_STRATEGY, read_uniform_video_frames

VIEWS = ("whole", "person_crop")
ATTENTION = torch.arange(41, 48)  # A042–A048, zero-based labels
STAGE0_VIEWS = Path("artifacts/person_crop_stage0/evaluation/view_predictions.pt")
# Pre-registered rules (docs/person-aware-crop-stage1-plan.md section 6).
EARLY_EPOCH, EARLY_MIN_PERSON = 5, 0.7913 - 0.01
BASELINE_CORRECT, BASELINE_ATTENTION, PROMISING_CORRECT = 11077, 413, 11103


def manifest_crops(path: Path) -> dict[str, list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    lock = path.with_name("experiment_lock.json")
    if payload["lock_sha256"] != sha256_file(lock):
        raise ValueError(f"Crop manifest does not match its localization lock: {path}")
    return {record["sample_id"]: record["crop"] for record in payload["samples"]}


class PersonCropUInt8Cache(ProcessedUInt8Cache):
    """Fill-on-miss uint8 cache of person crops, keyed by the crop manifests it was built from."""

    def __init__(self, root: Path, processor: object, crops: dict[str, list[int]], manifest_sha256: list[str],
                 *, model_name: str, model_revision: str) -> None:
        super().__init__(root, processor, model_name=model_name, model_revision=model_revision)
        self.crops = crops
        config = {"whole_config_sha256": self.config_hash, "view": "person_crop", "manifest_sha256": sorted(manifest_sha256),
                  "crop": "manifest [left, top, right, bottom] square applied to the uniform 8 frames before the processor"}
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
            left, top, right, bottom = self.crops[video_path.stem.removesuffix("_rgb")]
            frames = read_uniform_video_frames(video_path, 8)[:, top:bottom, left:right]
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


class MixedViewDataset(Dataset):
    def __init__(self, rows: list[dict[str, object]], caches: dict[str, ProcessedUInt8Cache], seed: int) -> None:
        self.rows, self.caches, self.seed = rows, caches, seed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, key: tuple[int, int]) -> tuple[torch.Tensor, int]:
        index, epoch = key
        row = self.rows[index]
        view = "whole" if view_for(self.seed, epoch, sample_id(row)) == "whole" else "person_crop"
        return self.caches[view].load(Path(str(row["video_path"]))), int(row["class_id"]) - 1


class WholePersonDataset(Dataset):
    def __init__(self, rows: list[dict[str, object]], caches: dict[str, ProcessedUInt8Cache]) -> None:
        self.rows, self.caches = rows, caches

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        path = Path(str(row["video_path"]))
        return torch.stack([self.caches[view].load(path) for view in VIEWS]), int(row["class_id"]) - 1


def evaluate(model: torch.nn.Module, loader, device: torch.device, *, max_batches: int | None) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
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


def probabilities(logits: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = {view: logits[view].float().softmax(-1) for view in VIEWS}
    result["whole_person_mean"] = (result["whole"] + result["person_crop"]) / 2
    return result


def view_metrics(logits: dict[str, torch.Tensor], labels: torch.Tensor) -> dict[str, dict[str, object]]:
    attention = torch.isin(labels, ATTENTION)
    result = {}
    for name, probability in probabilities(logits).items():
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
            "events": {f"A{c + 1:03d}": int(ok[labels.eq(c)].sum()) for c in ATTENTION.tolist()},
        }
    return result


def stage0_comparison(logits: dict[str, torch.Tensor], labels: torch.Tensor, ids: list[str], metrics: dict[str, dict]) -> dict[str, object]:
    stage0 = torch.load(STAGE0_VIEWS, map_location="cpu", weights_only=False)
    if stage0["sample_ids"] != ids or not torch.equal(stage0["labels"], labels) or stage0["test_evaluated"] is not False:
        raise ValueError("Stage 0 predictions do not match the validation order")
    reference = view_metrics(stage0["logits"], labels)
    old_probabilities, new_probabilities = probabilities(stage0["logits"]), probabilities(logits)
    views = {}
    for name in new_probabilities:
        ok, old = new_probabilities[name].argmax(-1).eq(labels), old_probabilities[name].argmax(-1).eq(labels)
        views[name] = {"crop_aware_stage0": reference[name], "person_trained": metrics[name],
                       "delta_percentage_points": 100 * (metrics[name]["accuracy"] - reference[name]["accuracy"]),
                       "fixed": int((ok & ~old).sum()), "regressed": int((~ok & old).sum())}
    person = metrics["person_crop"]
    if person["correct"] >= PROMISING_CORRECT and person["attention_correct"] >= BASELINE_ATTENTION:
        decision = "promising"
    elif person["correct"] < BASELINE_CORRECT:
        decision = "no_go"
    else:
        decision = "inconclusive"
    return {"decision": decision, "views": views,
            "rule": "promising: person_crop correct >= 11,103 and A042-A048 >= 413; no_go: person_crop correct < 11,077; else inconclusive"}


def run(cli: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Person crop training requires a visible CUDA GPU")
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
    crops = {**manifest_crops(cli.train_crops), **manifest_crops(cli.val_crops)}
    missing = {sample_id(row) for row in train_rows + val_rows} - crops.keys()
    if missing:
        raise ValueError(f"{len(missing)} samples have no person crop, e.g. {sorted(missing)[:3]}")
    manifest_hashes = [sha256_file(cli.train_crops), sha256_file(cli.val_crops)]
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
        "person_crop": PersonCropUInt8Cache(Path(cli.person_cache_dir), processor, crops, manifest_hashes,
                                            model_name=args.cache_model_name, model_revision=args.cache_model_revision),
    }
    train_data = train_loader(MixedViewDataset(train_rows, caches, args.seed), args)
    val_data = _loader(WholePersonDataset(val_rows, caches), args, train=False)
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
    source_files = [Path(__file__), Path(__file__).with_name("patch16_crop_aware.py"), Path(__file__).with_name("skeletonclip_phase3_gate1.py"),
                    Path(vlm.__file__).parent / "processed_uint8_cache.py",
                    Path(vlm.__file__).parent / "models/xclip_vision_lora.py", Path(vlm.__file__).parent / "models/lora.py"]
    identity = {
        "schema_version": 1, "experiment": "patch16_person_crop_v1",
        "pilot_run_lock_sha256": sha256_file(PILOT_LOCK), "recipe": pilot_lock["config"],
        "split_sha256": pilot_lock["split_sha256"], "winning_config_sha256": pilot_lock["winning_config_sha256"],
        "crop_manifest_sha256": {"train": manifest_hashes[0], "val": manifest_hashes[1]},
        "amp": {"enabled": args.amp, "dtype": "float16" if args.amp else "float32"},
        "train_view_policy": "patch16_crop_aware view bit per (seed, epoch, sample_id): whole or person_crop (in place of center_zoom70); labels unchanged",
        "validation_views": [*VIEWS, "whole_person_mean"],
        "cache_config_sha256": {view: cache.config_hash for view, cache in caches.items()},
        "person_cache_dir": str(cli.person_cache_dir),
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
        crop_count = sum(view_for(args.seed, epoch, identifier) != "whole" for identifier in train_ids)
        _log(output / "train.log", {"event": "view_assignment", "epoch": epoch, "whole": len(train_ids) - crop_count, "person_crop": crop_count})
        train_loss, train_count = _train_epoch(model, train_data, optimizer, scaler, args, device, epoch=epoch, output=output, class_weights=None)
        if args.max_train_batches is None and train_count != len(train_rows):
            raise RuntimeError(f"Expected {len(train_rows)} train samples, got {train_count}")
        _log(output / "train.log", {"event": "validation_started", "epoch": epoch})
        logits, labels = evaluate(model, val_data, device, max_batches=args.max_val_batches)
        if args.max_val_batches is None and len(labels) != 12960:
            raise RuntimeError(f"Expected 12,960 validation samples, got {len(labels)}")
        metrics = view_metrics(logits, labels)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": train_loss, "train_samples": train_count, "train_person_crop_assignments": crop_count,
               "validation": metrics, "elapsed_seconds": time.perf_counter() - started}
        history.append(row)
        if epoch == EARLY_EPOCH and args.max_val_batches is None and metrics["person_crop"]["accuracy"] < EARLY_MIN_PERSON:
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
            _log(output / "train.log", {"event": "early_stopped", "epoch": epoch, "person_crop_accuracy": metrics["person_crop"]["accuracy"], "minimum": EARLY_MIN_PERSON})
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
    pixels = pixels[:, 1].to(device)
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
    comparison = stage0_comparison(logits, labels, ids, history[-1]["validation"]) if args.max_val_batches is None else None
    report = {"status": status, "final_epoch": final_epoch, "stop_after_epoch": cli.stop_after_epoch,
              "decision": "no_go" if status == "early_stopped" else comparison["decision"] if comparison else None,
              "stage0_comparison": comparison, "initial_logits_sha256": initial_logits_sha256,
              "initial_logits_match_pilot": initial_logits_sha256 == PILOT_INITIAL_LOGITS_SHA256,
              "trainable_parameter_audit": audit, "history": history,
              "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
              "merge_max_abs_diff": parity, "merged_reload_max_abs_diff": reload_error, "test_evaluated": False}
    _atomic_json(output / "run_report.json", report)
    _atomic_json(output / "progress.json", {"state": status, "completed_epoch": final_epoch, "decision": report["decision"], "test_evaluated": False})
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", default="artifacts/patch16_person_crop/seed42")
    parser.add_argument("--train-crops", type=Path, default=Path("artifacts/person_crop_stage1/localization_train/crop_manifest.json"))
    parser.add_argument("--val-crops", type=Path, default=Path("artifacts/person_crop_stage0/localization/crop_manifest.json"))
    parser.add_argument("--person-cache-dir", default="cache/patch16_person_crop_uint8")
    parser.add_argument("--stop-after-epoch", type=int, default=20, help="Stop early while keeping the pilot's 60-epoch cosine schedule")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-batches", type=int, default=None, help="Bounded smoke only")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Bounded smoke only")
    report = run(parser.parse_args(argv))
    print(json.dumps({key: value for key, value in report.items() if key not in {"history", "trainable_parameter_audit"}}, indent=2))


if __name__ == "__main__":
    main()
