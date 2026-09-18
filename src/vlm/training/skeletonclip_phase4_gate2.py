"""Phase 4 Gate 2 experiment lock and fail-closed GPU preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import av
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from transformers import XCLIPModel, XCLIPProcessor

from vlm.experiments.video_sampling import uniform_frame_indices
from vlm.models.lora import load_lora_state_dict, lora_state_dict
from vlm.models.xclip_vision_lora import XCLIPSpatialLoRAClassifier
from vlm.processed_uint8_cache import ProcessedUInt8Cache
from vlm.reporting import write_json
from vlm.skeleton_teacher_cache import sha256_file
from vlm.spatial_joint_targets import build_spatial_targets, spatial_region_loss
from vlm.training.engine import set_seed
from vlm.xclip_features import FRAME_SAMPLING_STRATEGY


MODEL_REVISION = "a2e27a78a2b5d802e894b8a1ef14f3a8ce490963"
ARMS = ("ce", "joint", "shuffled_joint")


def _sample_id(video_path: str) -> str:
    return Path(video_path).stem.removesuffix("_rgb")


def make_lock(args: argparse.Namespace) -> dict[str, object]:
    """Verify source identity without opening any test sample or video."""
    gate0 = json.loads(args.gate0_decision.read_text())
    gate1 = json.loads(args.gate1_report.read_text())
    if gate0.get("status") != "pass" or gate0.get("test_evaluated") is not False:
        raise ValueError("Phase 4 Gate 0 did not pass with the test embargo intact")
    if gate1.get("status") != "pass" or gate1.get("test_evaluated") is not False:
        raise ValueError("Phase 4 Gate 1 did not pass with the test embargo intact")
    split = json.loads(args.split_manifest.read_text())
    paired = json.loads(args.paired_manifest.read_text())
    if split.get("frame_sampling_strategy") != FRAME_SAMPLING_STRATEGY or split.get("max_seq_len") != 8:
        raise ValueError("RGB frame sampling differs from Gate 0")
    if (len(split["train"]), len(split["val"])) != (80640, 12960):
        raise ValueError("Canonical RGB split counts changed")
    if (len(paired["train"]), len(paired["val"])) != (45236, 4180):
        raise ValueError("Canonical paired split counts changed")
    for part, paired_part in (("train", "train"), ("val", "val")):
        rgb_ids = {_sample_id(row["video_path"]) for row in split[part]}
        paired_ids = {row["sample_id"] for row in paired[paired_part]}
        if len(rgb_ids) != len(split[part]) or len(paired_ids) != len(paired[paired_part]) or not paired_ids <= rgb_ids:
            raise ValueError(f"Paired {part} sample IDs do not match the RGB split")
    if args.epochs < 1 or args.batch_size < 1 or args.spatial_weight <= 0:
        raise ValueError("Epochs, batch size and joint-arm spatial weight must be positive")
    checkpoint = torch.load(args.init_adapter, map_location="cpu", weights_only=False)
    identity = checkpoint.get("identity", {})
    if identity.get("model_revision") != MODEL_REVISION or identity.get("arm") != "vision":
        raise ValueError("Warm-start must be the Phase 3 vision adapter at the Gate 1 model revision")
    if identity.get("split_sha256") != sha256_file(args.split_manifest) or identity.get("test_evaluated") is not False:
        raise ValueError("Warm-start split or test embargo differs")
    if checkpoint.get("model", {}).get("target_modules") is None:
        raise ValueError("Warm-start adapter lacks LoRA target provenance")
    return {
        "schema_version": 1, "gate": "phase4_gate2", "status": "locked_pending_gpu_preflight",
        "model_name": "microsoft/xclip-base-patch32", "model_revision": MODEL_REVISION,
        "seed": args.seed, "arms": ARMS, "epochs": args.epochs, "batch_size": args.batch_size,
        "workers": args.workers, "max_train_batches": args.max_train_batches, "max_val_batches": args.max_val_batches,
        "spatial_weight": args.spatial_weight, "initialization": "shared Phase 3 best vision adapter; shared newly seeded spatial probe",
        "source_sha256": {"rgb_split": sha256_file(args.split_manifest), "paired_manifest": sha256_file(args.paired_manifest),
                          "gate0_decision": sha256_file(args.gate0_decision), "gate1_report": sha256_file(args.gate1_report),
                          "init_adapter": sha256_file(args.init_adapter), "runner": sha256_file(Path(__file__))},
        "train_samples": 80640, "validation_samples": 12960,
        "paired_train_samples": 45236, "paired_validation_samples": 4180,
        "target": "Gate 1 tracked-state-2 top-two-body patch32 occupancy; invalid frames masked",
        "loss": "label-smoothed 120-class CE plus weight times Gate 1 balanced patch BCE on valid paired frames",
        "shuffle": "deterministic seed permutation of paired train sample targets only; validation targets remain unshuffled",
        "selection": "fixed final epoch for matched budget; report best validation classification epoch separately; no test",
        "localization": "held-out paired validation valid-frame patch average precision and occupied-patch recall at top k, k=target occupied count; train CE probe on detached tokens with the same targets and probe budget, then compare against joint and fixed center prior on identical frames",
        "decision": "joint localization must exceed both CE and center on patch AP, and shuffled_joint must not match joint gain over CE; classify result as inconclusive if either comparison is unavailable",
        "test_evaluated": False,
    }


class Gate2Dataset(Dataset):
    def __init__(self, rows: list[dict], paired: dict[str, dict], rgb_cache: ProcessedUInt8Cache,
                 target_cache: Path, *, shuffle_targets: dict[str, str] | None = None) -> None:
        self.rows, self.paired, self.rgb_cache, self.target_cache = rows, paired, rgb_cache, target_cache
        self.shuffle_targets = shuffle_targets or {}

    def __len__(self) -> int:
        return len(self.rows)

    def _target(self, sample_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.paired[sample_id]
        path = self.target_cache / f"{sample_id}.pt"
        skeleton_stat = Path(row["skeleton_path"]).stat()
        video_stat = Path(row["rgb_path"]).stat()
        source = [skeleton_stat.st_size, skeleton_stat.st_mtime_ns, video_stat.st_size, video_stat.st_mtime_ns]
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload.get("sample_id") != sample_id or payload.get("shape") != [8, 49] or payload.get("source") != source:
                raise ValueError(f"Invalid spatial target cache: {path}")
            return payload["occupancy"].float(), payload["frame_mask"].bool()
        with av.open(str(row["rgb_path"])) as container:
            video = container.streams.video[0]
            count, width, height = video.frames, video.width, video.height
        if count < 8:
            raise ValueError(f"Unsupported RGB frame count for {sample_id}")
        indices = uniform_frame_indices(8, count).tolist()
        targets = build_spatial_targets(Path(row["skeleton_path"]), indices, source_width=width, source_height=height)
        payload = {"sample_id": sample_id, "shape": [8, 49], "source": source,
                   "occupancy": targets.occupancy.bool(), "frame_mask": targets.frame_mask}
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{random.getrandbits(64):016x}.tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        return targets.occupancy, targets.frame_mask

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        sample_id = _sample_id(row["video_path"])
        pixels = self.rgb_cache.load(Path(row["video_path"]))
        target_id = self.shuffle_targets.get(sample_id, sample_id)
        if target_id in self.paired:
            occupancy, mask = self._target(target_id)
        else:
            occupancy, mask = torch.zeros(8, 49), torch.zeros(8, dtype=torch.bool)
        return pixels, int(row["class_id"]) - 1, occupancy, mask


def _worker_seed(_: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def _loader(dataset: Dataset, args: argparse.Namespace, *, train: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=train, num_workers=args.workers,
                      pin_memory=True, persistent_workers=args.workers > 0, worker_init_fn=_worker_seed,
                      generator=torch.Generator().manual_seed(args.seed))


def _model(args: argparse.Namespace, device: torch.device) -> XCLIPSpatialLoRAClassifier:
    base = XCLIPModel.from_pretrained("microsoft/xclip-base-patch32", revision=MODEL_REVISION, local_files_only=True)
    model = XCLIPSpatialLoRAClassifier(base, num_classes=120, head_dropout=0.25, feature_noise_std=0.01,
                                      upper_blocks=2, lora_rank=16, lora_alpha=32.0, lora_dropout=0.05)
    source = torch.load(args.init_adapter, map_location="cpu", weights_only=False)["model"]
    if source["target_modules"] != model.target_modules:
        raise ValueError("Warm-start LoRA targets differ")
    load_lora_state_dict(model, source["lora_state_dict"])
    model.classifier.load_state_dict(source["classifier_state_dict"], strict=True)
    return model.to(device)


def _snapshot(model: XCLIPSpatialLoRAClassifier) -> dict[str, object]:
    return {"lora": lora_state_dict(model), "classifier": {k: v.detach().cpu() for k, v in model.classifier.state_dict().items()},
            "spatial_head": {k: v.detach().cpu() for k, v in model.spatial_head.state_dict().items()}}


def _restore(model: XCLIPSpatialLoRAClassifier, state: dict[str, object]) -> None:
    load_lora_state_dict(model, state["lora"])
    model.classifier.load_state_dict(state["classifier"], strict=True)
    model.spatial_head.load_state_dict(state["spatial_head"], strict=True)


def _optimizer(model: XCLIPSpatialLoRAClassifier) -> torch.optim.AdamW:
    groups: dict[tuple[str, bool], list[torch.nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        family = "spatial" if name.startswith("spatial_head.") else "head" if name.startswith("classifier.") else "vision" if name.startswith("vision_model.") else "mit"
        no_decay = parameter.ndim == 1 or name.endswith(".bias")
        groups.setdefault((family, no_decay), []).append(parameter)
    rates = {"spatial": 1e-4, "vision": 1e-4, "mit": 2e-5, "head": 8e-5}
    return torch.optim.AdamW([{"params": values, "lr": rates[family], "weight_decay": 0 if no_decay else 5e-4}
                              for (family, no_decay), values in groups.items()])


def _patch_ap(scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    order = scores.argsort(dim=-1, descending=True, stable=True)
    positives = target.gather(-1, order).float()
    precision = positives.cumsum(-1) / torch.arange(1, positives.shape[-1] + 1, device=scores.device)
    return (precision * positives).sum(-1) / positives.sum(-1).clamp_min(1)


def _evaluate(model: XCLIPSpatialLoRAClassifier, loader: DataLoader, args: argparse.Namespace,
              device: torch.device) -> dict[str, float]:
    model.eval()
    correct = seen = valid_frames = 0
    ap_sum = topk_sum = center_ap_sum = 0.0
    center = -torch.tensor([(r - 3) ** 2 + (c - 3) ** 2 for r in range(7) for c in range(7)], device=device).float()
    with torch.inference_mode():
        for index, (pixels, labels, target, mask) in enumerate(loader):
            if args.max_val_batches is not None and index >= args.max_val_batches:
                break
            pixels, labels, target, mask = (item.to(device, non_blocking=True) for item in (pixels, labels, target, mask))
            with torch.autocast("cuda"):
                logits, spatial = model(pixels, return_spatial=True)
            correct += int((logits.argmax(1) == labels).sum())
            seen += len(labels)
            if bool(mask.any()):
                scores, occupied = spatial.float()[mask], target[mask]
                count = occupied.sum(-1).long()
                ap_sum += float(_patch_ap(scores, occupied).sum())
                center_ap_sum += float(_patch_ap(center.expand_as(scores), occupied).sum())
                ranking = scores.argsort(dim=-1, descending=True, stable=True)
                hit = occupied.gather(-1, ranking).cumsum(-1).gather(-1, (count - 1).unsqueeze(-1)).squeeze(-1) / count
                topk_sum += float(hit.sum())
                valid_frames += len(scores)
    if seen == 0 or valid_frames == 0:
        raise RuntimeError("Validation lacks RGB samples or valid paired frames")
    if args.max_val_batches is None and seen != 12960:
        raise RuntimeError("Incomplete canonical validation split")
    return {"accuracy": correct / seen, "samples": seen, "patch_ap": ap_sum / valid_frames,
            "center_patch_ap": center_ap_sum / valid_frames, "topk_recall": topk_sum / valid_frames,
            "valid_frames": valid_frames}


def _atomic_save(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _train_arm(arm: str, args: argparse.Namespace, lock: dict, split: dict, paired: dict,
               processor: XCLIPProcessor, device: torch.device) -> dict[str, object]:
    output = args.artifacts_dir / arm
    output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    rgb_cache = ProcessedUInt8Cache(args.rgb_cache_dir, processor, model_name=lock["model_name"], model_revision=MODEL_REVISION)
    paired_train = {row["sample_id"]: row for row in paired["train"]}
    paired_val = {row["sample_id"]: row for row in paired["val"]}
    shuffled = {}
    if arm == "shuffled_joint":
        ids = sorted(paired_train)
        permutation = torch.randperm(len(ids), generator=torch.Generator().manual_seed(args.seed)).tolist()
        # A single cycle makes every target come from another sample.
        shuffled = {ids[permutation[i]]: ids[permutation[(i + 1) % len(ids)]] for i in range(len(ids))}
    train_loader = _loader(Gate2Dataset(split["train"], paired_train, rgb_cache, args.target_cache_dir, shuffle_targets=shuffled), args, train=True)
    val_loader = _loader(Gate2Dataset(split["val"], paired_val, rgb_cache, args.target_cache_dir), args, train=False)
    model = _model(args, device)
    # A detached probe in CE gives a matched localization reference without spatial gradients to the classifier.
    optimizer = _optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")
    checkpoint_path = output / "last_checkpoint.pt"
    history = []
    start_epoch = 1
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["lock"] != lock or checkpoint["arm"] != arm:
            raise ValueError("Existing checkpoint differs from Gate 2 lock")
        _restore(model, checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        train_loader.generator.set_state(checkpoint["loader_rng_state"])
        history = checkpoint["history"]
        start_epoch = checkpoint["epoch"] + 1
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        seen = 0
        loss_sum = 0.0
        started = time.perf_counter()
        for index, (pixels, labels, target, mask) in enumerate(train_loader):
            if args.max_train_batches is not None and index >= args.max_train_batches:
                break
            pixels, labels, target, mask = (item.to(device, non_blocking=True) for item in (pixels, labels, target, mask))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                if arm == "ce":
                    logits, tokens = model(pixels, return_patch_tokens=True)
                    spatial = model.spatial_head(tokens.detach()).squeeze(-1)
                else:
                    logits, spatial = model(pixels, return_spatial=True)
                ce = F.cross_entropy(logits.float(), labels, label_smoothing=0.05)
                loss = ce + args.spatial_weight * spatial_region_loss(spatial, target, mask)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite Gate 2 loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            scaler.step(optimizer)
            scaler.update()
            seen += len(labels)
            loss_sum += float(loss.detach()) * len(labels)
            if (index + 1) % 100 == 0:
                write_json(output / "progress.json", {"arm": arm, "epoch": epoch, "batches": index + 1,
                                                     "samples": seen, "elapsed_seconds": time.perf_counter() - started, "test_evaluated": False})
        if seen == 0 or (args.max_train_batches is None and seen != len(split["train"])):
            raise RuntimeError("Incomplete Gate 2 training epoch")
        metrics = _evaluate(model, val_loader, args, device)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": loss_sum / seen, "train_samples": seen, "validation": metrics,
               "elapsed_seconds": time.perf_counter() - started}
        history.append(row)
        _atomic_save(checkpoint_path, {"lock": lock, "arm": arm, "epoch": epoch, "model": _snapshot(model),
                                       "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                                       "scaler": scaler.state_dict(), "history": history, "torch_rng_state": torch.get_rng_state(),
                                       "cuda_rng_states": torch.cuda.get_rng_state_all(), "loader_rng_state": train_loader.generator.get_state()})
        write_json(output / "progress.json", {"arm": arm, "completed_epoch": epoch, "latest": row, "test_evaluated": False})
    report = {"status": "completed", "arm": arm, "history": history, "final": history[-1]["validation"],
              "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device), "test_evaluated": False}
    write_json(output / "run_report.json", report)
    return report


def run(args: argparse.Namespace) -> dict[str, object]:
    lock = make_lock(args)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.artifacts_dir / "experiment_lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise ValueError("Existing Gate 2 lock differs; use a new artifacts directory")
    write_json(lock_path, lock)
    if not torch.cuda.is_available():
        return {"status": "locked_pending_gpu", "lock": str(lock_path), "reason": "CUDA GPU unavailable", "test_evaluated": False}
    device = torch.device("cuda:0")
    processor = XCLIPProcessor.from_pretrained(lock["model_name"], revision=MODEL_REVISION, local_files_only=True)
    split = json.loads(args.split_manifest.read_text())
    paired = json.loads(args.paired_manifest.read_text())
    results = {}
    for arm in ARMS:
        results[arm] = _train_arm(arm, args, lock, split, paired, processor, device)
        torch.cuda.empty_cache()
    return {"status": "completed", "arms": {arm: result["final"] for arm, result in results.items()}, "test_evaluated": False}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Lock Phase 4 Gate 2 paired ablation before GPU training")
    parser.add_argument("--split-manifest", type=Path, default=Path("artifacts/xclip/selected_split.json"))
    parser.add_argument("--paired-manifest", type=Path, default=Path("artifacts/skeleton_data/paired_manifest.json"))
    parser.add_argument("--gate0-decision", type=Path, default=Path("artifacts/skeletonclip_phase4/gate0/gate0_decision.json"))
    parser.add_argument("--gate1-report", type=Path, default=Path("artifacts/skeletonclip_phase4/gate1/gate1_report.json"))
    parser.add_argument("--init-adapter", type=Path, default=Path("artifacts/skeletonclip_phase3/vision_batch16_workers24_uint8_seed42/vision_lora_ce_seed42/best_adapter.pt"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/skeletonclip_phase4/gate2"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--spatial-weight", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--rgb-cache-dir", type=Path, default=Path("cache/skeletonclip_phase3_rgb_uint8"))
    parser.add_argument("--target-cache-dir", type=Path, default=Path("cache/skeletonclip_phase4_targets_v1"))
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    print(json.dumps(run(parser.parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
