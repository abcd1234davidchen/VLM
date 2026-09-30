"""Locked decision-coupled spatial training, canonical validation and masking."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import av
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import XCLIPModel, XCLIPProcessor

from vlm.experiments.video_sampling import uniform_frame_indices
from vlm.models.lora import load_lora_state_dict, lora_state_dict, merge_lora, merged_base_state_dict
from vlm.models.xclip_vision_lora import XCLIPAttentionClassifier, XCLIPAttentionLoRAClassifier
from vlm.processed_uint8_cache import ProcessedUInt8Cache
from vlm.reporting import write_json
from vlm.skeleton_teacher_cache import sha256_file
from vlm.spatial_joint_targets import MOTION_TARGET_VERSION, build_motion_spatial_targets, spatial_attention_loss
from vlm.training.engine import set_seed
from vlm.training.skeletonclip_phase4_gate2 import Gate2Dataset, _sample_id, _worker_seed
from vlm.training.spatial_attention_smoke import MODEL, REVISION
from vlm.xclip_features import FRAME_SAMPLING_STRATEGY


def configure_adaptation(model: XCLIPAttentionLoRAClassifier, adaptation: str) -> None:
    if adaptation == "attention-only":
        model.requires_grad_(False)
        model.spatial_head.requires_grad_(True)
        model.pooling_alpha.requires_grad_(True)
        model.eval()
    elif adaptation != "joint":
        raise ValueError(f"Unsupported adaptation: {adaptation}")


class MotionDataset(Gate2Dataset):
    """Separate versioned cache; never consume occupancy targets from v1."""

    def _target(self, sample_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.paired[sample_id]
        skeleton, video = Path(row["skeleton_path"]), Path(row["rgb_path"])
        if _sample_id(str(video)) != sample_id or skeleton.stem != sample_id:
            raise ValueError("Paired source identity changed")
        skeleton_stat, video_stat = skeleton.stat(), video.stat()
        source = {"skeleton_sha256": sha256_file(skeleton),
                  "rgb_stat": [video_stat.st_size, video_stat.st_mtime_ns],
                  "skeleton_stat": [skeleton_stat.st_size, skeleton_stat.st_mtime_ns]}
        path = self.target_cache / MOTION_TARGET_VERSION / f"{sample_id}.pt"
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if (payload.get("source") != source or payload.get("version") != MOTION_TARGET_VERSION
                    or payload.get("sample_id") != sample_id):
                raise ValueError(f"Motion cache provenance differs: {path}")
            distribution, mask = payload["distribution"], payload["frame_mask"]
            if distribution.shape != (8, 49) or mask.shape != (8,):
                raise ValueError(f"Motion cache geometry differs: {path}")
            spatial_attention_loss(torch.zeros_like(distribution).unsqueeze(0), distribution.unsqueeze(0), mask.unsqueeze(0))
            return distribution, mask
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            count, width, height = stream.frames, stream.width, stream.height
        if count < 8 or abs(count - row["skeleton"]["frame_count"]) > 2:
            # Samples rejected by alignment cannot supply motion supervision.
            return torch.zeros(8, 49), torch.zeros(8, dtype=torch.bool)
        target = build_motion_spatial_targets(skeleton, uniform_frame_indices(8, count).tolist(),
                                             source_width=width, source_height=height)
        skeleton_after, video_after = skeleton.stat(), video.stat()
        if ((skeleton_after.st_size, skeleton_after.st_mtime_ns) != (skeleton_stat.st_size, skeleton_stat.st_mtime_ns)
                or (video_after.st_size, video_after.st_mtime_ns) != (video_stat.st_size, video_stat.st_mtime_ns)):
            raise RuntimeError("Motion source changed during extraction")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{random.getrandbits(64):016x}.tmp")
        torch.save({"source": source, "version": MOTION_TARGET_VERSION, "sample_id": sample_id,
                    "distribution": target.distribution, "frame_mask": target.frame_mask}, temporary)
        temporary.replace(path)
        return target.distribution, target.frame_mask


def classification_report(logits: torch.Tensor, labels: torch.Tensor, ids: list[str], paired_ids: set[str]) -> dict:
    prediction = logits.argmax(-1)
    confusion = torch.bincount(labels * 120 + prediction, minlength=120 * 120).reshape(120, 120)
    count = confusion.sum(1)
    f1 = 2 * confusion.diag() / (confusion.sum(0) + count).clamp_min(1)
    accuracy = prediction.eq(labels)
    paired_mask = torch.tensor([sample in paired_ids for sample in ids])
    groups = {"A001_A059": labels < 59, "A060_A120": labels >= 59,
              "paired": paired_mask, "unpaired": ~paired_mask}
    return {"raw_accuracy": float(accuracy.float().mean()), "raw_macro_f1": float(f1[count > 0].mean()),
            "groups": {name: {"samples": int(mask.sum()), "accuracy": float(accuracy[mask].float().mean()) if mask.any() else None}
                       for name, mask in groups.items()},
            "per_action": {f"A{index + 1:03d}": {"samples": int(count[index]),
                            "accuracy": float(confusion[index, index] / count[index]) if count[index] else None,
                            "f1": float(f1[index])} for index in range(120)}, "confusion": confusion.tolist()}


def mask_patches(pixels: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
    masked = pixels.clone()
    for sample in range(patches.shape[0]):
        for frame in range(patches.shape[1]):
            for patch in patches[sample, frame].tolist():
                row, col = divmod(patch, 7)
                masked[sample, frame, :, row * 32:(row + 1) * 32, col * 32:(col + 1) * 32] = 0
    return masked


def evaluate(model: XCLIPAttentionLoRAClassifier, loader: DataLoader, *, device: torch.device,
             causal_samples: int, progress_path: Path | None = None) -> tuple[torch.Tensor, torch.Tensor, dict]:
    model.eval()
    logits_all, labels_all = [], []
    ce_sum = center_sum = recall_sum = 0.0
    valid_frames = 0
    causal = []
    center = -torch.tensor([(r - 3) ** 2 + (c - 3) ** 2 for r in range(7) for c in range(7)], device=device).float()
    generator = torch.Generator().manual_seed(42)
    processed = 0
    started = time.monotonic()
    with torch.inference_mode():
        for pixels, labels, target, mask in loader:
            pixels, labels, target, mask = pixels.to(device), labels.to(device), target.to(device), mask.to(device)
            logits, attention = model(pixels, return_spatial=True)
            logits_all.append(logits.cpu())
            labels_all.append(labels.cpu())
            frames = int(mask.sum())
            if frames:
                ce_sum += float(spatial_attention_loss(attention, target, mask)) * frames
                center_sum += float(spatial_attention_loss(center.expand_as(attention), target, mask)) * frames
                recall_sum += float(target.gather(-1, attention.topk(5, -1).indices).sum(-1)[mask].sum())
                valid_frames += frames
            for sample in range(len(labels)):
                if len(causal) >= causal_samples or not bool(mask[sample].any()):
                    continue
                scores = attention[sample:sample + 1]
                rankings = {"highest": scores.topk(5, -1).indices, "lowest": (-scores).topk(5, -1).indices,
                            "random": torch.rand(scores.shape, generator=generator).topk(5, -1).indices.to(device)}
                record = {}
                for name, selected in rankings.items():
                    masked_logits = model(mask_patches(pixels[sample:sample + 1], selected))
                    record[name] = float(logits[sample, labels[sample]] - masked_logits[0, labels[sample]])
                causal.append(record)
            processed += len(labels)
            if progress_path is not None and (processed % (loader.batch_size * 100) == 0 or processed == len(loader.dataset)):
                write_json(progress_path, {"state": "validation", "samples": processed, "total_samples": len(loader.dataset),
                                           "elapsed_seconds": time.monotonic() - started, "test_evaluated": False})
    return torch.cat(logits_all), torch.cat(labels_all), {
        "valid_motion_frames": valid_frames,
        "attention_ce": ce_sum / valid_frames if valid_frames else None,
        "center_ce": center_sum / valid_frames if valid_frames else None,
        "motion_mass_recall_at_5": recall_sum / valid_frames if valid_frames else None,
        "causal_samples": len(causal), "causal_per_sample": causal,
        "causal_mean_logit_drop": {name: sum(row[name] for row in causal) / len(causal) if causal else None
                                  for name in ("highest", "lowest", "random")},
        "causal_scope": "first valid paired validation samples in canonical order, five masked patches per frame"}


def run(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    torch.set_num_threads(8)
    split = json.loads(args.split_manifest.read_text())
    paired = json.loads(args.paired_manifest.read_text())
    alignment = json.loads(args.gate0_decision.read_text())
    screen = json.loads(args.mechanism_report.read_text())
    if (alignment.get("status") != "pass" or alignment.get("test_evaluated") is not False
            or alignment.get("manifest_sha256") != sha256_file(args.paired_manifest)):
        raise ValueError("Current paired manifest is not alignment-certified")
    if screen.get("status") != "pass" or screen.get("test_evaluated") is not False:
        raise ValueError("Motion target mechanism screen did not pass")
    if args.mode == "pilot":
        smoke = json.loads(args.joint_smoke_report.read_text())
        if smoke.get("gate1_pass") is not True or smoke.get("mode") != "smoke" or smoke.get("test_evaluated") is not False:
            raise ValueError("Full joint-training Gate 1 has not passed")
        smoke_lock = json.loads((args.joint_smoke_report.parent / "experiment_lock.json").read_text())
        if smoke_lock.get("adaptation", "joint") != args.adaptation:
            raise ValueError("Smoke adaptation policy differs from requested pilot")
        for key, path in (("targets", Path(__file__).parents[1] / "spatial_joint_targets.py"),
                          ("model", Path(__file__).parents[1] / "models/xclip_vision_lora.py"),
                          ("init_adapter", args.init_adapter), ("split", args.split_manifest), ("paired", args.paired_manifest)):
            if smoke_lock["source_sha256"][key] != sha256_file(path):
                raise ValueError(f"Joint smoke does not certify current {key}")
    if (split.get("frame_sampling_strategy") != FRAME_SAMPLING_STRATEGY or split.get("max_seq_len") != 8
            or (len(split["train"]), len(split["val"])) != (80640, 12960)):
        raise ValueError("Canonical RGB split/sampling differs")
    paired_rows = {row["sample_id"]: row for part in ("train", "val") for row in paired[part]}
    for part in ("train", "val"):
        rgb = {_sample_id(row["video_path"]): row for row in split[part]}
        if len(rgb) != len(split[part]):
            raise ValueError("Duplicate RGB sample identities")
        for row in paired[part]:
            source = rgb.get(row["sample_id"])
            if source is None or row["class_id"] != f"A{int(source['class_id']):03d}":
                raise ValueError("Paired labels/split differ from RGB split")
    source = torch.load(args.init_adapter, map_location="cpu", weights_only=False)
    identity = source["identity"]
    if (identity.get("model_revision") != REVISION or identity.get("arm") != "vision"
            or identity.get("split_sha256") != sha256_file(args.split_manifest) or identity.get("test_evaluated") is not False):
        raise ValueError("Warm-start provenance differs")
    train_rows, val_rows = split["train"], split["val"]
    if args.mode == "smoke":
        # Fixed paired examples cover several classes and unpaired NTU120 examples.
        train_ids = {row["sample_id"] for row in paired["train"][::max(1, len(paired["train"]) // 12)][:12]}
        train_rows = [row for row in train_rows if _sample_id(row["video_path"]) in train_ids] + [row for row in train_rows if int(row["class_id"]) > 60][:4]
        val_ids = {row["sample_id"] for row in paired["val"][::max(1, len(paired["val"]) // 12)][:12]}
        val_rows = [row for row in val_rows if _sample_id(row["video_path"]) in val_ids] + [row for row in val_rows if int(row["class_id"]) > 60][:4]
    elif args.train_samples < len(train_rows):
        # Equal-class round robin of fixed hash orders; full validation is untouched.
        buckets = {class_id: [] for class_id in range(1, 121)}
        for row in train_rows:
            buckets[int(row["class_id"])].append(row)
        for bucket in buckets.values():
            bucket.sort(key=lambda row: hashlib.sha256(f"{args.seed}:{_sample_id(row['video_path'])}".encode()).hexdigest())
        train_rows = [buckets[class_id][index] for index in range(max(map(len, buckets.values())))
                      for class_id in buckets if index < len(buckets[class_id])][:args.train_samples]
    lock = {"schema_version": 2, "target_version": MOTION_TARGET_VERSION, "mode": args.mode,
            "adaptation": args.adaptation,
            "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size, "spatial_weight": 0.1,
            "arms": ["correct", "shuffled"] if args.mode == "smoke" else ["correct"],
            "train_samples": len(train_rows), "validation_samples": len(val_rows),
            "train_ids": [_sample_id(row["video_path"]) for row in train_rows],
            "validation_ids": [_sample_id(row["video_path"]) for row in val_rows],
            "source_sha256": {"split": sha256_file(args.split_manifest), "paired": sha256_file(args.paired_manifest),
                              "init_adapter": sha256_file(args.init_adapter), "baseline_logits": sha256_file(args.baseline_logits),
                              "runner": sha256_file(Path(__file__)), "mechanism_report": sha256_file(args.mechanism_report),
                              "targets": sha256_file(Path(__file__).parents[1] / "spatial_joint_targets.py"),
                              "model": sha256_file(Path(__file__).parents[1] / "models/xclip_vision_lora.py")},
            "test_evaluated": False, "selection": "fixed final epoch; no epoch or weight search",
            "causal_samples": args.causal_samples, "workers": args.workers,
            "precision": "fp32",
            "train_selection": "seeded identity hash, equal-class round robin" if args.mode == "pilot" else "fixed mechanism subset",
            "joint_smoke_sha256": sha256_file(args.joint_smoke_report) if args.mode == "pilot" else None,
            "learning_rates": {"vision": 1e-4, "mit": 2e-5, "classifier": 8e-5, "attention_and_alpha": 1e-4}}
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.artifacts_dir / "experiment_lock.json"
    if lock_path.exists():
        raise ValueError("Run directory already locked; reuse requires explicit checkpoint resume support")
    write_json(lock_path, lock)
    if not torch.cuda.is_available():
        return {"status": "locked_pending_gpu", "test_evaluated": False}
    device = torch.device("cuda:0")
    processor = XCLIPProcessor.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    cache = ProcessedUInt8Cache(args.rgb_cache_dir, processor, model_name=MODEL, model_revision=REVISION)
    baseline = torch.load(args.baseline_logits, map_location="cpu", weights_only=True)
    baseline_index = {sample: index for index, sample in enumerate(baseline["sample_ids"])}
    if len(baseline_index) != 12960 or set(baseline_index) != {_sample_id(row["video_path"]) for row in split["val"]}:
        raise ValueError("Baseline predictions do not cover canonical validation")
    selected = torch.tensor([baseline_index[_sample_id(row["video_path"])] for row in val_rows])
    expected_labels = torch.tensor([int(row["class_id"]) - 1 for row in val_rows])
    if not torch.equal(baseline["labels"][selected], expected_labels):
        raise ValueError("Baseline validation labels differ")
    baseline_report = classification_report(baseline["logits"][selected], expected_labels, lock["validation_ids"], set(paired_rows))
    write_json(args.artifacts_dir / "baseline_report.json", baseline_report)
    results = {}
    for arm in lock["arms"]:
        set_seed(args.seed)
        base = XCLIPModel.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        model = XCLIPAttentionLoRAClassifier(base, num_classes=120, head_dropout=0.25, feature_noise_std=0.01,
                                            upper_blocks=2, lora_rank=16, lora_alpha=32, lora_dropout=0.05).to(device)
        if source["model"]["target_modules"] != model.target_modules:
            raise ValueError("LoRA target provenance changed")
        load_lora_state_dict(model, source["model"]["lora_state_dict"])
        model.classifier.load_state_dict(source["model"]["classifier_state_dict"], strict=True)
        configure_adaptation(model, args.adaptation)
        groups = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("vision_model."):
                family, rate = "vision", 1e-4
            elif name.startswith("mit."):
                family, rate = "mit", 2e-5
            elif name.startswith("classifier."):
                family, rate = "classifier", 8e-5
            elif name.startswith("spatial_head.") or name == "pooling_alpha":
                family, rate = "spatial", 1e-4
            else:
                raise ValueError(f"Unexpected trainable parameter: {name}")
            if family in ("vision", "mit") and not name.endswith(("lora_a", "lora_b")):
                raise ValueError(f"Unfrozen base parameter: {name}")
            groups.setdefault((family, rate), []).append(parameter)
        optimizer = torch.optim.AdamW([{"params": values, "lr": rate, "weight_decay": 0 if family == "spatial" else 5e-4}
                                       for (family, rate), values in groups.items()])
        mapping = {}
        if arm == "shuffled":
            ids = sorted(_sample_id(row["video_path"]) for row in train_rows if _sample_id(row["video_path"]) in paired_rows)
            mapping = dict(zip(ids, ids[1:] + ids[:1]))
        train_dataset = MotionDataset(train_rows, paired_rows, cache, args.target_cache_dir, shuffle_targets=mapping)
        val_dataset = MotionDataset(val_rows, paired_rows, cache, args.target_cache_dir)
        common = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                      persistent_workers=args.workers > 0, worker_init_fn=_worker_seed)
        train_loader = DataLoader(train_dataset, shuffle=True, generator=torch.Generator().manual_seed(args.seed), **common)
        val_loader = DataLoader(val_dataset, shuffle=False, **common)
        identity_error = None
        if args.adaptation == "attention-only":
            with torch.inference_mode():
                initial_pixels = next(iter(val_loader))[0].to(device)
                initial_logits = model(initial_pixels).cpu()
            identity_error = float((initial_logits - baseline["logits"][selected[:len(initial_logits)]]).abs().max())
            if identity_error > 1e-4:
                raise RuntimeError(f"Frozen baseline identity parity failed: {identity_error}")
        arm_dir = args.artifacts_dir / arm
        arm_dir.mkdir()
        history = []
        for epoch in range(args.epochs):
            if args.adaptation == "attention-only":
                # Frozen backbone/head dropout and feature noise would obscure the
                # baseline-preserving hypothesis; gradients still reach pooling.
                model.eval()
            else:
                model.train()
            started = time.monotonic()
            ce_sum = spatial_sum = count = supervised = 0
            for index, (pixels, labels, target, mask) in enumerate(train_loader):
                pixels, labels, target, mask = pixels.to(device), labels.to(device), target.to(device), mask.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits, attention = model(pixels, return_spatial=True)
                ce = F.cross_entropy(logits.float(), labels, label_smoothing=0.1)
                spatial = spatial_attention_loss(attention, target, mask)
                loss = ce + 0.1 * spatial
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite joint loss")
                loss.backward()
                if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                    raise RuntimeError("Nonfinite joint gradient")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                ce_sum += float(ce.detach()) * len(labels)
                spatial_sum += float(spatial.detach()) * int(mask.sum())
                count += len(labels)
                supervised += int(mask.sum())
                if index % 100 == 0:
                    write_json(arm_dir / "progress.json", {"epoch": epoch + 1, "batch": index + 1, "samples": count,
                                                           "alpha": float(model.pooling_alpha.detach()), "test_evaluated": False,
                                                           "state": "training", "elapsed_seconds": time.monotonic() - started,
                                                           "samples_per_second": count / (time.monotonic() - started)})
            write_json(arm_dir / "progress.json", {"epoch": epoch + 1, "state": "validation", "test_evaluated": False})
            logits, labels, spatial_metrics = evaluate(model, val_loader, device=device, causal_samples=args.causal_samples,
                                                       progress_path=arm_dir / "progress.json")
            if not torch.equal(labels, expected_labels):
                raise RuntimeError("Validation loader order changed")
            metrics = classification_report(logits, labels, lock["validation_ids"], set(paired_rows))
            record = {"epoch": epoch + 1, "train_ce": ce_sum / count,
                      "train_attention_ce": spatial_sum / supervised if supervised else None,
                      "alpha": float(model.pooling_alpha.detach()), "classification": metrics, "spatial": spatial_metrics}
            history.append(record)
            write_json(arm_dir / "history.json", history)
            torch.save({"logits": logits, "labels": labels, "sample_ids": lock["validation_ids"], "provenance": lock}, arm_dir / "validation_predictions.pt")
            torch.save({"lora": lora_state_dict(model), "classifier": model.classifier.state_dict(),
                        "spatial_head": model.spatial_head.state_dict(), "pooling_alpha": model.pooling_alpha.detach(),
                        "optimizer": optimizer.state_dict(), "epoch": epoch + 1, "provenance": lock}, arm_dir / "last_checkpoint.pt")
            print(json.dumps({"arm": arm, "epoch": epoch + 1, "raw_accuracy": metrics["raw_accuracy"],
                              "attention_ce": spatial_metrics["attention_ce"], "alpha": record["alpha"]}), flush=True)
        model.eval()
        baseline_unchanged = None
        if args.adaptation == "attention-only":
            baseline_unchanged = (all(torch.equal(value.cpu(), source["model"]["lora_state_dict"][name])
                                      for name, value in lora_state_dict(model).items())
                                  and all(torch.equal(value.cpu(), source["model"]["classifier_state_dict"][name])
                                          for name, value in model.classifier.state_dict().items()))
            if not baseline_unchanged:
                raise RuntimeError("Attention-only training changed frozen baseline weights")
        rgb = next(iter(val_loader))[0][:1].to(device)
        with torch.no_grad():
            reference = model(rgb)
        merge_lora(model)
        merged_state = merged_base_state_dict(model)
        torch.save({"state_dict": merged_state, "provenance": lock}, arm_dir / "merged_rgb_only.pt")
        del model, base, optimizer
        torch.cuda.empty_cache()
        restored_base = XCLIPModel.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        restored = XCLIPAttentionClassifier(restored_base, num_classes=120, head_dropout=0.25).to(device).eval()
        restored.load_state_dict(merged_state, strict=True)
        with torch.no_grad():
            difference = float((restored(rgb) - reference).abs().max())
        if difference > 1e-4:
            raise RuntimeError("Merged RGB-only checkpoint parity failed")
        results[arm] = {"final": history[-1], "rgb_only_reload_max_abs_diff": difference,
                        "baseline_weights_unchanged": baseline_unchanged, "baseline_identity_max_abs_diff": identity_error}
        write_json(arm_dir / "progress.json", {"epoch": args.epochs, "state": "complete", "test_evaluated": False})
        del restored, restored_base, merged_state, train_loader, val_loader
        torch.cuda.empty_cache()
    correct = results["correct"]["final"]
    raw_gain = correct["classification"]["raw_accuracy"] - baseline_report["raw_accuracy"]
    decision = "smoke_complete_pending_gate_review" if args.mode == "smoke" else "no_go"
    gate1_pass = None
    if args.mode == "smoke":
        gate1_pass = (correct["spatial"]["attention_ce"] < results["shuffled"]["final"]["spatial"]["attention_ce"]
                      and abs(correct["alpha"]) > 1e-6)
        decision = "gate1_pass" if gate1_pass else "gate1_no_go"
    if args.mode == "pilot":
        drop = correct["spatial"]["causal_mean_logit_drop"]
        # A four-sample head screen cannot be compared numerically to canonical validation.
        attention_ok = correct["spatial"]["attention_ce"] < correct["spatial"]["center_ce"]
        causal_ok = drop["highest"] > max(drop["random"], drop["lowest"])
        if attention_ok and causal_ok and raw_gain > 0:
            decision = "confirmation_required" if raw_gain >= 0.002 else "mechanism_only"
    report = {"status": decision, "mode": args.mode, "gate1_pass": gate1_pass, "test_evaluated": False,
              "raw_accuracy_gain": raw_gain, "arms": results,
              "limitation": "Smoke is a 16-sample mechanism check and cannot support canonical validation accuracy claims." if args.mode == "smoke" else "Single-seed pilot; confirmation and matched held-out shuffled comparison required before a winner."}
    write_json(args.artifacts_dir / "run_report.json", report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--adaptation", choices=("joint", "attention-only"), default="joint")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-samples", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--causal-samples", type=int, default=8)
    parser.add_argument("--paired-manifest", type=Path, default=Path("artifacts/skeleton_data/paired_manifest.json"))
    parser.add_argument("--split-manifest", type=Path, default=Path("artifacts/xclip/selected_split.json"))
    parser.add_argument("--gate0-decision", type=Path, default=Path("artifacts/skeletonclip_phase4/gate0/gate0_decision.json"))
    parser.add_argument("--mechanism-report", type=Path, default=Path("artifacts/spatial_attention_v2/smoke/mechanism_report.json"))
    parser.add_argument("--joint-smoke-report", type=Path, default=Path("artifacts/spatial_attention_v2/joint_smoke_v2/run_report.json"))
    parser.add_argument("--init-adapter", type=Path, default=Path("artifacts/skeletonclip_phase3/vision_batch16_workers24_uint8_seed42/vision_lora_ce_seed42/best_adapter.pt"))
    parser.add_argument("--baseline-logits", type=Path, default=Path("artifacts/skeletonclip_phase3/vision_batch16_workers24_uint8_seed42/vision_lora_ce_seed42/best_validation_logits.pt"))
    parser.add_argument("--rgb-cache-dir", type=Path, default=Path("cache/skeletonclip_phase3_rgb_uint8"))
    parser.add_argument("--target-cache-dir", type=Path, default=Path("cache/spatial_attention_v2"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/spatial_attention_v2/joint_smoke"))
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.causal_samples) < 1 or args.workers < 0 or not 120 <= args.train_samples <= 80640:
        raise ValueError("Invalid run budget")
    args.workers = min(args.workers, 24)
    print(json.dumps({key: value for key, value in run(args).items() if key != "arms"}, indent=2))


if __name__ == "__main__":
    main()
