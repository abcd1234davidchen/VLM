"""Direct 120-class decision KD with a frozen partial59 teacher and RGB anchor.

This is a supervised adaptation, not SKI online semantic co-training.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import XCLIPModel, XCLIPProcessor

from vlm.data.skeleton import SKELETON_TENSOR_PREPROCESSING_ID, load_skeleton_tensor
from vlm.models.lora import load_lora_state_dict, lora_state_dict, merge_lora, merged_base_state_dict
from vlm.models.skeletonclip import SkeletonPrototypeTeacher
from vlm.models.xclip_vision_lora import XCLIPVisionClassifier, XCLIPVisionLoRAClassifier
from vlm.processed_uint8_cache import ProcessedUInt8Cache
from vlm.reporting import write_json
from vlm.skeleton_teacher_cache import sha256_file
from vlm.training.engine import set_seed
from vlm.training.ski_semantic import MODEL, MODEL_REVISION, SemanticDataset, balanced, loader
from vlm.training.skeletonclip_phase4_gate2 import _sample_id
from vlm.training.spatial_attention import classification_report
from vlm.xclip_features import FRAME_SAMPLING_STRATEGY


def score_kl(student: torch.Tensor, target: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    """Frozen target KL, averaged per sample, with conventional T squared scaling."""
    if student.shape != target.shape or student.ndim != 2 or temperature <= 0:
        raise ValueError("KD needs equal [samples, classes] shapes and positive temperature")
    if not torch.isfinite(student).all() or not torch.isfinite(target).all():
        raise ValueError("KD scores must be finite")
    if not len(student):
        return student.sum() * 0
    return F.kl_div((student.float() / temperature).log_softmax(-1),
                    (target.detach().float() / temperature).softmax(-1),
                    reduction="batchmean") * temperature ** 2


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; no pending lock will be created")
    set_seed(args.seed)
    torch.set_num_threads(8)
    root = args.artifacts_dir
    if (root / "experiment_lock.json").exists():
        raise ValueError("Existing immutable run; use a fresh directory")
    split = json.loads(args.split_manifest.read_text())
    paired = json.loads(args.paired_manifest.read_text())
    if ((len(split["train"]), len(split["val"])) != (80640, 12960)
            or split["max_seq_len"] != 8 or split["frame_sampling_strategy"] != FRAME_SAMPLING_STRATEGY
            or (len(paired["train"]), len(paired["val"])) != (45236, 4180)):
        raise ValueError("Canonical splits/sampling changed")
    source = torch.load(args.init_adapter, map_location="cpu", weights_only=False)
    if (source["identity"].get("split_sha256") != sha256_file(args.split_manifest)
            or source["identity"].get("test_evaluated") is not False
            or source["identity"].get("model_revision") != MODEL_REVISION
            or source["identity"].get("arm") != "vision"):
        raise ValueError("RGB baseline provenance differs")
    teacher_source = torch.load(args.teacher, map_location="cpu", weights_only=False)
    identity = teacher_source["identity"]
    if (identity["head"] != "xclip" or identity["backbone"] != "blockgcn"
            or identity["ordered_class_ids"] != [f"A{i:03d}" for i in range(1, 60)]
            or identity["frames"] != 64 or identity["manifest_sha256"] != sha256_file(args.paired_manifest)
            or identity["preprocessing_id"] != SKELETON_TENSOR_PREPROCESSING_ID
            or identity["text"]["model_revision"] != MODEL_REVISION):
        raise ValueError("Partial59 text-aligned teacher provenance differs")
    skeleton, paired_ids = {}, set()
    for part in ("train", "val"):
        rows = paired[part]
        payload = torch.load(args.skeleton_cache / f"{part}_45236_4180_64.pt", map_location="cpu", weights_only=True, mmap=True)
        ids = [row["sample_id"] for row in rows]
        if (payload["sample_ids"] != ids or len(set(ids)) != len(ids)
                or payload["preprocessing_id"] != SKELETON_TENSOR_PREPROCESSING_ID
                or not torch.equal(payload["labels"], torch.tensor([row["class_index"] for row in rows]))
                or payload["inputs"].shape != (len(rows), 3, 64, 25, 2)):
            raise ValueError("Skeleton tensor cache contract differs")
        rgb = {_sample_id(row["video_path"]): row for row in split[part]}
        if len(rgb) != len(split[part]):
            raise ValueError("Duplicate RGB IDs")
        for row in rows:
            match = rgb.get(row["sample_id"])
            if match is None or int(match["class_id"]) - 1 != row["class_index"] or not 0 <= row["class_index"] < 59:
                raise ValueError("RGB/skeleton label or split mismatch")
        for index in (0, len(rows) // 2, len(rows) - 1):
            if not torch.equal(load_skeleton_tensor(Path(rows[index]["skeleton_path"]), frames=64), payload["inputs"][index]):
                raise ValueError("Raw skeleton/cache preprocessing differs")
        skeleton[part] = payload
        paired_ids.update(ids)
    if set(skeleton["train"]["sample_ids"]) & set(skeleton["val"]["sample_ids"]):
        raise ValueError("Train/validation leakage")
    baseline = torch.load(args.baseline_logits, map_location="cpu", weights_only=True)
    baseline_index = {sample: index for index, sample in enumerate(baseline["sample_ids"])}
    if len(baseline_index) != 12960 or set(baseline_index) != {_sample_id(row["video_path"]) for row in split["val"]}:
        raise ValueError("Baseline predictions lack canonical validation coverage")
    train_rows = balanced(split["train"], 128 if args.mode == "smoke" else args.train_samples, args.seed)
    val_rows = balanced(split["val"], 120, args.seed) if args.mode == "smoke" else split["val"]
    val_ids = [_sample_id(row["video_path"]) for row in val_rows]
    selected = torch.tensor([baseline_index[sample] for sample in val_ids])
    labels = torch.tensor([int(row["class_id"]) - 1 for row in val_rows])
    if not torch.equal(baseline["labels"][selected], labels):
        raise ValueError("Baseline validation labels differ")
    hashes = {"runner": sha256_file(Path(__file__)), "model": sha256_file(Path(__file__).parents[1] / "models/xclip_vision_lora.py"),
              "data_helpers": sha256_file(Path(__file__).with_name("ski_semantic.py")),
              "teacher": sha256_file(args.teacher), "init_adapter": sha256_file(args.init_adapter),
              "split": sha256_file(args.split_manifest), "paired": sha256_file(args.paired_manifest),
              "baseline_logits": sha256_file(args.baseline_logits)}
    if args.mode == "pilot":
        smoke = json.loads(args.smoke_report.read_text())
        smoke_lock = json.loads((args.smoke_report.parent / "experiment_lock.json").read_text())
        if smoke.get("status") != "pass" or smoke.get("mode") != "smoke" or smoke.get("test_evaluated") is not False or smoke_lock["source_sha256"] != hashes:
            raise ValueError("Current decision-distillation implementation lacks passing smoke")
    root.mkdir(parents=True, exist_ok=True)
    phase = root / "correct"
    phase.mkdir(exist_ok=True)
    def progress(state, **fields):
        record = {"state": state, "test_evaluated": False, **fields}
        write_json(phase / "progress.json", record)
        print(json.dumps(record), flush=True)
    lock = {"mode": args.mode, "seed": args.seed, "epochs": 1, "batch_size": args.batch_size,
            "workers": args.workers, "train_ids": [_sample_id(row["video_path"]) for row in train_rows],
            "validation_ids": val_ids, "source_sha256": hashes, "precision": "fp32",
            "loss": "CE120(label smoothing .1) + .5*T^2*KL(skeleton59 || conditional main logits59) on paired + T^2*KL(frozen RGB120 || main logits120) on unpaired",
            "temperature": 2.0, "skeleton_weight": 0.5, "anchor_weight": 1.0,
            "learning_rate": 2.25e-5, "weight_decay": 5e-4, "gradient_clip": 1.0,
            "teacher_policy": "frozen pretrained text-aligned59 skeleton teacher; not SKI online co-training",
            "anchor_policy": "frozen deep copy of original RGB120 warm-start; unpaired training data only",
            "student_policy": "vision/MIT LoRA and classifier trainable; eval mode disables dropout/noise while gradients remain enabled",
            "selection": "fixed final epoch, no weight search; promising requires +.20 pp overall and nonnegative unpaired gain",
            "test_evaluated": False}
    write_json(root / "experiment_lock.json", lock)
    (root / "runner_snapshot.py").write_text(Path(__file__).read_text())
    device = torch.device("cuda")
    teacher = SkeletonPrototypeTeacher("blockgcn", teacher_source["state_dict"]["text_prototypes"])
    teacher.load_state_dict(teacher_source["state_dict"], strict=True)
    teacher.to(device).requires_grad_(False).eval()
    model = XCLIPVisionLoRAClassifier(XCLIPModel.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=True),
                                     num_classes=120, head_dropout=0.25, feature_noise_std=0.01,
                                     upper_blocks=2, lora_rank=16, lora_alpha=32, lora_dropout=0.05).to(device).eval()
    if model.target_modules != source["model"]["target_modules"]:
        raise ValueError("RGB LoRA targets differ")
    load_lora_state_dict(model, source["model"]["lora_state_dict"])
    model.classifier.load_state_dict(source["model"]["classifier_state_dict"], strict=True)
    trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    if any(not (name.startswith("classifier.") or name.endswith(("lora_a", "lora_b"))) for name, _ in trainable):
        raise RuntimeError("Unexpected unfrozen student base parameters")
    anchor = copy.deepcopy(model).requires_grad_(False).eval()
    processor = XCLIPProcessor.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=True)
    cache = ProcessedUInt8Cache(args.rgb_cache, processor, model_name=MODEL, model_revision=MODEL_REVISION)
    train_loader = loader(SemanticDataset(train_rows, cache, skeleton["train"]), args, shuffle=True)
    val_loader = loader(SemanticDataset(val_rows, cache), args)
    pixels = next(iter(val_loader))[0].to(device)
    with torch.no_grad():
        initial = model(pixels)
        anchor_identity = float(score_kl(initial, anchor(pixels)))
        parity = float((initial.cpu() - baseline["logits"][selected[:len(initial)]]).abs().max())
    if parity > 1e-4 or abs(anchor_identity) > 1e-6:
        raise RuntimeError("Baseline/anchor identity failed")
    # Pure teacher KD, not task CE, must reach the classifier AND vision/MIT.
    batch = next(batch for batch in train_loader if batch[3].any())
    pixels, label, joints, mask = batch
    mask = mask.to(device)
    logits = model(pixels.to(device))
    with torch.no_grad():
        teacher_scores = teacher(joints.to(device)[mask])
    kd = 0.5 * score_kl(logits[mask, :59], teacher_scores)
    kd.backward()
    gradient = {family: sum(float(value.grad.abs().sum()) for name, value in trainable
                           if name.startswith(prefix) and value.grad is not None)
                for family, prefix in (("vision", "vision_model."), ("mit", "mit."), ("classifier", "classifier."))}
    if not all(value > 0 and torch.isfinite(torch.tensor(value)) for value in gradient.values()):
        raise RuntimeError(f"Decision KD gradient failure: {gradient}")
    model.zero_grad(set_to_none=True)
    progress("decision_gradient_verified", gradients=gradient, initial_anchor_kl=anchor_identity, baseline_logit_error=parity)
    parameters = [value for _, value in trainable]
    optimizer = torch.optim.AdamW(parameters, lr=2.25e-5, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader))
    seen = paired_seen = unpaired_seen = steps = 0
    sums = {key: 0.0 for key in ("ce120", "skeleton_kl59", "anchor_kl120")}
    started = time.monotonic()
    for pixels, labels_batch, joints, mask in train_loader:
        pixels, labels_batch, mask = pixels.to(device), labels_batch.to(device), mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(pixels)
        ce = F.cross_entropy(logits, labels_batch, label_smoothing=0.1)
        kd, retention = logits.sum() * 0, logits.sum() * 0
        with torch.no_grad():
            teacher_scores = teacher(joints.to(device)[mask]) if mask.any() else None
            anchor_scores = anchor(pixels[~mask]) if (~mask).any() else None
        if mask.any():
            if not bool((labels_batch[mask] < 59).all()):
                raise RuntimeError("Skeleton teacher taxonomy mismatch")
            kd = score_kl(logits[mask, :59], teacher_scores)
        if (~mask).any():
            retention = score_kl(logits[~mask], anchor_scores)
        loss = ce + 0.5 * kd + retention
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite decision loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step(); scheduler.step()
        n, np = len(labels_batch), int(mask.sum())
        seen += n; paired_seen += np; unpaired_seen += n - np; steps += 1
        sums["ce120"] += float(ce.detach()) * n
        sums["skeleton_kl59"] += float(kd.detach()) * np
        sums["anchor_kl120"] += float(retention.detach()) * (n - np)
        if steps % 25 == 0 or seen == len(train_rows):
            progress("decision_training", samples=seen, total_samples=len(train_rows), paired_samples=paired_seen,
                     losses={"ce120": sums["ce120"] / seen, "skeleton_kl59": sums["skeleton_kl59"] / max(1, paired_seen),
                             "anchor_kl120": sums["anchor_kl120"] / max(1, unpaired_seen)}, elapsed_seconds=time.monotonic() - started)
    frozen_unchanged = (all(torch.equal(value.detach().cpu(), teacher_source["state_dict"][name]) for name, value in teacher.state_dict().items())
                        and all(torch.equal(value.detach().cpu(), source["model"]["lora_state_dict"][name]) for name, value in lora_state_dict(anchor).items())
                        and all(torch.equal(value.detach().cpu(), source["model"]["classifier_state_dict"][name]) for name, value in anchor.classifier.state_dict().items()))
    if not frozen_unchanged or any(value.grad is not None for value in teacher.parameters()) or any(value.grad is not None for value in anchor.parameters()):
        raise RuntimeError("Frozen teacher/anchor changed or received gradients")
    predictions = []
    with torch.inference_mode():
        for pixels, _, _, _ in val_loader:
            predictions.append(model(pixels.to(device)).cpu())
            n = sum(len(value) for value in predictions)
            if n % (args.batch_size * 100) == 0 or n == len(val_rows):
                progress("validation", samples=n, total_samples=len(val_rows))
    logits = torch.cat(predictions)
    if not torch.isfinite(logits).all():
        raise RuntimeError("Nonfinite validation predictions")
    report = classification_report(logits, labels, val_ids, paired_ids)
    old = classification_report(baseline["logits"][selected], labels, val_ids, paired_ids)
    old_ok = baseline["logits"][selected].argmax(-1).eq(labels)
    new_ok = logits.argmax(-1).eq(labels)
    changes = {"fixed": int((~old_ok & new_ok).sum()), "regressed": int((old_ok & ~new_ok).sum())}
    pm = torch.tensor([sample in paired_ids for sample in val_ids])
    drift = float(score_kl(logits[~pm], baseline["logits"][selected][~pm]))
    write_json(root / "classification_report.json", report)
    write_json(root / "baseline_report.json", old)
    torch.save({"logits": logits, "labels": labels, "sample_ids": val_ids, "test_evaluated": False}, root / "validation_predictions.pt")
    torch.save({"lock": lock, "lora": lora_state_dict(model), "classifier": model.classifier.state_dict()}, root / "last_checkpoint.pt")
    pixels = next(iter(val_loader))[0].to(device)
    with torch.no_grad():
        reference = model(pixels)
    del anchor, teacher
    merge_lora(model)
    merged = merged_base_state_dict(model)
    restored = XCLIPVisionClassifier(XCLIPModel.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=True),
                                    num_classes=120, head_dropout=0.25, feature_noise_std=0.01).to(device).eval()
    restored.load_state_dict(merged, strict=True)
    with torch.no_grad():
        reload_error = float((restored(pixels) - reference).abs().max())
    if reload_error > 1e-4:
        raise RuntimeError("RGB-only merged reload failed")
    torch.save(merged, root / "merged_rgb_state.pt")
    gain = report["raw_accuracy"] - old["raw_accuracy"]
    retention_gain = report["groups"]["unpaired"]["accuracy"] - old["groups"]["unpaired"]["accuracy"]
    status = "pass" if args.mode == "smoke" else "promising" if gain >= 0.002 and retention_gain >= 0 else "no_go" if gain < 0 else "inconclusive"
    return {"status": status, "mode": args.mode, "raw_accuracy_gain": gain, "unpaired_accuracy_gain": retention_gain,
            "classification": report, "baseline": old, "prediction_changes": changes, "unpaired_validation_kl_T2": drift,
            "kd_gradients": gradient, "frozen_teacher_anchor_unchanged": frozen_unchanged,
            "baseline_identity_max_abs_diff": parity, "rgb_only_reload_max_abs_diff": reload_error,
            "train_samples": seen, "paired_train_samples": paired_seen, "test_evaluated": False,
            "interpretation": "Frozen partial59 decision KD and RGB retention; not official SKI or an attribution-controlled experiment."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--train-samples", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    paths = {"split_manifest": "artifacts/xclip/selected_split.json", "paired_manifest": "artifacts/skeleton_data/paired_manifest.json",
             "teacher": "artifacts/skeletonclip_gate3/xclip_seed42/best_checkpoint.pt", "skeleton_cache": "cache/skeletonclip_gate3",
             "rgb_cache": "cache/skeletonclip_phase3_rgb_uint8", "smoke_report": "artifacts/decision_distillation_v1/smoke/run_report.json",
             "init_adapter": "artifacts/skeletonclip_phase3/vision_batch16_workers24_uint8_seed42/vision_lora_ce_seed42/best_adapter.pt",
             "baseline_logits": "artifacts/skeletonclip_phase3/vision_batch16_workers24_uint8_seed42/vision_lora_ce_seed42/best_validation_logits.pt"}
    for name, path in paths.items():
        parser.add_argument("--" + name.replace("_", "-"), type=Path, default=Path(path))
    args = parser.parse_args()
    if not 0 <= args.workers <= 24 or args.batch_size < 1 or args.train_samples < 120:
        parser.error("Require workers 0-24, positive batch size, train_samples >=120")
    report = run(args)
    write_json(args.artifacts_dir / "run_report.json", report)
    print(json.dumps({key: value for key, value in report.items() if key not in ("classification", "baseline")}), flush=True)


if __name__ == "__main__":
    main()
