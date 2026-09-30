"""Bounded SKI-style adaptation: semantic prealignment, then online score KD.

Not a zero-shot paper reproduction: supervised RGB120, paired skeleton59,
frozen text prototypes and teacher BatchNorm statistics, LoRA-only vision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import XCLIPModel, XCLIPProcessor

from vlm.data.skeleton import SKELETON_TENSOR_PREPROCESSING_ID, load_skeleton_tensor
from vlm.models.lora import load_lora_state_dict, lora_state_dict, merge_lora, merged_base_state_dict
from vlm.models.skeletonclip import SkeletonPrototypeTeacher
from vlm.models.ski_semantic import SKISemanticClassifier
from vlm.models.xclip_vision_lora import XCLIPVisionClassifier
from vlm.processed_uint8_cache import ProcessedUInt8Cache
from vlm.reporting import write_json
from vlm.skeleton_teacher_cache import sha256_file
from vlm.training.engine import set_seed
from vlm.training.spatial_attention import classification_report
from vlm.training.skeletonclip_phase4_gate2 import _sample_id, _worker_seed, MODEL_REVISION
from vlm.xclip_features import FRAME_SAMPLING_STRATEGY

MODEL = "microsoft/xclip-base-patch32"


def balanced(rows: list[dict], count: int, seed: int) -> list[dict]:
    buckets = {}
    for row in rows:
        buckets.setdefault(int(row["class_id"]), []).append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: hashlib.sha256(f"{seed}:{_sample_id(row['video_path'])}".encode()).hexdigest())
    return [buckets[key][index] for index in range(max(map(len, buckets.values())))
            for key in sorted(buckets) if index < len(buckets[key])][:count]


class SemanticDataset(Dataset):
    def __init__(self, rows, cache, skeleton=None):
        self.rows, self.cache, self.skeleton = rows, cache, skeleton
        self.index = {} if skeleton is None else {sample: index for index, sample in enumerate(skeleton["sample_ids"])}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        sample = _sample_id(row["video_path"])
        label = int(row["class_id"]) - 1
        paired = sample in self.index
        joints = self.skeleton["inputs"][self.index[sample]] if paired else torch.zeros(3, 64, 25, 2)
        return self.cache.load(Path(row["video_path"])), label, joints, paired


def loader(dataset, args, shuffle=False):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers,
                      pin_memory=True, persistent_workers=args.workers > 0, worker_init_fn=_worker_seed,
                      generator=torch.Generator().manual_seed(args.seed))


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required; do not create a pending run lock")
    set_seed(args.seed)
    torch.set_num_threads(8)
    root = args.artifacts_dir
    if (root / "experiment_lock.json").exists():
        raise ValueError("Run already locked; use a fresh directory")
    split = json.loads(args.split_manifest.read_text())
    paired = json.loads(args.paired_manifest.read_text())
    if (len(split["train"]), len(split["val"])) != (80640, 12960) or split["max_seq_len"] != 8 or split["frame_sampling_strategy"] != FRAME_SAMPLING_STRATEGY:
        raise ValueError("RGB split/sampling changed")
    source = torch.load(args.init_adapter, map_location="cpu", weights_only=False)
    if (source["identity"].get("split_sha256") != sha256_file(args.split_manifest)
            or source["identity"].get("test_evaluated") is not False
            or source["identity"].get("model_revision") != MODEL_REVISION
            or source["identity"].get("arm") != "vision"):
        raise ValueError("RGB warm-start identity changed")
    teacher_source = torch.load(args.teacher, map_location="cpu", weights_only=False)
    identity = teacher_source["identity"]
    class_ids = [f"A{index:03d}" for index in range(1, 60)]
    if (identity["head"] != "xclip" or identity["backbone"] != "blockgcn"
            or identity["ordered_class_ids"] != class_ids or identity["frames"] != 64
            or identity["manifest_sha256"] != sha256_file(args.paired_manifest)
            or identity["preprocessing_id"] != SKELETON_TENSOR_PREPROCESSING_ID
            or identity["text"]["model_revision"] != MODEL_REVISION):
        raise ValueError("Text-aligned skeleton teacher identity changed")
    skeleton = {}
    paired_ids = set()
    for part in ("train", "val"):
        rows = paired[part]
        payload = torch.load(args.skeleton_cache / f"{part}_45236_4180_64.pt", map_location="cpu", weights_only=True, mmap=True)
        expected_ids = [row["sample_id"] for row in rows]
        expected_labels = torch.tensor([row["class_index"] for row in rows])
        if (payload["sample_ids"] != expected_ids or len(set(expected_ids)) != len(expected_ids)
                or payload["preprocessing_id"] != SKELETON_TENSOR_PREPROCESSING_ID
                or not torch.equal(payload["labels"], expected_labels)
                or payload["inputs"].shape != (len(rows), 3, 64, 25, 2)):
            raise ValueError("Skeleton cache identity/labels differ")
        rgb = {_sample_id(row["video_path"]): row for row in split[part]}
        if len(rgb) != len(split[part]):
            raise ValueError("Duplicate RGB IDs")
        for row in rows:
            match = rgb.get(row["sample_id"])
            if match is None or int(match["class_id"]) - 1 != row["class_index"] or row["class_id"] not in class_ids:
                raise ValueError("Skeleton/RGB split or action mapping differs")
        # Refresh representatives against the raw source, not just cache metadata.
        for index in (0, len(rows) // 2, len(rows) - 1):
            fresh = load_skeleton_tensor(Path(rows[index]["skeleton_path"]), frames=64)
            if not torch.equal(fresh, payload["inputs"][index]):
                raise ValueError("Skeleton cache/raw preprocessing differs")
        skeleton[part] = payload
        paired_ids.update(expected_ids)
    if set(skeleton["train"]["sample_ids"]) & set(skeleton["val"]["sample_ids"]):
        raise ValueError("Train/validation leakage")
    baseline = torch.load(args.baseline_logits, map_location="cpu", weights_only=True)
    baseline_index = {sample: index for index, sample in enumerate(baseline["sample_ids"])}
    if len(baseline_index) != 12960 or set(baseline_index) != {_sample_id(row["video_path"]) for row in split["val"]}:
        raise ValueError("Saved RGB baseline does not cover canonical validation")
    train_rows = balanced(split["train"], 128 if args.mode == "smoke" else args.train_samples, args.seed)
    semantic_train_rows = [row for row in train_rows if _sample_id(row["video_path"]) in paired_ids]
    if args.mode == "smoke":
        semantic_train_rows = balanced([row for row in split["train"] if _sample_id(row["video_path"]) in paired_ids], 256, args.seed)
    semantic_val_rows = balanced([row for row in split["val"] if _sample_id(row["video_path"]) in paired_ids],
                                 118 if args.mode == "smoke" else 472, args.seed)
    val_rows = balanced(split["val"], 120, args.seed) if args.mode == "smoke" else split["val"]
    val_ids = [_sample_id(row["video_path"]) for row in val_rows]
    selected = torch.tensor([baseline_index[sample] for sample in val_ids])
    labels = torch.tensor([int(row["class_id"]) - 1 for row in val_rows])
    if not torch.equal(baseline["labels"][selected], labels):
        raise ValueError("Saved baseline validation labels differ")
    hashes = {"runner": sha256_file(Path(__file__)), "model": sha256_file(Path(__file__).parents[1] / "models/ski_semantic.py"),
              "rgb_model": sha256_file(Path(__file__).parents[1] / "models/xclip_vision_lora.py"),
              "teacher": sha256_file(args.teacher), "init_adapter": sha256_file(args.init_adapter),
              "split": sha256_file(args.split_manifest), "paired": sha256_file(args.paired_manifest),
              "baseline_logits": sha256_file(args.baseline_logits)}
    if args.mode == "pilot":
        smoke = json.loads(args.smoke_report.read_text())
        smoke_lock = json.loads((args.smoke_report.parent / "experiment_lock.json").read_text())
        if smoke["status"] != "pass" or smoke.get("test_evaluated") is not False or smoke_lock["source_sha256"] != hashes:
            raise ValueError("Current semantic implementation has not passed smoke")
    root.mkdir(parents=True, exist_ok=True)
    phase = root / "correct"
    phase.mkdir(exist_ok=True)
    def progress(state, **fields):
        record = {"state": state, "test_evaluated": False, **fields}
        write_json(phase / "progress.json", record)
        print(json.dumps(record), flush=True)
    lock = {"mode": args.mode, "seed": args.seed, "train_ids": [_sample_id(row["video_path"]) for row in train_rows],
            "validation_ids": val_ids, "semantic_train_samples": len(semantic_train_rows),
            "semantic_train_ids": [_sample_id(row["video_path"]) for row in semantic_train_rows],
            "semantic_validation_ids": [_sample_id(row["video_path"]) for row in semantic_val_rows],
            "source_sha256": hashes, "epochs": 1, "prealignment_epochs": 12, "batch_size": args.batch_size,
            "workers": args.workers, "precision": "fp32", "distillation_weight": 0.01,
            "loss": "CE120(all RGB) + CE59(video-text, paired) + CE59(skeleton-text, paired) + .01 MSE(scaled text-conditioned scores, paired)",
            "adaptations": ["supervised RGB120, partial paired59; not zero-shot replication", "shared frozen text prototypes",
                            "teacher BatchNorm statistics frozen for small batches; teacher parameters online",
                            "RGB vision/MIT LoRA only; no spatial head or pooling change",
                            "RGB semantic linear projection prealigned on frozen train features, fixed 12 epochs",
                            "temperature-scaled scores differ from official raw-cosine implementation"],
            "selection": "fixed final epoch; no weight/epoch search; full canonical validation for pilot",
            "rates": {"semantic_prealignment": 0.001, "vision_mit": 2.25e-5, "classifier": 2.25e-5,
                      "semantic_projection": 2.25e-5, "teacher": 2.25e-5},
            "test_evaluated": False}
    write_json(root / "experiment_lock.json", lock)
    (root / "runner_snapshot.py").write_text(Path(__file__).read_text())
    device = torch.device("cuda")
    prototypes = teacher_source["state_dict"]["text_prototypes"]
    teacher = SkeletonPrototypeTeacher("blockgcn", prototypes)
    teacher.load_state_dict(teacher_source["state_dict"], strict=True)
    teacher.to(device).eval()
    base = XCLIPModel.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=True)
    model = SKISemanticClassifier(base, text_prototypes=prototypes, semantic_scale=float(teacher.log_logit_scale.exp()),
                                  num_classes=120, head_dropout=0.25, feature_noise_std=0.01,
                                  upper_blocks=2, lora_rank=16, lora_alpha=32, lora_dropout=0.05).to(device)
    if source["model"]["target_modules"] != model.target_modules:
        raise ValueError("RGB LoRA targets differ")
    load_lora_state_dict(model, source["model"]["lora_state_dict"])
    model.classifier.load_state_dict(source["model"]["classifier_state_dict"], strict=True)
    processor = XCLIPProcessor.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=True)
    cache = ProcessedUInt8Cache(args.rgb_cache, processor, model_name=MODEL, model_revision=MODEL_REVISION)
    val_loader = loader(SemanticDataset(val_rows, cache), args)
    model.eval()
    pixels, label, _, _ = next(iter(val_loader))
    with torch.no_grad():
        initial = model(pixels.to(device)).cpu()
    parity = float((initial - baseline["logits"][selected[:len(initial)]]).abs().max())
    if parity > 1e-4:
        raise RuntimeError(f"Baseline RGB identity parity failed: {parity}")
    progress("baseline_identity_verified", max_abs_difference=parity)
    # Extract frozen RGB features once: prealignment is inexpensive and cannot alter RGB predictions.
    features = {}
    for part, rows in (("train", semantic_train_rows), ("val", semantic_val_rows)):
        xs, ys = [], []
        with torch.no_grad():
            for batch_index, (pixels, label, _, _) in enumerate(loader(SemanticDataset(rows, cache), args)):
                _, embedding = model(pixels.to(device), return_features=True)
                xs.append(embedding.cpu()); ys.append(label)
                if batch_index % 100 == 0:
                    progress("extract_semantic_features", split=part, samples=sum(len(x) for x in ys), total_samples=len(rows))
        features[part] = (torch.cat(xs).to(device), torch.cat(ys).to(device))
    def semantic_metrics(part):
        with torch.no_grad():
            x, y = features[part]
            logits = model.semantic_logits(x)
            return {"ce": float(F.cross_entropy(logits, y)), "accuracy": float(logits.argmax(-1).eq(y).float().mean())}
    before = semantic_metrics("val")
    optimizer = torch.optim.AdamW(model.semantic_projection.parameters(), lr=1e-3, weight_decay=5e-4)
    x, y = features["train"]
    for epoch in range(12):
        for indices in torch.randperm(len(y), device=device).split(128):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model.semantic_logits(x[indices]), y[indices])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite semantic prealignment")
            loss.backward(); optimizer.step()
        progress("semantic_prealignment", epoch=epoch + 1, train=semantic_metrics("train"), validation=semantic_metrics("val"))
    after = semantic_metrics("val")
    prealignment_pass = after["ce"] < before["ce"] and after["accuracy"] > 1 / 59
    write_json(root / "prealignment_report.json", {"before": before, "after": after, "pass": prealignment_pass,
                                                 "rgb_predictions_unchanged": True, "test_evaluated": False})
    if not prealignment_pass:
        return {"status": "prealignment_no_go", "raw_accuracy_gain": 0.0, "test_evaluated": False}
    train_loader = loader(SemanticDataset(train_rows, cache, skeleton["train"]), args, shuffle=True)
    # Pure KD backward verifies that semantic guidance actually reaches vision and the online skeleton branch.
    pixels, label, joints, mask = next(batch for batch in train_loader if batch[3].any())
    mask = mask.to(device)
    model.zero_grad(set_to_none=True); teacher.zero_grad(set_to_none=True)
    _, video_scores = model(pixels.to(device), return_semantic=True)
    skeleton_scores = teacher(joints.to(device)[mask])
    kd = F.mse_loss(video_scores[mask], skeleton_scores)
    kd.backward()
    gradient = {}
    for family, values in (("vision", [value for name, value in model.named_parameters() if name.startswith("vision_model.") and name.endswith(("lora_a", "lora_b"))]),
                           ("mit", [value for name, value in model.named_parameters() if name.startswith("mit.") and name.endswith(("lora_a", "lora_b"))]),
                           ("teacher", list(teacher.parameters()))):
        gradient[family] = sum(float(value.grad.abs().sum()) for value in values if value.grad is not None)
    if not all(value > 0 and torch.isfinite(torch.tensor(value)) for value in gradient.values()):
        raise RuntimeError(f"Semantic KD gradient did not reach required branches: {gradient}")
    progress("semantic_gradient_verified", kd=float(kd.detach()), gradients=gradient)
    model.zero_grad(set_to_none=True); teacher.zero_grad(set_to_none=True)
    parameters = [value for value in model.parameters() if value.requires_grad] + list(teacher.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=2.25e-5, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader))
    model.train()
    teacher.eval()  # BN running statistics stay fixed; all parameters still receive gradients.
    totals = {key: 0.0 for key in ("ce120", "video_ce59", "skeleton_ce59", "kd")}
    steps = seen = paired_seen = 0
    started = time.monotonic()
    for pixels, label, joints, mask in train_loader:
        pixels, label, mask = pixels.to(device), label.to(device), mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, video_scores = model(pixels, return_semantic=True)
        losses = {"ce120": F.cross_entropy(logits, label, label_smoothing=0.1)}
        if mask.any():
            if not bool((label[mask] < 59).all()):
                raise RuntimeError("Paired label outside semantic taxonomy")
            skeleton_scores = teacher(joints.to(device)[mask])
            losses.update(video_ce59=F.cross_entropy(video_scores[mask], label[mask]),
                          skeleton_ce59=F.cross_entropy(skeleton_scores, label[mask]),
                          kd=F.mse_loss(video_scores[mask], skeleton_scores))
        loss = sum(value * (0.01 if key == "kd" else 1) for key, value in losses.items())
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite collaborative loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step(); scheduler.step()
        for key, value in losses.items():
            totals[key] += float(value.detach())
        steps += 1; seen += len(label); paired_seen += int(mask.sum())
        if steps % 25 == 0 or seen == len(train_rows):
            progress("collaborative_training", samples=seen, total_samples=len(train_rows), paired_samples=paired_seen,
                     loss_means={key: value / steps for key, value in totals.items()}, elapsed_seconds=time.monotonic() - started)
    model.eval()
    predictions = []
    with torch.no_grad():
        for pixels, _, _, _ in val_loader:
            predictions.append(model(pixels.to(device)).cpu())
            count = sum(len(value) for value in predictions)
            if count % (args.batch_size * 100) == 0 or count == len(val_rows):
                progress("validation", samples=count, total_samples=len(val_rows))
    logits = torch.cat(predictions)
    if not torch.isfinite(logits).all():
        raise RuntimeError("Nonfinite RGB validation logits")
    classification = classification_report(logits, labels, val_ids, paired_ids)
    baseline_report = classification_report(baseline["logits"][selected], labels, val_ids, paired_ids)
    write_json(root / "classification_report.json", classification)
    write_json(root / "baseline_report.json", baseline_report)
    torch.save({"logits": logits, "labels": labels, "sample_ids": val_ids, "test_evaluated": False}, root / "validation_predictions.pt")
    torch.save({"lock": lock, "lora": lora_state_dict(model), "classifier": model.classifier.state_dict(),
                "semantic_projection": model.semantic_projection.state_dict(), "teacher": teacher.state_dict()}, root / "last_checkpoint.pt")
    pixels = next(iter(val_loader))[0].to(device)
    with torch.no_grad():
        reference = model(pixels)
    merge_lora(model)
    merged = {key: value for key, value in merged_base_state_dict(model).items()
              if key not in ("text_prototypes", "semantic_scale") and not key.startswith("semantic_projection.")}
    restored = XCLIPVisionClassifier(XCLIPModel.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=True),
                                    num_classes=120, head_dropout=0.25, feature_noise_std=0.01)
    restored.load_state_dict(merged, strict=True)
    restored.to(device).eval()
    with torch.no_grad():
        reload_error = float((restored(pixels) - reference).abs().max())
    if reload_error > 1e-4:
        raise RuntimeError(f"Merged RGB-only reload differs: {reload_error}")
    torch.save(merged, root / "merged_rgb_state.pt")
    gain = classification["raw_accuracy"] - baseline_report["raw_accuracy"]
    status = "pass" if args.mode == "smoke" else "promising" if gain >= 0.002 else "no_go" if gain < 0 else "inconclusive"
    return {"status": status, "mode": args.mode, "raw_accuracy_gain": gain,
            "classification": classification, "baseline": baseline_report, "semantic_prealignment": {"before": before, "after": after},
            "kd_gradients": gradient, "baseline_identity_max_abs_diff": parity, "rgb_only_reload_max_abs_diff": reload_error,
            "paired_train_samples": paired_seen, "train_samples": seen, "test_evaluated": False,
            "interpretation": "Smoke certifies mechanism only; pilot is a bounded supervised120/partial59 SKI-style adaptation, not paper reproduction."}


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
             "rgb_cache": "cache/skeletonclip_phase3_rgb_uint8", "smoke_report": "artifacts/ski_semantic_v1/smoke/run_report.json",
             "init_adapter": "artifacts/skeletonclip_phase3/vision_batch16_workers24_uint8_seed42/vision_lora_ce_seed42/best_adapter.pt",
             "baseline_logits": "artifacts/skeletonclip_phase3/vision_batch16_workers24_uint8_seed42/vision_lora_ce_seed42/best_validation_logits.pt"}
    for name, path in paths.items():
        parser.add_argument("--" + name.replace("_", "-"), type=Path, default=Path(path))
    args = parser.parse_args()
    if not 0 <= args.workers <= 24 or args.batch_size < 1 or args.train_samples < 120:
        parser.error("Require workers 0-24, positive batch size and at least 120 train samples")
    report = run(args)
    write_json(args.artifacts_dir / "run_report.json", report)
    print(json.dumps({key: value for key, value in report.items() if key not in ("classification", "baseline")}), flush=True)


if __name__ == "__main__":
    main()
