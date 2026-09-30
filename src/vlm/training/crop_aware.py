"""Short raw120 whole/zoom fine-tuning pilot; no teacher or alarm policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import XCLIPModel, XCLIPProcessor

from vlm.models.lora import load_lora_state_dict, lora_state_dict, merge_lora, merged_base_state_dict
from vlm.models.xclip_vision_lora import XCLIPVisionClassifier, XCLIPVisionLoRAClassifier
from vlm.reporting import write_json
from vlm.skeleton_teacher_cache import sha256_file
from vlm.training.rgb_input_audit import ViewDataset, action_names, saved_prediction_audit, PRODUCT_EXCLUDED_IDS
from vlm.training.ski_semantic import MODEL, MODEL_REVISION, balanced
from vlm.training.skeletonclip_phase4_gate2 import _sample_id
from vlm.xclip_features import FRAME_SAMPLING_STRATEGY

VIEWS = ("whole", "center_zoom70")


class MixedViews(Dataset):
    def __init__(self, rows, processor, seed):
        self.rows, self.processor, self.seed = rows, processor, seed
        self.zoom = [int(hashlib.sha256(f"crop{seed}:{_sample_id(r['video_path'])}".encode()).hexdigest(), 16) % 2 for r in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        view = VIEWS[self.zoom[index]]
        pixels, label = ViewDataset([self.rows[index]], self.processor, (view,))[0]
        return pixels[0], label, self.zoom[index]


def save_strip(rows, processor, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pixels, _ = ViewDataset(rows, processor, VIEWS)[0]
    mean = torch.tensor(processor.image_processor.image_mean).view(3, 1, 1)
    std = torch.tensor(processor.image_processor.image_std).view(3, 1, 1)
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for v in range(2):
        for frame in range(8):
            rgb = (pixels[v, frame] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
            axes[v, frame].imshow(rgb); axes[v, frame].axis("off")
    fig.suptitle(f"{_sample_id(rows[0]['video_path'])}: whole(top), zoom(bottom)")
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def run(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    root = args.artifacts_dir
    if (root / "experiment_lock.json").exists():
        raise ValueError("Use a new artifact directory; locked runs cannot be overwritten")
    if args.mode == "pilot":
        smoke = json.loads(args.smoke_report.read_text())
        if (smoke["status"] != "pass" or smoke["runner_sha256"] != sha256_file(Path(__file__))
                or smoke["source_sha256"]["view_runner"] != sha256_file(Path(__file__).with_name("rgb_input_audit.py"))):
            raise ValueError("Pilot requires passing smoke from this exact runner")
    split = json.loads(args.split_manifest.read_text())
    if (len(split["train"]), len(split["val"])) != (80640, 12960) or split["max_seq_len"] != 8 or split["frame_sampling_strategy"] != FRAME_SAMPLING_STRATEGY:
        raise ValueError("Canonical split/sampling changed")
    source = torch.load(args.adapter, map_location="cpu", weights_only=False)
    reference = torch.load(args.reference_predictions, map_location="cpu", weights_only=True)
    if source["identity"]["split_sha256"] != sha256_file(args.split_manifest) or source["identity"]["model_revision"] != MODEL_REVISION or source["identity"]["test_evaluated"] is not False:
        raise ValueError("Adapter provenance differs")
    index = {s: i for i, s in enumerate(reference["sample_ids"])}
    if len(index) != 12960 or set(index) != {_sample_id(r["video_path"]) for r in split["val"]}:
        raise ValueError("Full whole/zoom reference IDs differ")
    train_rows = balanced(split["train"], 120 if args.mode == "smoke" else args.train_samples, args.seed)
    val_rows = balanced(split["val"], 120, args.seed) if args.mode == "smoke" else split["val"]
    ids = [_sample_id(r["video_path"]) for r in val_rows]
    selected = torch.tensor([index[s] for s in ids])
    labels = torch.tensor([int(r["class_id"]) - 1 for r in val_rows])
    if not torch.equal(reference["labels"][selected], labels):
        raise ValueError("Reference labels differ")
    root.mkdir(parents=True, exist_ok=True)
    phase = root / args.mode
    phase.mkdir(exist_ok=True)
    def progress(state, **fields):
        record = {"state": state, "test_evaluated": False, **fields}
        write_json(phase / "progress.json", record)
        print(json.dumps(record), flush=True)
    processor = XCLIPProcessor.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=True)
    training = MixedViews(train_rows, processor, args.seed)
    hashes = {
        "runner": sha256_file(Path(__file__)), "view_runner": sha256_file(Path(__file__).with_name("rgb_input_audit.py")),
        "adapter": sha256_file(args.adapter), "reference_predictions": sha256_file(args.reference_predictions),
        "split": sha256_file(args.split_manifest), "names": sha256_file(args.action_names)}
    lock = {"mode": args.mode, "seed": args.seed, "epochs": 1, "source_sha256": hashes,
            "train_ids": [_sample_id(r["video_path"]) for r in train_rows], "validation_ids": ids,
            "train_zoom_mask": training.zoom, "views": list(VIEWS), "batch_size": args.batch_size, "workers": args.workers,
            "learning_rate": 1e-5, "weight_decay": 5e-4, "label_smoothing": .1, "gradient_clip": 1.0,
            "training_policy": "One pass on fixed balanced subset; per-ID seed hash chooses one whole or zoom view (~50/50); same frames, label unchanged",
            "student_policy": "Only upper2 vision/MIT q/v LoRA and classifier trainable; eval mode disables dropout/noise while gradients enabled",
            "validation_policy": "Fixed final checkpoint; whole, zoom, equal probability mean on canonical validation; no epoch/crop/weight search",
            "selection": "Promising if ensemble improves >=0.20 percentage points over original ensemble and attention top1 does not regress; single-view goal reported separately",
            "negative_policy": "All120 retained; product exclusions are report-only, not filtering",
            "threshold_runtime_changes": False, "test_evaluated": False}
    write_json(root / "experiment_lock.json", lock)
    (root / "runner_snapshot.py").write_text(Path(__file__).read_text())
    (root / "view_runner_snapshot.py").write_text(Path(__file__).with_name("rgb_input_audit.py").read_text())
    # Visual evidence from original validation regressions, not training sample selection.
    lookup = {_sample_id(r["video_path"]): r for r in split["val"]}
    for sample in ("S011C001P038R001A043", "S006C003P022R002A048", "S007C001P027R001A047"):
        save_strip([lookup[sample]], processor, root / f"{sample}_strip.png")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    model = XCLIPVisionLoRAClassifier(XCLIPModel.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=True),
                                     num_classes=120, head_dropout=.25, feature_noise_std=.01,
                                     upper_blocks=2, lora_rank=16, lora_alpha=32, lora_dropout=.05).cuda().eval()
    if model.target_modules != source["model"]["target_modules"]:
        raise ValueError("LoRA target modules differ")
    load_lora_state_dict(model, source["model"]["lora_state_dict"])
    model.classifier.load_state_dict(source["model"]["classifier_state_dict"], strict=True)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if any(not (n.startswith("classifier.") or n.endswith(("lora_a", "lora_b"))) for n, _ in named):
        raise RuntimeError("Unexpected trainable base parameters")
    write_json(root / "trainable_parameters.json", {"names": [n for n, _ in named], "count": sum(p.numel() for _, p in named)})
    validation = DataLoader(ViewDataset(val_rows, processor, VIEWS), batch_size=args.batch_size, num_workers=args.workers,
                            pin_memory=True, persistent_workers=args.workers > 0)
    pixels, _ = next(iter(validation))
    parity = {}
    with torch.no_grad():
        for v, name in enumerate(VIEWS):
            parity[name] = float((model(pixels[:, v].cuda()).cpu() - reference["logits"][name][selected[:len(pixels)]]).abs().max())
    if max(parity.values()) > 1e-3:
        raise RuntimeError(f"Warm-start parity failed: {parity}")
    progress("warm_start_verified", errors=parity)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, generator=generator,
                        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    parameters = [p for _, p in named]
    optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=5e-4)
    steps_limit = 2 if args.mode == "smoke" else len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps_limit)
    seen = zoom_seen = 0
    started = time.monotonic()
    gradients = {}
    for step, (pixels, y, zoom) in enumerate(loader, 1):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(pixels.cuda()), y.cuda(), label_smoothing=.1)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        if step == 1:
            gradients = {family: sum(float(p.grad.abs().sum()) for n, p in named if n.startswith(prefix) and p.grad is not None)
                         for family, prefix in (("vision", "vision_model."), ("mit", "mit."), ("classifier", "classifier."))}
            if not all(v > 0 and torch.isfinite(torch.tensor(v)) for v in gradients.values()):
                raise RuntimeError(f"Missing/nonfinite gradient: {gradients}")
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step(); scheduler.step()
        seen += len(y); zoom_seen += int(zoom.sum())
        if step % 25 == 0 or step == steps_limit:
            progress("crop_training", step=step, total_steps=steps_limit, samples=seen,
                     zoom_samples=zoom_seen, whole_samples=seen-zoom_seen, loss=float(loss.detach()), elapsed_seconds=time.monotonic()-started)
        if step == steps_limit:
            break
    torch.save({"lock": lock, "lora": lora_state_dict(model), "classifier": model.classifier.state_dict()}, root / "last_checkpoint.pt")
    outputs = {v: [] for v in VIEWS}
    ys = []
    with torch.inference_mode():
        for batch, (pixels, y) in enumerate(validation, 1):
            for v, name in enumerate(VIEWS):
                outputs[name].append(model(pixels[:, v].cuda()).cpu())
            ys.append(y)
            if batch % 25 == 0 or batch == len(validation):
                progress("validation", samples=sum(len(t) for t in ys), total_samples=len(val_rows))
    if not torch.equal(torch.cat(ys), labels):
        raise RuntimeError("Validation order changed")
    logits = {k: torch.cat(v) for k, v in outputs.items()}
    if any(not torch.isfinite(z).all() for z in logits.values()):
        raise RuntimeError("Nonfinite validation logits")
    torch.save({"logits": logits, "labels": labels, "sample_ids": ids, "test_evaluated": False}, root / "validation_predictions.pt")
    def scores(z):
        p = {k: v.softmax(-1) for k, v in z.items()}
        p["whole_zoom_mean"] = (p["whole"] + p["center_zoom70"]) / 2
        return p
    old = scores({k: v[selected] for k, v in reference["logits"].items()})
    new = scores(logits)
    attention = torch.isin(labels, torch.arange(41, 48))
    excluded = torch.isin(labels, torch.tensor(PRODUCT_EXCLUDED_IDS)-1)
    results = {}
    names = action_names(args.action_names)
    for view, p in new.items():
        ok, prior = p.argmax(-1).eq(labels), old[view].argmax(-1).eq(labels)
        results[view] = {"accuracy": float(ok.float().mean()), "original_accuracy": float(prior.float().mean()),
                         "correct": int(ok.sum()), "fixed": int((ok & ~prior).sum()), "regressed": int((~ok & prior).sum()),
                         "attention_correct": int(ok[attention].sum()), "original_attention_correct": int(prior[attention].sum()),
                         "attention_samples": int(attention.sum()), "excluded24_correct": int(ok[excluded].sum())}
        if args.mode == "pilot":
            write_json(root / f"{view}_confusion_alert_report.json", saved_prediction_audit({"logits": p.clamp_min(torch.finfo(p.dtype).tiny).log(), "labels": labels, "sample_ids": ids}, names))
    pixels, _ = next(iter(validation))
    pixels = pixels[:, 0].cuda()
    with torch.no_grad():
        before = model(pixels)
    merge_lora(model)
    merged = merged_base_state_dict(model)
    restored = XCLIPVisionClassifier(XCLIPModel.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=True),
                                    num_classes=120, head_dropout=.25, feature_noise_std=.01).cuda().eval()
    restored.load_state_dict(merged, strict=True)
    with torch.no_grad():
        reload_error = float((restored(pixels)-before).abs().max())
    if reload_error > 1e-4:
        raise RuntimeError("Merged strict reload parity failed")
    torch.save(merged, root / "merged_rgb_state.pt")
    r = results["whole_zoom_mean"]
    gain = r["accuracy"] - r["original_accuracy"]
    promising = gain >= .002 and r["attention_correct"] >= r["original_attention_correct"]
    return {"status": "pass" if args.mode == "smoke" else "promising" if promising else "no_go" if gain < 0 else "inconclusive",
            "raw_accuracy_gain": gain, "mode": args.mode, "runner_sha256": hashes["runner"], "source_sha256": hashes,
            "results": results, "baseline_identity_errors": parity, "gradients": gradients,
            "merged_reload_max_abs_diff": reload_error, "train_samples": seen, "zoom_train_samples": zoom_seen,
            "whole_train_samples": seen-zoom_seen, "single_view_target_accuracy": float(old["whole_zoom_mean"].argmax(-1).eq(labels).float().mean()),
            "test_evaluated": False, "interpretation": "Fixed short crop-aware pilot, not a control-isolated augmentation attribution or deployment/medical approval"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    for name, path in {"split_manifest": "artifacts/xclip/selected_split.json",
                       "adapter": "artifacts/skeletonclip_phase3/vision_batch16_workers24_uint8_seed42/vision_lora_ce_seed42/best_adapter.pt",
                       "reference_predictions": "artifacts/whole_zoom_validation_v1/view_predictions.pt",
                       "action_names": "docs/xclip-vision-lora-validation-action-accuracy.md",
                       "smoke_report": "artifacts/crop_aware_v1_smoke/run_report.json"}.items():
        parser.add_argument("--"+name.replace('_','-'), type=Path, default=Path(path))
    args = parser.parse_args()
    if not 0 <= args.workers <= 24 or not 120 <= args.train_samples <= 80640 or args.batch_size < 1:
        parser.error("Require workers0-24, train_samples120-80640 and positive batch size")
    report = run(args)
    write_json(args.artifacts_dir / "run_report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
