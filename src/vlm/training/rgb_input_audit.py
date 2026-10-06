"""No-training RGB-view diagnostic, raw120 confusion ranking and alert audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import av
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import XCLIPModel, XCLIPProcessor

from vlm.action_taxonomy import IRRELEVANT_GROUPS
from vlm.experiments.video_sampling import read_video_pyav, uniform_frame_indices
from vlm.models.lora import load_lora_state_dict
from vlm.models.xclip_vision_lora import XCLIPVisionLoRAClassifier
from vlm.reporting import write_json
from vlm.skeleton_teacher_cache import sha256_file
from vlm.training.ski_semantic import MODEL, MODEL_REVISION, balanced
from vlm.training.skeletonclip_phase4_gate2 import _sample_id
from vlm.xclip_features import FRAME_SAMPLING_STRATEGY

VIEWS = ("whole", "early", "middle", "late", "center_zoom70")
ATTENTION_IDS = tuple(range(42, 49))
PRODUCT_EXCLUDED_IDS = tuple(sorted({cls for group in IRRELEVANT_GROUPS.values() for cls in group} | {86, 34, 11, 78, 12, 110, 4, 30, 89, 90, 67, 10, 32, 79, 23, 107, 106}))


def action_names(path: Path) -> dict[int, str]:
    names = {}
    for line in path.read_text().splitlines():
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) > 4 and cells[2].startswith("A") and cells[2][1:].isdigit():
            names[int(cells[2][1:])] = cells[3]
    if set(names) != set(range(1, 121)):
        raise ValueError("Reviewed action-name table must cover all120 classes")
    return names


def binary_metrics(score, truth, threshold):
    prediction = score >= threshold
    tp = int((prediction & truth).sum()); fn = int((~prediction & truth).sum())
    fp = int((prediction & ~truth).sum()); tn = int((~prediction & ~truth).sum())
    return {"threshold": threshold, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "recall": tp / max(1, tp + fn), "precision": tp / max(1, tp + fp),
            "false_positive_rate": fp / max(1, fp + tn)}


def recall_threshold(score, truth, target=0.95):
    values = score[truth].sort().values
    if not len(values):
        raise ValueError("No positive calibration samples")
    return float(values[len(values) - math.ceil(target * len(values))])


def saved_prediction_audit(payload, names):
    logits, labels, ids = payload["logits"], payload["labels"], payload["sample_ids"]
    prediction = logits.argmax(-1)
    confusion = torch.bincount(labels * 120 + prediction, minlength=14400).reshape(120, 120)
    ignored = {cls for group in IRRELEVANT_GROUPS.values() for cls in group}
    rows = []
    for cls in range(120):
        n = int(confusion[cls].sum()); correct = int(confusion[cls, cls])
        wrong = confusion[cls].clone(); wrong[cls] = 0
        counts, indices = wrong.topk(3)
        rows.append({"class_id": cls + 1, "name": names[cls + 1], "samples": n,
                     "correct": correct, "accuracy": correct / n, "error_rate": 1 - correct / n,
                     "previously_ignored": cls + 1 in ignored,
                     "top_confusions": [{"class_id": int(i) + 1, "name": names[int(i) + 1], "count": int(v)}
                                        for i, v in zip(indices, counts) if v > 0]})
    rows.sort(key=lambda row: (row["accuracy"], row["class_id"]))
    cal = torch.zeros(len(labels), dtype=torch.bool)
    for cls in range(120):
        indices = labels.eq(cls).nonzero().flatten().tolist()
        indices.sort(key=lambda i: hashlib.sha256(f"alert-calibration42:{ids[i]}".encode()).hexdigest())
        cal[indices[:len(indices) // 2]] = True
    evaluation = ~cal
    probabilities = logits.softmax(-1)
    attention_indices = torch.tensor([cls - 1 for cls in ATTENTION_IDS])
    truth = torch.isin(labels, attention_indices)
    score = probabilities[:, attention_indices].sum(-1)
    threshold = recall_threshold(score[cal], truth[cal])
    events = []
    for cls in ATTENTION_IDS:
        t, s = labels.eq(cls - 1), probabilities[:, cls - 1]
        th = recall_threshold(s[cal], t[cal])
        events.append({"class_id": cls, "name": names[cls], "samples": int(t.sum()),
                       "top1_correct": int(prediction[t].eq(cls - 1).sum()),
                       "top1_recall": float(prediction[t].eq(cls - 1).float().mean()),
                       "calibrated95": binary_metrics(s[evaluation], t[evaluation], th),
                       "fixed05": binary_metrics(s[evaluation], t[evaluation], 0.5)})
    return {"ranking_all120": rows, "top20": rows[:20], "attention_events": events,
            "attention_source_class_ids": list(ATTENTION_IDS),
            "internal_split": {"calibration_samples": int(cal.sum()), "evaluation_samples": int(evaluation.sum()),
                               "calibration_ids": [s for s, keep in zip(ids, cal.tolist()) if keep],
                               "evaluation_ids": [s for s, keep in zip(ids, evaluation.tolist()) if keep],
                               "policy": "within-class sample-ID hash halves; exploratory only, baseline already selected on validation; related views/subjects may cross halves"},
            "group_fixed_thresholds_evaluation": [binary_metrics(score[evaluation], truth[evaluation], th) for th in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)],
            "group_calibrated95_evaluation": binary_metrics(score[evaluation], truth[evaluation], threshold),
            "group_fixed_thresholds_full_validation": [binary_metrics(score, truth, th) for th in (0.1, 0.5)],
            "scope": "attention-associated acted NTU poses, not clinical danger or streaming alarms",
            "test_evaluated": False}


class ViewDataset(Dataset):
    def __init__(self, rows, processor, views=VIEWS):
        self.rows, self.processor = rows, processor
        self.views = views

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with av.open(row["video_path"]) as container:
            count = container.streams.video[0].frames
            if count < 8:
                raise ValueError("Unsupported frame count")
            indices = {"whole": uniform_frame_indices(8, count)}
            for name, start, end in (("early", 0, .5), ("middle", .25, .75), ("late", .5, 1)):
                if name in self.views:
                    indices[name] = np.linspace(start * (count - 1), end * (count - 1), 8).round().astype(np.int64)
            all_indices = np.unique(np.concatenate(list(indices.values())))
            frames = read_video_pyav(container, all_indices)
        lookup = {int(i): frame for i, frame in zip(all_indices, frames)}
        pixels = []
        for name in self.views:
            view = np.stack([lookup[int(i)] for i in indices["whole" if name == "center_zoom70" else name]])
            if name == "center_zoom70":
                h, w = view.shape[1:3]
                side = round(min(h, w) * .7)
                top, left = (h - side) // 2, (w - side) // 2
                view = view[:, top:top + side, left:left + side]
            pixels.append(self.processor(images=list(view), return_tensors="pt")["pixel_values"].squeeze(0))
        return torch.stack(pixels), int(row["class_id"]) - 1


def run(args):
    root = args.artifacts_dir
    if (root / "experiment_lock.json").exists():
        raise ValueError("Audit already locked; use a new directory")
    split = json.loads(args.split_manifest.read_text())
    if (len(split["train"]), len(split["val"])) != (80640, 12960) or split["max_seq_len"] != 8 or split["frame_sampling_strategy"] != FRAME_SAMPLING_STRATEGY:
        raise ValueError("Canonical split/sampling changed")
    names = action_names(args.action_names)
    baseline = torch.load(args.baseline_logits, map_location="cpu", weights_only=True)
    index = {sample: i for i, sample in enumerate(baseline["sample_ids"])}
    if len(index) != 12960 or set(index) != {_sample_id(row["video_path"]) for row in split["val"]}:
        raise ValueError("Baseline IDs differ from validation")
    source = torch.load(args.adapter, map_location="cpu", weights_only=False)
    # The adapter's own identity selects the XCLIP backbone (patch32 or patch16).
    model_name = source["identity"].get("model_name", MODEL)
    model_revision = source["identity"]["model_revision"]
    if model_name == MODEL and model_revision != MODEL_REVISION:
        raise ValueError("Baseline checkpoint provenance changed")
    if source["identity"]["split_sha256"] != sha256_file(args.split_manifest) or source["identity"]["test_evaluated"] is not False:
        raise ValueError("Baseline checkpoint provenance changed")
    if args.samples < 1 or args.samples > 12960 or not 0 <= args.workers <= 24:
        raise ValueError("Invalid sample count or workers (maximum24)")
    views = ("whole", "center_zoom70") if args.whole_zoom_only else VIEWS
    rows = list(split["val"]) if args.full_validation else balanced(split["val"], args.samples, 42)
    ids = [_sample_id(row["video_path"]) for row in rows]
    selected = torch.tensor([index[sample] for sample in ids])
    labels = torch.tensor([int(row["class_id"]) - 1 for row in rows])
    if not torch.equal(labels, baseline["labels"][selected]):
        raise ValueError("Baseline labels differ")
    root.mkdir(parents=True, exist_ok=True)
    lock = {"purpose": "read-only inference diagnostic; no optimizer", "samples": len(rows), "sample_ids": ids,
            "model_name": model_name, "model_revision": model_revision,
            "views": list(views), "view_policy": "whole-video uniform8; fixed70%-short-side center zoom; half-clip windows only if requested; not person detection",
            "ensemble_policy": "equal mean whole+zoom; equal mean temporal windows only if requested",
            "source_sha256": {"runner": sha256_file(Path(__file__)), "adapter": sha256_file(args.adapter),
                              "baseline_logits": sha256_file(args.baseline_logits), "split": sha256_file(args.split_manifest),
                              "names": sha256_file(args.action_names)},
            "attention_source_class_ids": list(ATTENTION_IDS), "training": False, "test_evaluated": False,
            "selection": "all canonical validation" if args.full_validation else f"fixed{args.samples} seed42 balanced selection independent of correctness",
            "full_validation": args.full_validation, "batch_size": 4, "workers": args.workers,
            "product_excluded_source_class_ids": list(PRODUCT_EXCLUDED_IDS),
            "negative_policy": "all120 probabilities and samples retained, no exclusion/renormalization"}
    write_json(root / "experiment_lock.json", lock)
    (root / "runner_snapshot.py").write_text(Path(__file__).read_text())
    write_json(root / "confusion_alert_report.json", saved_prediction_audit(baseline, names))
    if not torch.cuda.is_available():
        raise RuntimeError("GPU needed for bounded RGB-view diagnostic")
    torch.set_num_threads(8)
    processor = XCLIPProcessor.from_pretrained(model_name, revision=model_revision, local_files_only=True)
    model = XCLIPVisionLoRAClassifier(XCLIPModel.from_pretrained(model_name, revision=model_revision, local_files_only=True),
                                     num_classes=120, head_dropout=.25, feature_noise_std=.01,
                                     upper_blocks=2, lora_rank=16, lora_alpha=32, lora_dropout=.05).cuda().requires_grad_(False).eval()
    if model.target_modules != source["model"]["target_modules"]:
        raise ValueError("Adapter LoRA targets differ")
    load_lora_state_dict(model, source["model"]["lora_state_dict"])
    model.classifier.load_state_dict(source["model"]["classifier_state_dict"], strict=True)
    data = DataLoader(ViewDataset(rows, processor, views), batch_size=4, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    outputs = [[] for _ in views]
    forward_seconds = {name: 0.0 for name in views}
    labels_all = []
    phase = root / "inference"
    phase.mkdir(exist_ok=True)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (pixels, batch_labels) in enumerate(data):
            for view_index, view_name in enumerate(views):
                torch.cuda.synchronize()
                view_started = time.perf_counter()
                outputs[view_index].append(model(pixels[:, view_index].cuda()).cpu())
                forward_seconds[view_name] += time.perf_counter() - view_started
            labels_all.append(batch_labels)
            if batch_index % 10 == 0 or batch_index + 1 == len(data):
                record = {"state": "inference_only", "samples": sum(len(y) for y in labels_all), "total_samples": len(rows), "elapsed_seconds": time.perf_counter() - started, "training": False, "test_evaluated": False}
                write_json(phase / "progress.json", record)
                print(json.dumps(record), flush=True)
    if not torch.equal(torch.cat(labels_all), labels):
        raise ValueError("Inference labels reordered")
    inference_seconds = time.perf_counter() - started
    logits = {name: torch.cat(values) for name, values in zip(views, outputs)}
    if any(not torch.isfinite(z).all() or z.shape != (len(rows), 120) for z in logits.values()):
        raise RuntimeError("Invalid inference logits")
    parity = float((logits["whole"] - baseline["logits"][selected]).abs().max())
    disagreements = int((logits["whole"].argmax(-1) != baseline["logits"][selected].argmax(-1)).sum())
    if parity > 1e-3 or disagreements:
        raise RuntimeError(f"Baseline inference parity failed: {parity}, {disagreements}")
    probabilities = {name: z.softmax(-1) for name, z in logits.items()}
    if all(name in probabilities for name in ("early", "middle", "late")):
        probabilities["temporal_mean"] = sum(probabilities[name] for name in ("early", "middle", "late")) / 3
    probabilities["whole_zoom_mean"] = (probabilities["whole"] + probabilities["center_zoom70"]) / 2
    old_ok = probabilities["whole"].argmax(-1).eq(labels)
    ignored = torch.tensor([cls - 1 for group in IRRELEVANT_GROUPS.values() for cls in group])
    masks = {"A001_A060": labels < 60, "A061_A120": labels >= 60, "previously_ignored": torch.isin(labels, ignored),
             "attention_A042_A048": torch.isin(labels, torch.tensor([cls - 1 for cls in ATTENTION_IDS])),
             f"product_excluded{len(PRODUCT_EXCLUDED_IDS)}": torch.isin(labels, torch.tensor([cls - 1 for cls in PRODUCT_EXCLUDED_IDS]))}
    results = {}
    for name, probs in probabilities.items():
        ok = probs.argmax(-1).eq(labels)
        results[name] = {"accuracy": float(ok.float().mean()), "correct": int(ok.sum()),
                         "fixed": int((ok & ~old_ok).sum()), "regressed": int((~ok & old_ok).sum()),
                         "groups": {tag: {"n": int(mask.sum()), "correct": int(ok[mask].sum()), "baseline_correct": int(old_ok[mask].sum())} for tag, mask in masks.items()}}
    torch.save({"logits": logits, "labels": labels, "sample_ids": ids, "training": False, "test_evaluated": False}, root / "view_predictions.pt")
    if args.full_validation:
        for name, probs in probabilities.items():
            write_json(root / f"{name}_confusion_alert_report.json",
                       saved_prediction_audit({"logits": probs.clamp_min(torch.finfo(probs.dtype).tiny).log(), "labels": labels, "sample_ids": ids}, names))
    # Evidence strips show actual post-processor pixels; no external image model.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    wanted = {43, 48, 72, 74, 84, 89}
    for row in rows:
        cls = int(row["class_id"])
        if cls not in wanted:
            continue
        wanted.remove(cls)
        pixels, _ = ViewDataset([row], processor, views)[0]
        mean = torch.tensor(processor.image_processor.image_mean).view(3, 1, 1)
        std = torch.tensor(processor.image_processor.image_std).view(3, 1, 1)
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        for r, view in enumerate((views.index("whole"), views.index("center_zoom70"))):
            for c in range(8):
                rgb = (pixels[view, c] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
                axes[r, c].imshow(rgb); axes[r, c].axis("off")
        fig.suptitle(f"{_sample_id(row['video_path'])}: whole(top) / center_zoom70(bottom)")
        fig.tight_layout(); fig.savefig(root / f"A{cls:03d}_input_strip.png", dpi=100); plt.close(fig)
    return {"status": "validation_complete" if args.full_validation else "diagnostic_complete",
            "raw_accuracy_gain": results["whole_zoom_mean"]["accuracy"] - results["whole"]["accuracy"], "samples": len(rows), "results": results,
            "full_validation": args.full_validation,
            "whole_zoom_accuracy_gain": results["whole_zoom_mean"]["accuracy"] - results["whole"]["accuracy"],
            "timing": {"inference_wall_seconds": inference_seconds, "samples_per_second": len(rows) / inference_seconds,
                       "forward_seconds_including_host_device_transfer": forward_seconds,
                       "scope": "offline batched pipeline includes decode/preprocessing; excludes startup and report generation; not streaming event latency"},
            "baseline_logit_max_abs_diff": parity, "baseline_prediction_disagreements": disagreements,
            "interpretation": "Validation-only fixed-view inference; no retraining or clinical alarm certification. Thresholds are exploratory (checkpoint already selected on validation), not deployment approval. Summed softmax is an event score, not danger probability.",
            "training": False, "test_evaluated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/rgb_input_audit_v1"))
    parser.add_argument("--full-validation", action="store_true")
    parser.add_argument("--whole-zoom-only", action="store_true")
    parser.add_argument("--samples", type=int, default=480, help="Subset size; ignored with --full-validation")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--split-manifest", type=Path, default=Path("artifacts/xclip/selected_split.json"))
    parser.add_argument("--action-names", type=Path, default=Path("docs/xclip-vision-lora-validation-action-accuracy.md"))
    parser.add_argument("--adapter", type=Path, default=Path("artifacts/skeletonclip_phase3/vision_batch16_workers24_uint8_seed42/vision_lora_ce_seed42/best_adapter.pt"))
    parser.add_argument("--baseline-logits", type=Path, default=Path("artifacts/skeletonclip_phase3/vision_batch16_workers24_uint8_seed42/vision_lora_ce_seed42/best_validation_logits.pt"))
    args = parser.parse_args()
    report = run(args)
    write_json(args.artifacts_dir / "run_report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
