"""Person-aware crop Stage 0: crop-aware epoch 20 inference on RGB person crops.

Runs the merged crop-aware checkpoint on the person crop view of every validation sample in a
person_crop_localization manifest, reuses the saved whole and center_zoom70 logits, and applies
the pre-registered decision rules. Contract: docs/person-aware-crop-stage0-plan.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from transformers import XCLIPModel, XCLIPProcessor

from vlm.experiments.video_sampling import read_video_pyav, uniform_frame_indices
from vlm.models.xclip_vision_lora import XCLIPVisionClassifier
from vlm.skeleton_teacher_cache import sha256_file
from vlm.training.person_crop_localization import center_zoom70_box
from vlm.training.skeletonclip_phase3_gate1 import _atomic_checkpoint, _atomic_json

ATTENTION_IDS = tuple(range(42, 49))
TWO_PERSON_IDS = (*range(50, 61), *range(106, 121))
REVIEW_GROUPS = {"A043": (43,), "A048": (48,), "A045": (45,), "two_person": TWO_PERSON_IDS}
# Pre-registered thresholds (plan section 6): whole + zoom correct and A042–A048 correct at epoch 20.
WHOLE_ZOOM_CORRECT, ATTENTION_CORRECT, NO_GO_BELOW = 10742, 404, 10716


def crop(frames: np.ndarray, box: list[int]) -> np.ndarray:
    left, top, right, bottom = box
    return frames[:, top:bottom, left:right]


class CropDataset(Dataset):
    def __init__(self, records: list[dict], processor: XCLIPProcessor, views: tuple[str, ...]) -> None:
        self.records, self.processor, self.views = records, processor, views

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> torch.Tensor:
        record = self.records[index]
        path = record["video_path"]
        if sha256_file(Path(path)) != record["rgb_sha256"]:
            raise ValueError(f"Source video changed since localization: {path}")
        with av.open(path) as container:
            indices = uniform_frame_indices(8, container.streams.video[0].frames)
            if indices.tolist() != record["frame_indices"]:
                raise ValueError(f"Frame indices differ from localization: {path}")
            frames = read_video_pyav(container, indices)
        boxes = {"whole": None, "center_zoom70": center_zoom70_box(record["width"], record["height"]), "person_crop": record["crop"]}
        views = [frames if boxes[view] is None else crop(frames, boxes[view]) for view in self.views]
        return torch.stack([self.processor(images=list(view), return_tensors="pt")["pixel_values"].squeeze(0) for view in views])


def infer(model: torch.nn.Module, dataset: CropDataset, args: argparse.Namespace) -> torch.Tensor:
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    outputs = []
    with torch.inference_mode():
        for pixels in loader:
            pixels = pixels.cuda(non_blocking=True)
            outputs.append(torch.stack([model(pixels[:, i]).float().cpu() for i in range(pixels.shape[1])], 1))
    return torch.cat(outputs)


def metrics(probability: torch.Tensor, labels: torch.Tensor) -> dict[str, object]:
    predicted = probability.argmax(-1)
    ok = predicted.eq(labels)
    confusion = torch.bincount(labels * 120 + predicted, minlength=120 * 120).reshape(120, 120)
    tp = confusion.diag().float()
    denominator = confusion.sum(0).float() + confusion.sum(1).float()
    attention = torch.isin(labels, torch.tensor(ATTENTION_IDS) - 1)
    return {"accuracy": float(ok.float().mean()), "correct": int(ok.sum()), "samples": int(labels.numel()),
            "macro_f1": float(torch.where(denominator > 0, 2 * tp / denominator.clamp_min(1), 0).mean()),
            "nll": float(-probability[torch.arange(len(labels)), labels].clamp_min(1e-38).log().mean()),
            "attention_correct": int(ok[attention].sum()), "attention_samples": int(attention.sum()),
            "events": {f"A{c:03d}": int(ok[labels.eq(c - 1)].sum()) for c in ATTENTION_IDS}}


def iou(a: list[int], b: list[int]) -> float:
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = width * height
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def geometry(records: list[dict], labels: torch.Tensor) -> dict[str, object]:
    classes = (labels + 1).tolist()
    rows = []
    for record, cls in zip(records, classes):
        zoom = center_zoom70_box(record["width"], record["height"])
        box = record["landmark_box"]
        beyond = box is not None and (box[0] < zoom[0] or box[1] < zoom[1] or box[2] > zoom[2] or box[3] > zoom[3])
        rows.append({"cls": cls, "fallback": record["fallback"] is not None, "beyond": beyond,
                     "iou": iou(record["crop"], zoom), "side": (record["crop"][2] - record["crop"][0]) / min(record["width"], record["height"])})

    def summary(selected: list[dict]) -> dict[str, object]:
        located = [r for r in selected if not r["fallback"]]
        iou_values = np.array([r["iou"] for r in located]) if located else np.zeros(1)
        sides = np.array([r["side"] for r in located]) if located else np.zeros(1)
        return {"samples": len(selected), "fallbacks": len(selected) - len(located),
                "person_beyond_center_zoom70": sum(r["beyond"] for r in located),
                "person_beyond_fraction": sum(r["beyond"] for r in located) / max(1, len(located)),
                "iou_quantiles": dict(zip(("p10", "p50", "p90"), np.quantile(iou_values, [.1, .5, .9]).round(4).tolist())),
                "side_over_short_edge_quantiles": dict(zip(("p10", "p50", "p90"), np.quantile(sides, [.1, .5, .9]).round(4).tolist()))}

    groups = {"all": rows, **{name: [r for r in rows if r["cls"] in ids] for name, ids in REVIEW_GROUPS.items() if name != "A045"}}
    fallback_classes: dict[str, int] = {}
    for r in rows:
        if r["fallback"]:
            fallback_classes[f"A{r['cls']:03d}"] = fallback_classes.get(f"A{r['cls']:03d}", 0) + 1
    return {**{name: summary(selected) for name, selected in groups.items()}, "fallbacks_by_class": dict(sorted(fallback_classes.items()))}


def review(records: list[dict], labels: torch.Tensor, root: Path) -> list[str]:
    classes = (labels + 1).tolist()
    key = lambda i: hashlib.sha256(f"person-crop-review:{records[i]['sample_id']}".encode()).hexdigest()
    chosen = {name: sorted((i for i, c in enumerate(classes) if c in ids), key=key)[:3] for name, ids in REVIEW_GROUPS.items()}
    chosen["fallback"] = [i for i, r in enumerate(records) if r["fallback"] is not None][:6]
    root.mkdir(parents=True, exist_ok=True)
    written = []
    for group, indices in chosen.items():
        for i in indices:
            record = records[i]
            with av.open(record["video_path"]) as container:
                frames = read_video_pyav(container, np.array(record["frame_indices"]))
            source = Image.fromarray(frames[len(frames) // 2])
            draw = ImageDraw.Draw(source)
            draw.rectangle(center_zoom70_box(record["width"], record["height"]), outline=(255, 200, 0), width=6)
            draw.rectangle(record["crop"], outline=(0, 220, 255), width=6)
            for persons in record["poses"][len(frames) // 2:len(frames) // 2 + 1]:
                for person in persons:
                    for x, y in person:
                        draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(0, 220, 255))
            source.thumbnail((960, 960))
            strip = Image.new("RGB", (224 * 8, 224))
            for position, frame in enumerate(crop(frames, record["crop"])):
                strip.paste(Image.fromarray(frame).resize((224, 224)), (224 * position, 0))
            sheet = Image.new("RGB", (max(source.width, strip.width), source.height + 224), "white")
            sheet.paste(source, (0, 0))
            sheet.paste(strip, (0, source.height))
            name = f"{group}_{record['sample_id']}.jpg"
            sheet.save(root / name, quality=85)
            written.append(name)
    return written


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Person crop inference requires a visible CUDA GPU")
    manifest = json.loads(args.crops.read_text(encoding="utf-8"))
    lock_path = args.crops.with_name("experiment_lock.json")
    if manifest["lock_sha256"] != sha256_file(lock_path):
        raise ValueError("Crop manifest does not match its localization lock")
    records = manifest["samples"][:args.max_samples]
    saved = torch.load(args.predictions, map_location="cpu", weights_only=False)
    if [r["sample_id"] for r in records] != saved["sample_ids"][:len(records)]:
        raise ValueError("Manifest sample order differs from the saved crop-aware predictions")
    labels = saved["labels"][:len(records)]
    run_lock = json.loads(args.run_lock.read_text(encoding="utf-8"))
    recipe = run_lock["recipe"]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["identity"] != {k: v for k, v in run_lock.items() if k not in ("vlm_package_path", "initial_logits_sha256", "initial_logits_match_pilot")}:
        raise ValueError("Checkpoint identity differs from the crop-aware run lock")
    root = args.artifacts_dir
    root.mkdir(parents=True, exist_ok=True)
    _atomic_json(root / "experiment_lock.json", {
        "experiment": "person_crop_stage0_evaluation", "samples": len(records), "checkpoint_epoch": checkpoint["epoch"],
        "source_sha256": {"runner": sha256_file(Path(__file__)), "crops": sha256_file(args.crops), "localization_lock": sha256_file(lock_path),
                          "checkpoint": sha256_file(args.checkpoint), "predictions": sha256_file(args.predictions), "run_lock": sha256_file(args.run_lock)},
        "precision": "float32", "reused_views": ["whole", "center_zoom70"], "inferred_views": ["person_crop"],
        "parity_samples": args.parity_samples, "training": False, "test_evaluated": False})
    processor = XCLIPProcessor.from_pretrained(recipe["model_name"], revision=recipe["model_revision"], local_files_only=True)
    model = XCLIPVisionClassifier(
        XCLIPModel.from_pretrained(recipe["model_name"], revision=recipe["model_revision"], local_files_only=True),
        num_classes=recipe["num_classes"], head_dropout=recipe["head_dropout"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.cuda().eval()

    parity_logits = infer(model, CropDataset(records[:args.parity_samples], processor, ("whole", "center_zoom70")), args)
    parity = {}
    for position, view in enumerate(("whole", "center_zoom70")):
        reference = saved["logits"][view][:args.parity_samples].float()
        parity[view] = {"max_abs_diff": float((parity_logits[:, position] - reference).abs().max()),
                        "argmax_equal": bool(parity_logits[:, position].argmax(-1).eq(reference.argmax(-1)).all())}
        if not parity[view]["argmax_equal"] or parity[view]["max_abs_diff"] > 1e-3:
            raise RuntimeError(f"Saved-logit parity failed: {parity}")
    print(json.dumps({"state": "parity_passed", **parity}), flush=True)

    logits = {view: saved["logits"][view][:len(records)].float() for view in ("whole", "center_zoom70")}
    logits["person_crop"] = infer(model, CropDataset(records, processor, ("person_crop",)), args)[:, 0]
    if not torch.isfinite(logits["person_crop"]).all():
        raise RuntimeError("Non-finite person crop logits")
    _atomic_checkpoint(root / "view_predictions.pt", {"logits": logits, "labels": labels, "sample_ids": [r["sample_id"] for r in records],
                                                      "training": False, "test_evaluated": False})
    probability = {view: value.softmax(-1) for view, value in logits.items()}
    probability["whole_zoom_mean"] = (probability["whole"] + probability["center_zoom70"]) / 2
    probability["whole_person_mean"] = (probability["whole"] + probability["person_crop"]) / 2
    probability["whole_zoom_person_mean"] = (probability["whole"] + probability["center_zoom70"] + probability["person_crop"]) / 3
    results = {name: metrics(value, labels) for name, value in probability.items()}
    fallback = torch.tensor([r["fallback"] is not None for r in records])
    results["person_crop_on_fallbacks"] = metrics(probability["person_crop"][fallback], labels[fallback]) if fallback.any() else None

    ok = {name: probability[name].argmax(-1).eq(labels) for name in ("center_zoom70", "whole_zoom_mean", "person_crop", "whole_person_mean")}
    changes = {}
    for new, old in (("person_crop", "center_zoom70"), ("whole_person_mean", "whole_zoom_mean")):
        changes[f"{new}_vs_{old}"] = {
            "fixes": [{"sample_id": records[i]["sample_id"], "class_id": int(labels[i]) + 1} for i in (ok[new] & ~ok[old]).nonzero().flatten().tolist()],
            "regressions": [{"sample_id": records[i]["sample_id"], "class_id": int(labels[i]) + 1} for i in (~ok[new] & ok[old]).nonzero().flatten().tolist()]}
    _atomic_json(root / "case_changes.json", changes)

    full = len(records) == 12960
    candidate = results["whole_person_mean"]
    decision = None
    if full:
        if candidate["correct"] >= WHOLE_ZOOM_CORRECT and candidate["attention_correct"] >= ATTENTION_CORRECT:
            decision = "go"
        elif candidate["correct"] < NO_GO_BELOW:
            decision = "no_go"
        else:
            decision = "inconclusive"
    report = {"decision": decision, "full_validation": full, "metrics": results, "parity": parity,
              "fixes_regressions": {name: {k: len(v) for k, v in value.items()} for name, value in changes.items()},
              "geometry": geometry(records, labels), "review_images": review(records, labels, root / "review"),
              "test_evaluated": False}
    _atomic_json(root / "report.json", report)
    print(json.dumps({"state": "complete", "decision": decision,
                      **{name: results[name]["correct"] for name in ("center_zoom70", "whole_zoom_mean", "person_crop", "whole_person_mean")}}), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    crop_aware = Path("artifacts/patch16_crop_aware/seed42")
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=crop_aware / "final_merged.pt")
    parser.add_argument("--predictions", type=Path, default=crop_aware / "final_view_predictions.pt")
    parser.add_argument("--run-lock", type=Path, default=crop_aware / "run_lock.json")
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--parity-samples", type=int, default=64)
    parser.add_argument("--max-samples", type=int)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
