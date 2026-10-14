"""One-shot test evaluation of the locked person crop configuration.

Final prediction: the mean of the crop-aware checkpoint's softmax on center_zoom70 (A) and the
person-crop-trained checkpoint's softmax on the RGB person crop (B). Both models first reproduce
their saved validation logits; the test split is then read once and the runner refuses to rerun.
Contract: docs/person-crop-test-evaluation-plan.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import XCLIPModel, XCLIPProcessor

from vlm.models.xclip_vision_lora import XCLIPVisionClassifier
from vlm.skeleton_teacher_cache import sha256_file
from vlm.training.person_crop_audit import CropDataset, metrics
from vlm.training.skeletonclip_phase3_gate1 import _atomic_checkpoint, _atomic_json

VIEWS = ("whole", "center_zoom70", "person_crop")
MODEL_VIEWS = {"A": VIEWS, "B": ("whole", "person_crop")}
FINAL = ("A_center_zoom70", "B_person_crop")
REFERENCE = "A_person_crop"
RUN_LOCK_ONLY = ("vlm_package_path", "initial_logits_sha256", "initial_logits_match_pilot")


def manifest_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["lock_sha256"] != sha256_file(path.with_name("experiment_lock.json")):
        raise ValueError(f"Crop manifest does not match its localization lock: {path}")
    return payload["samples"]


def load_model(root: Path) -> tuple[XCLIPVisionClassifier, dict, XCLIPProcessor]:
    run_lock = json.loads((root / "run_lock.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(root / "final_merged.pt", map_location="cpu", weights_only=False)
    if checkpoint["identity"] != {k: v for k, v in run_lock.items() if k not in RUN_LOCK_ONLY}:
        raise ValueError(f"Checkpoint identity differs from its run lock: {root}")
    recipe = run_lock["recipe"]
    processor = XCLIPProcessor.from_pretrained(recipe["model_name"], revision=recipe["model_revision"], local_files_only=True)
    model = XCLIPVisionClassifier(
        XCLIPModel.from_pretrained(recipe["model_name"], revision=recipe["model_revision"], local_files_only=True),
        num_classes=recipe["num_classes"], head_dropout=recipe["head_dropout"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.cuda().eval(), recipe, processor


def infer(models: dict[str, torch.nn.Module], dataset: CropDataset, args: argparse.Namespace) -> dict[str, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    outputs: dict[str, list[torch.Tensor]] = {f"{name}_{view}": [] for name, views in MODEL_VIEWS.items() for view in views}
    with torch.inference_mode():
        for count, pixels in enumerate(loader, 1):
            pixels = pixels.cuda(non_blocking=True)
            for name, views in MODEL_VIEWS.items():
                for view in views:
                    outputs[f"{name}_{view}"].append(models[name](pixels[:, VIEWS.index(view)]).float().cpu())
            if count % 100 == 0:
                print(json.dumps({"state": "inferring", "samples": min(count * args.batch_size, len(dataset)), "total": len(dataset)}), flush=True)
    return {key: torch.cat(parts) for key, parts in outputs.items()}


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Test evaluation requires a visible CUDA GPU")
    root = args.artifacts_dir
    if (root / "report.json").exists():
        raise RuntimeError(f"Test evaluation already completed: {root / 'report.json'}")
    test_records = manifest_records(args.test_crops)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    rows = split["test"]
    if [Path(r["video_path"]).stem.removesuffix("_rgb") for r in rows] != [r["sample_id"] for r in test_records]:
        raise ValueError("Test crop manifest order differs from the split")
    if len(rows) != 20880:
        raise ValueError(f"Expected 20,880 test samples, got {len(rows)}")

    models, recipes, processor = {}, {}, None
    for name, path in (("A", args.crop_aware), ("B", args.person_trained)):
        models[name], recipes[name], processor = load_model(path)
    if recipes["A"]["model_name"] != recipes["B"]["model_name"] or recipes["A"]["model_revision"] != recipes["B"]["model_revision"]:
        raise ValueError("A and B use different backbones")
    root.mkdir(parents=True, exist_ok=True)
    _atomic_json(root / "experiment_lock.json", {
        "experiment": "person_crop_test_evaluation", "samples": len(test_records), "final": list(FINAL), "reference": REFERENCE,
        "source_sha256": {"runner": sha256_file(Path(__file__)), "split": sha256_file(args.split),
                          "test_crops": sha256_file(args.test_crops), "val_crops": sha256_file(args.val_crops),
                          **{f"{name}_{file}": sha256_file(path / file) for name, path in (("A", args.crop_aware), ("B", args.person_trained))
                             for file in ("final_merged.pt", "run_lock.json")}},
        "precision": "float32", "parity_samples": args.parity_samples, "training": False, "test_evaluated": True})

    # Validation parity against the saved logits, before any test video is read.
    val_records = manifest_records(args.val_crops)[:args.parity_samples]
    saved = {"A": torch.load(args.stage0_predictions, map_location="cpu", weights_only=False),
             "B": torch.load(args.person_trained / "final_view_predictions.pt", map_location="cpu", weights_only=False)}
    ids = [r["sample_id"] for r in val_records]
    if any(value["sample_ids"][:len(ids)] != ids for value in saved.values()):
        raise ValueError("Validation crop manifest order differs from the saved predictions")
    parity_logits = infer(models, CropDataset(val_records, processor, VIEWS), args)
    parity = {}
    for key, value in parity_logits.items():
        name, view = key.split("_", 1)
        reference = saved[name]["logits"][view][:len(ids)].float()
        parity[key] = {"max_abs_diff": float((value - reference).abs().max()), "argmax_equal": bool(value.argmax(-1).eq(reference.argmax(-1)).all())}
        if not parity[key]["argmax_equal"] or parity[key]["max_abs_diff"] > 1e-3:
            raise RuntimeError(f"Validation parity failed: {parity}")
    print(json.dumps({"state": "parity_passed", **parity}), flush=True)

    logits = infer(models, CropDataset(test_records, processor, VIEWS), args)
    if not all(torch.isfinite(value).all() for value in logits.values()):
        raise RuntimeError("Non-finite test logits")
    labels = torch.tensor([int(r["class_id"]) - 1 for r in rows])
    _atomic_checkpoint(root / "test_predictions.pt", {"logits": logits, "labels": labels, "sample_ids": [r["sample_id"] for r in test_records],
                                                      "training": False, "test_evaluated": True})
    probability = {key: value.softmax(-1) for key, value in logits.items()}
    probability["A_whole_center_zoom70"] = (probability["A_whole"] + probability["A_center_zoom70"]) / 2
    probability["final"] = sum(probability[key] for key in FINAL) / len(FINAL)
    results = {name: metrics(value, labels) for name, value in probability.items()}
    ok_final, ok_reference = probability["final"].argmax(-1).eq(labels), probability[REFERENCE].argmax(-1).eq(labels)
    final, reference = results["final"], results[REFERENCE]

    report = {"metrics": results, "validation_parity": parity,
              "final_vs_reference": {"fixes": int((ok_final & ~ok_reference).sum()), "regressions": int((~ok_final & ok_reference).sum()),
                                     "delta_percentage_points": 100 * (final["accuracy"] - reference["accuracy"])},
              "test_evaluated": True}
    _atomic_json(root / "report.json", report)
    print(json.dumps({"state": "complete", **{name: value["correct"] for name, value in results.items()}}), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=Path("artifacts/xclip/selected_split.json"))
    parser.add_argument("--test-crops", type=Path, default=Path("artifacts/person_crop_test/localization/crop_manifest.json"))
    parser.add_argument("--val-crops", type=Path, default=Path("artifacts/person_crop_stage0/localization/crop_manifest.json"))
    parser.add_argument("--stage0-predictions", type=Path, default=Path("artifacts/person_crop_stage0/evaluation/view_predictions.pt"))
    parser.add_argument("--crop-aware", type=Path, default=Path("artifacts/patch16_crop_aware/seed42"))
    parser.add_argument("--person-trained", type=Path, default=Path("artifacts/patch16_person_crop/seed42"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/person_crop_test/evaluation"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--parity-samples", type=int, default=64)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
