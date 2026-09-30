"""Post-hoc pooling ablation of the completed checkpoint; no training or test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import XCLIPModel, XCLIPProcessor

from vlm.models.xclip_vision_lora import XCLIPAttentionClassifier
from vlm.processed_uint8_cache import ProcessedUInt8Cache
from vlm.reporting import write_json
from vlm.skeleton_teacher_cache import sha256_file
from vlm.training.spatial_attention import classification_report
from vlm.training.spatial_attention_smoke import MODEL, REVISION


class RGBRows(Dataset):
    def __init__(self, rows: list[dict], cache: ProcessedUInt8Cache) -> None:
        self.rows, self.cache = rows, cache

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        return self.cache.load(Path(row["video_path"])), int(row["class_id"]) - 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=Path("artifacts/spatial_attention_v2/pilot_seed42"))
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--samples-per-class", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.samples_per_class <= 8:
        raise ValueError("Diagnostic budget must be 1–8 samples per class")
    torch.set_num_threads(8)
    pilot_lock = json.loads((args.pilot_dir / "experiment_lock.json").read_text())
    split_path = Path("artifacts/xclip/selected_split.json")
    paired_path = Path("artifacts/skeleton_data/paired_manifest.json")
    baseline_path = Path("artifacts/skeletonclip_phase3/vision_batch16_workers24_uint8_seed42/vision_lora_ce_seed42/best_validation_logits.pt")
    for key, path in (("split", split_path), ("paired", paired_path), ("baseline_logits", baseline_path)):
        if pilot_lock["source_sha256"][key] != sha256_file(path):
            raise ValueError(f"Pilot {key} provenance changed")
    split = json.loads(split_path.read_text())
    buckets = {label: [] for label in range(1, 121)}
    for row in split["val"]:
        buckets[int(row["class_id"])].append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: hashlib.sha256(Path(row["video_path"]).stem.encode()).hexdigest())
    rows = [row for bucket in buckets.values() for row in bucket[:args.samples_per_class]]
    ids = [Path(row["video_path"]).stem.removesuffix("_rgb") for row in rows]
    checkpoint_path = args.pilot_dir / "correct/merged_rgb_only.pt"
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.artifacts_dir / "experiment_lock.json"
    if lock_path.exists():
        raise ValueError("Diagnostic directory already locked")
    write_json(lock_path, {"mode": "post_hoc_inference_diagnostic", "samples": len(rows), "sample_ids": ids,
                          "checkpoint_sha256": sha256_file(checkpoint_path), "test_evaluated": False,
                          "arms": ["existing_baseline_logits", "existing_checkpoint_pooling_on", "existing_checkpoint_alpha_zero"],
                          "interpretation": "Same trained vision/MIT/head weights with alpha ablation; no parameter updates. Subset results cannot replace canonical pilot gate."})
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["provenance"] != pilot_lock:
        raise ValueError("Checkpoint is not from the locked pilot")
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=True)
    indices = {sample: index for index, sample in enumerate(baseline["sample_ids"])}
    selected = torch.tensor([indices[sample] for sample in ids])
    labels = torch.tensor([int(row["class_id"]) - 1 for row in rows])
    if not torch.equal(baseline["labels"][selected], labels):
        raise ValueError("Baseline labels differ")
    paired = json.loads(paired_path.read_text())
    paired_ids = {row["sample_id"] for row in paired["val"]}
    processor = XCLIPProcessor.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    cache = ProcessedUInt8Cache(Path("cache/skeletonclip_phase3_rgb_uint8"), processor, model_name=MODEL, model_revision=REVISION)
    loader = DataLoader(RGBRows(rows, cache), batch_size=4, shuffle=False, num_workers=4)
    model = XCLIPAttentionClassifier(XCLIPModel.from_pretrained(MODEL, revision=REVISION, local_files_only=True),
                                     num_classes=120, head_dropout=0.25).cuda().eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    alpha = float(model.pooling_alpha.detach())
    outputs_on, outputs_off = [], []
    progress_path = args.artifacts_dir / "evaluation/progress.json"
    with torch.inference_mode():
        for index, (pixels, _) in enumerate(loader):
            pixels = pixels.cuda()
            model.pooling_alpha.fill_(alpha)
            outputs_on.append(model(pixels).cpu())
            model.pooling_alpha.zero_()
            outputs_off.append(model(pixels).cpu())
            if index % 20 == 0:
                write_json(progress_path, {"state": "evaluating_pooling_ablation", "samples": min((index + 1) * 4, len(rows)), "total_samples": len(rows)})
    on, off = torch.cat(outputs_on), torch.cat(outputs_off)
    saved = torch.load(args.pilot_dir / "correct/validation_predictions.pt", map_location="cpu", weights_only=True)
    saved_indices = {sample: index for index, sample in enumerate(saved["sample_ids"])}
    reload_error = float((on - saved["logits"][[saved_indices[sample] for sample in ids]]).abs().max())
    if reload_error > 1e-4:
        raise RuntimeError(f"On-disk checkpoint prediction parity failed: {reload_error}")
    reports = {name: classification_report(logits, labels, ids, paired_ids)
               for name, logits in (("baseline", baseline["logits"][selected]), ("pooling_on", on), ("alpha_zero", off))}
    report = {"status": "diagnostic_complete", "test_evaluated": False, "samples": len(rows), "alpha": alpha,
              "raw_accuracy_gain": reports["pooling_on"]["raw_accuracy"] - reports["baseline"]["raw_accuracy"],
              "pooling_effect_accuracy": reports["pooling_on"]["raw_accuracy"] - reports["alpha_zero"]["raw_accuracy"],
              "adapted_backbone_head_effect_accuracy": reports["alpha_zero"]["raw_accuracy"] - reports["baseline"]["raw_accuracy"],
              "prediction_reload_max_abs_diff": reload_error, "classification": reports,
              "limitation": "Post-hoc 480-sample inference diagnostic, not a new pilot or trained control. Alpha-zero isolates pooling at fixed trained weights, not the causal contribution of spatial gradients during training."}
    write_json(args.artifacts_dir / "run_report.json", report)
    torch.save({"baseline_logits": baseline["logits"][selected], "pooling_on_logits": on, "alpha_zero_logits": off,
                "labels": labels, "sample_ids": ids}, args.artifacts_dir / "predictions.pt")
    write_json(progress_path, {"state": "complete", "samples": len(rows)})
    print(json.dumps({key: value for key, value in report.items() if key != "classification"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
