"""Gate 4 validation-only masked-KL search orchestration."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path

from vlm.skeleton_teacher_cache import canonical_sha256, sha256_file
from vlm.training import xclip_lora


TEMPERATURES = (1.0, 2.0, 4.0)
KD_WEIGHTS = (0.1, 0.5, 1.0)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _slug(temperature: float, weight: float) -> str:
    return f"t{temperature:g}-w{weight:g}-seed42".replace(".", "p")


def _metrics(rows: list[dict[str, object]], class_ids: set[int]) -> dict[str, object]:
    selected = [row for row in rows if int(row["true_class_id"]) in class_ids]
    if not selected:
        return {"samples": 0, "accuracy": None, "macro_f1": None}
    correct = sum(row["true_class_id"] == row["predicted_class_id"] for row in selected)
    f1 = []
    for class_id in sorted(class_ids):
        tp = sum(row["true_class_id"] == class_id and row["predicted_class_id"] == class_id for row in selected)
        fp = sum(row["true_class_id"] != class_id and row["predicted_class_id"] == class_id for row in selected)
        fn = sum(row["true_class_id"] == class_id and row["predicted_class_id"] != class_id for row in selected)
        denominator = 2 * tp + fp + fn
        if denominator:
            f1.append(2 * tp / denominator)
    return {"samples": len(selected), "accuracy": correct / len(selected), "macro_f1": sum(f1) / len(f1) if f1 else 0.0}


def _subgroups(predictions: list[dict[str, object]], paired_manifest: Path) -> dict[str, object]:
    paired = json.loads(paired_manifest.read_text())
    paired_ids = {str(row["sample_id"]) for row in paired["val"]}
    first, latter, paired_rows, unpaired_rows = [], [], [], []
    for row in predictions:
        class_id = int(row["true_class_id"])
        (first if class_id <= 59 else latter).append(row)
        if class_id <= 59:
            sample_id = Path(str(row["video"])).stem.removesuffix("_rgb")
            (paired_rows if sample_id in paired_ids else unpaired_rows).append(row)
    return {
        "a001_a059": _metrics(first, set(range(1, 60))),
        "a060_a120": _metrics(latter, set(range(60, 121))),
        "a001_a059_paired": _metrics(paired_rows, set(range(1, 60))),
        "a001_a059_unpaired": _metrics(unpaired_rows, set(range(1, 60))),
    }


def _run_args(args: argparse.Namespace, *, output: Path, temperature: float, weight: float) -> argparse.Namespace:
    cache_report = json.loads((args.artifacts_dir / "cache_validation.json").read_text())
    return xclip_lora.parse_args(
        [
            "--config", str(args.winning_config), "--split-manifest", str(args.rgb_manifest),
            "--artifacts-dir", str(output), "--epochs", "60", "--seed", "42",
            "--teacher-cache", str(cache_report["cache_path"]),
            "--teacher-cache-validation", str(args.artifacts_dir / "cache_validation.json"),
            "--class-mapping", str(args.artifacts_dir / "class_mapping.json"),
            "--kd-temperature", str(temperature), "--kd-weight", str(weight),
            "--device", args.device, "--no-evaluate-test", "--no-batch-progress",
            "--no-cache-progress", "--epoch-log-frequency", "1", "--resume-training",
        ]
    )


def _candidate_record(result: dict[str, object], *, paired_manifest: Path) -> dict[str, object]:
    predictions = result["validation_predictions"]
    history = result["history"]
    numeric = [value for row in history for value in row.values() if isinstance(value, (int, float)) and value is not None]
    if not all(math.isfinite(value) for value in numeric):
        raise RuntimeError("Candidate contains a non-finite training metric")
    if any(int(row["total_samples"]) != 80_640 or int(row["paired_samples"]) != 45_236 for row in history):
        raise RuntimeError("Candidate sample accounting differs from the Gate 4 contract")
    if not any(float(row["kd_loss"]) > 0 for row in history):
        raise RuntimeError("Candidate KD loss was identically zero")
    return {
        "validation_accuracy": result["validation_accuracy"],
        "validation_macro_f1": result["validation_macro_f1"],
        "validation_nll": result["validation_nll"],
        "validation_ece": result["validation_ece"],
        "best_epoch": result["best_epoch"],
        "subgroups": _subgroups(predictions, paired_manifest),
        "history": history,
        "peak_memory_allocated_bytes": result["peak_memory_allocated_bytes"],
        "training_elapsed_seconds": result["training_elapsed_seconds"],
        "merge_parity_max_absolute_error": result["merge_parity_max_absolute_error"],
        "test_evaluated": result["test_evaluated"],
    }


def _select(summary: dict[str, object], control: dict[str, object]) -> dict[str, object]:
    completed = [record for record in summary["candidates"].values() if record["state"] == "completed"]
    if len(completed) != 9:
        return {"decision": "incomplete", "completed_candidates": len(completed)}
    best_accuracy = max(float(record["validation_accuracy"]) for record in completed)
    contenders = [record for record in completed if best_accuracy - float(record["validation_accuracy"]) < 0.001]
    winner = min(contenders, key=lambda record: (float(record["weight"]), float(record["temperature"])))
    accuracy_delta = float(winner["validation_accuracy"]) - float(control["validation_accuracy"])
    macro_delta = float(winner["validation_macro_f1"]) - float(control["validation_macro_f1"])
    advances = accuracy_delta >= 0.002 or (abs(accuracy_delta) <= 0.002 and macro_delta >= 0.003)
    return {
        "decision": "advance" if advances else "no_go",
        "winner": {key: winner[key] for key in ("candidate_id", "temperature", "weight", "validation_accuracy", "validation_macro_f1", "validation_nll", "validation_ece")},
        "accuracy_delta_percentage_points": 100 * accuracy_delta,
        "macro_f1_delta_percentage_points": 100 * macro_delta,
        "rule": "accuracy +0.20 points, or within +/-0.20 with macro-F1 +0.30 points",
    }


def run_gate4(args: argparse.Namespace) -> dict[str, object]:
    root = args.artifacts_dir
    summary_path = root / "search_summary.json"
    gate3 = json.loads((root / "gate3_summary.json").read_text())
    if gate3.get("state") != "completed" or not gate3.get("control", {}).get("parity_passed"):
        raise ValueError("Gate 4 requires a completed, parity-passed Gate 3 control")
    control_results = json.loads((root / "control_seed42" / "evaluation_results.json").read_text())
    control = {
        "validation_accuracy": control_results["validation_accuracy"],
        "validation_macro_f1": control_results["validation_macro_f1"],
        "best_epoch": control_results["best_epoch"],
        "artifacts_dir": str(root / "control_seed42"),
    }
    cache_report = json.loads((root / "cache_validation.json").read_text())
    source_hashes = {
        "winning_config": sha256_file(args.winning_config), "rgb_manifest": sha256_file(args.rgb_manifest),
        "paired_manifest": sha256_file(args.paired_manifest), "teacher_cache": sha256_file(Path(cache_report["cache_path"])),
        "cache_validation": sha256_file(root / "cache_validation.json"), "class_mapping": sha256_file(root / "class_mapping.json"),
        "gate3_summary": sha256_file(root / "gate3_summary.json"), "xclip_lora_source": sha256_file(Path(xclip_lora.__file__)),
        "gate4_source": sha256_file(Path(__file__)),
    }
    identity = {"schema_version": 1, "source_hashes": source_hashes, "temperatures": list(TEMPERATURES), "weights": list(KD_WEIGHTS), "seed": 42, "epochs": 60, "test_evaluated": False}
    identity["identity_sha256"] = canonical_sha256(identity)
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary.get("identity") != identity:
            raise ValueError("Gate 4 resume identity differs from the existing search")
        if summary.get("state") == "completed": return summary
    else:
        summary = {"schema_version": 1, "state": "running", "identity": identity, "control": control, "candidates": {}, "started_at_unix": time.time(), "test_evaluated": False}
        _write_json(summary_path, summary)

    preparation = _run_args(args, output=root / "search" / "cache_preparation", temperature=1, weight=0.1)
    preparation.extract_only = True
    prepared = xclip_lora.run_pipeline(preparation, evaluate_test=False, cache_read_only=True, print_results=False, print_run_details=False)["_cached_features"]
    datasets = xclip_lora.build_cached_datasets(prepared, preload=True)
    for temperature in TEMPERATURES:
        for weight in KD_WEIGHTS:
            candidate_id = _slug(temperature, weight)
            existing = summary["candidates"].get(candidate_id)
            if existing and existing.get("state") == "completed": continue
            record = {"candidate_id": candidate_id, "temperature": temperature, "weight": weight, "state": "running", "artifacts_dir": str(root / "search" / candidate_id), "started_at_unix": time.time()}
            summary["candidates"][candidate_id] = record; _write_json(summary_path, summary)
            try:
                run_args = _run_args(args, output=Path(record["artifacts_dir"]), temperature=temperature, weight=weight)
                result = xclip_lora.run_pipeline(run_args, evaluate_test=False, cache_read_only=True, prepared_cached=prepared, prepared_datasets=datasets, print_results=False, print_run_details=False)
                record.update(state="completed", runtime_seconds=time.time() - record["started_at_unix"], **_candidate_record(result, paired_manifest=args.paired_manifest))
            except BaseException as error:
                record.update(state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=f"{type(error).__name__}: {error}", runtime_seconds=time.time() - record["started_at_unix"])
                _write_json(summary_path, summary); raise
            _write_json(summary_path, summary)
    summary["selection"] = _select(summary, control)
    summary["state"] = "completed"
    summary["completed_at_unix"] = time.time()
    _write_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SkeletonCLIP Phase 2 Gate 4")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/skeletonclip_phase2"))
    parser.add_argument("--winning-config", type=Path, default=Path("artifacts/xclip_lora_tuning/winning_config.json"))
    parser.add_argument("--rgb-manifest", type=Path, default=Path("artifacts/xclip/selected_split.json"))
    parser.add_argument("--paired-manifest", type=Path, default=Path("artifacts/skeleton_data/paired_manifest.json"))
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    result = run_gate4(build_parser().parse_args(argv))
    print(json.dumps({"state": result["state"], "selection": result.get("selection")}, indent=2))


if __name__ == "__main__": main()
