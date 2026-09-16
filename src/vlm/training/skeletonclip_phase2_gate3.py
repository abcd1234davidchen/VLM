"""Gate 3 controlled smoke and full CE-control orchestration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from vlm.skeleton_teacher_cache import canonical_sha256, sha256_file
from vlm.training import xclip_lora


SMOKE_RECIPES = (
    {"name": "ce_control", "temperature": 2.0, "weight": 0.0},
    {"name": "kd_low", "temperature": 2.0, "weight": 0.1},
    {"name": "kd_high", "temperature": 4.0, "weight": 1.0},
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _stable_subset(rows: list[dict[str, object]], fraction: float) -> list[dict[str, object]]:
    by_class: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_class[int(row["class_id"])].append(row)
    selected = []
    for class_id in sorted(by_class):
        ranked = sorted(
            by_class[class_id],
            key=lambda row: hashlib.sha256(str(row["video_path"]).encode()).hexdigest(),
        )
        selected.extend(ranked[: max(1, round(len(ranked) * fraction))])
    return sorted(selected, key=lambda row: str(row["video_path"]))


def _build_smoke_manifest(source: Path, target: Path) -> dict[str, int]:
    payload = json.loads(source.read_text())
    smoke = {
        key: value
        for key, value in payload.items()
        if key not in {"train", "val", "test", "participants", "source_manifest"}
    }
    smoke["source_manifest"] = str(source)
    smoke["train"] = _stable_subset(payload["train"], 0.025)
    smoke["val"] = _stable_subset(payload["val"], 0.04)
    smoke["test"] = []
    smoke["participants"] = {
        name: sorted(
            {
                int(Path(row["video_path"]).stem.split("P", 1)[1][:3])
                for row in rows
            }
        )
        for name, rows in (("train", smoke["train"]), ("val", smoke["val"]), ("test", []))
    }
    _write_json(target, smoke)
    return {"train": len(smoke["train"]), "val": len(smoke["val"]), "test": 0}


def _run_args(
    *, config: Path, manifest: Path, output: Path, teacher_cache: Path,
    cache_validation: Path, class_mapping: Path, epochs: int,
    temperature: float, weight: float, device: str,
) -> argparse.Namespace:
    return xclip_lora.parse_args(
        [
            "--config", str(config), "--split-manifest", str(manifest),
            "--artifacts-dir", str(output), "--epochs", str(epochs),
            "--teacher-cache", str(teacher_cache),
            "--teacher-cache-validation", str(cache_validation),
            "--class-mapping", str(class_mapping),
            "--kd-temperature", str(temperature), "--kd-weight", str(weight),
            "--device", device, "--no-evaluate-test", "--no-batch-progress",
            "--no-cache-progress", "--epoch-log-frequency", "1", "--resume-training",
        ]
    )


def _assert_run(result: dict[str, object], *, expected_train: int, require_kd: bool) -> None:
    history = result["history"]
    if not history:
        raise RuntimeError("Gate 3 run produced no epochs")
    numeric = (
        value
        for row in history
        for value in row.values()
        if isinstance(value, (int, float)) and value is not None
    )
    if not all(math.isfinite(value) for value in numeric):
        raise RuntimeError("Gate 3 run contains a non-finite metric")
    if any(int(row["total_samples"]) != expected_train for row in history):
        raise RuntimeError("Gate 3 run consumed an unexpected CE sample count")
    if require_kd and not any(float(row["kd_loss"]) > 0 for row in history):
        raise RuntimeError("Gate 3 KD loss was identically zero")
    if result.get("test_evaluated") is not False or int(result.get("test_samples", -1)) != 0:
        raise RuntimeError("Gate 3 test embargo was violated")


def run_gate3(args: argparse.Namespace) -> dict[str, object]:
    root = args.artifacts_dir
    summary_path = root / "gate3_summary.json"
    cache_validation = root / "cache_validation.json"
    class_mapping = root / "class_mapping.json"
    cache_report = json.loads(cache_validation.read_text())
    teacher_cache = Path(cache_report["cache_path"])
    smoke_manifest = root / "smoke_split.json"
    smoke_counts = _build_smoke_manifest(args.rgb_manifest, smoke_manifest)
    source_hashes = {
        "winning_config": sha256_file(args.winning_config),
        "rgb_manifest": sha256_file(args.rgb_manifest),
        "teacher_cache": sha256_file(teacher_cache),
        "cache_validation": sha256_file(cache_validation),
        "class_mapping": sha256_file(class_mapping),
        "xclip_lora_source": sha256_file(Path(xclip_lora.__file__)),
        "gate3_source": sha256_file(Path(__file__)),
    }
    identity = {
        "schema_version": 1,
        "source_hashes": source_hashes,
        "smoke_counts": smoke_counts,
        "smoke_epochs": args.smoke_epochs,
        "smoke_recipes": list(SMOKE_RECIPES),
        "control_epochs": 60,
        "control_seed": 42,
        "test_evaluated": False,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary.get("identity") != identity:
            if summary.get("state") not in {"failed", "running_smoke", "running_control"}:
                raise ValueError("Gate 3 resume identity differs from the existing summary")
            archived = summary_path.with_name(
                f"gate3_summary.preflight_failed_{int(time.time())}.json"
            )
            summary_path.replace(archived)
            summary = {
                "schema_version": 1,
                "state": "running_smoke",
                "identity": identity,
                "smoke_runs": {},
                "control": None,
                "started_at_unix": time.time(),
                "previous_preflight_failure": str(archived),
            }
            _write_json(summary_path, summary)
        if summary.get("state") == "completed":
            return summary
    else:
        summary = {
            "schema_version": 1,
            "state": "running_smoke",
            "identity": identity,
            "smoke_runs": {},
            "control": None,
            "started_at_unix": time.time(),
        }
        _write_json(summary_path, summary)

    for recipe in SMOKE_RECIPES:
        name = str(recipe["name"])
        if summary["smoke_runs"].get(name, {}).get("state") == "completed":
            continue
        record = {
            "state": "running", "recipe": recipe,
            "artifacts_dir": str(root / "smoke" / name),
            "started_at_unix": time.time(),
        }
        summary["smoke_runs"][name] = record
        _write_json(summary_path, summary)
        try:
            run_args = _run_args(
                config=args.winning_config, manifest=smoke_manifest,
                output=Path(record["artifacts_dir"]), teacher_cache=teacher_cache,
                cache_validation=cache_validation, class_mapping=class_mapping,
                epochs=args.smoke_epochs, temperature=float(recipe["temperature"]),
                weight=float(recipe["weight"]), device=args.device,
            )
            result = xclip_lora.run_pipeline(run_args, evaluate_test=False, cache_read_only=True, print_results=False)
            _assert_run(result, expected_train=smoke_counts["train"], require_kd=float(recipe["weight"]) > 0)
            record.update(
                state="completed", runtime_seconds=time.time() - record["started_at_unix"],
                validation_accuracy=result["validation_accuracy"],
                validation_macro_f1=result["validation_macro_f1"],
                history=result["history"], peak_memory_allocated_bytes=result["peak_memory_allocated_bytes"],
                merge_parity_max_absolute_error=result["merge_parity_max_absolute_error"],
            )
        except BaseException as error:
            record.update(state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=f"{type(error).__name__}: {error}")
            _write_json(summary_path, summary)
            raise
        _write_json(summary_path, summary)

    summary["state"] = "running_control"
    _write_json(summary_path, summary)
    if not summary.get("control") or summary["control"].get("state") != "completed":
        record = {
            "state": "running", "seed": 42, "temperature": 2.0, "weight": 0.0,
            "artifacts_dir": str(root / "control_seed42"), "started_at_unix": time.time(),
        }
        summary["control"] = record
        _write_json(summary_path, summary)
        try:
            run_args = _run_args(
                config=args.winning_config, manifest=args.rgb_manifest,
                output=Path(record["artifacts_dir"]), teacher_cache=teacher_cache,
                cache_validation=cache_validation, class_mapping=class_mapping,
                epochs=60, temperature=2.0, weight=0.0, device=args.device,
            )
            result = xclip_lora.run_pipeline(run_args, evaluate_test=False, cache_read_only=True, print_results=False)
            _assert_run(result, expected_train=80_640, require_kd=False)
            historical = json.loads((args.historical_control / "evaluation_results.json").read_text())
            delta_points = 100 * (float(result["validation_accuracy"]) - float(historical["validation_accuracy"]))
            record.update(
                state="completed", runtime_seconds=time.time() - record["started_at_unix"],
                validation_accuracy=result["validation_accuracy"], validation_macro_f1=result["validation_macro_f1"],
                historical_validation_accuracy=historical["validation_accuracy"], delta_percentage_points=delta_points,
                parity_passed=abs(delta_points) <= 0.10, history=result["history"],
                peak_memory_allocated_bytes=result["peak_memory_allocated_bytes"],
                merge_parity_max_absolute_error=result["merge_parity_max_absolute_error"],
            )
        except BaseException as error:
            record.update(state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=f"{type(error).__name__}: {error}")
            _write_json(summary_path, summary)
            raise
        _write_json(summary_path, summary)
    summary["state"] = "completed" if summary["control"]["parity_passed"] else "failed"
    summary["completed_at_unix"] = time.time()
    _write_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SkeletonCLIP Phase 2 Gate 3")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/skeletonclip_phase2"))
    parser.add_argument("--winning-config", type=Path, default=Path("artifacts/xclip_lora_tuning/winning_config.json"))
    parser.add_argument("--rgb-manifest", type=Path, default=Path("artifacts/xclip/selected_split.json"))
    parser.add_argument("--historical-control", type=Path, default=Path("artifacts/xclip_lora_tuning/winner"))
    parser.add_argument("--smoke-epochs", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    result = run_gate3(build_parser().parse_args(argv))
    print(f"Gate 3 state={result['state']}")


if __name__ == "__main__":
    main()
