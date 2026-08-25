"""Validation-only staged hyperparameter tuning for XCLIP MIT LoRA."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections.abc import Sequence
from pathlib import Path

from vlm.training import xclip_lora


HYPERPARAMETER_KEYS = (
    "lora_learning_rate",
    "head_learning_rate",
    "warmup_epochs",
    "early_stopping_patience",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "dropout",
    "weight_decay",
    "label_smoothing",
    "frame_dropout",
    "temporal_jitter",
    "feature_noise_std",
)
RESUME_CONTROL_KEYS = {
    "artifacts_dir",
    "extract_only",
    "fail_fast",
    "force_extract",
    "max_trials",
    "preload_cache",
    "rerun_failed",
    "resume",
    "train_eval_frequency",
}


def _control() -> dict[str, object]:
    return {
        "lora_learning_rate": 1e-4,
        "head_learning_rate": 8e-4,
        "warmup_epochs": 0,
        "early_stopping_patience": None,
        "lora_rank": 8,
        "lora_alpha": 16.0,
        "lora_dropout": 0.05,
        "dropout": 0.4,
        "weight_decay": 5e-4,
        "label_smoothing": 0.05,
        "frame_dropout": 0.0,
        "temporal_jitter": 0,
        "feature_noise_std": 0.0,
    }


def _with(base: dict[str, object], **changes: object) -> dict[str, object]:
    return {**base, **changes}


def stage_candidates(
    stage: int,
    parents: Sequence[dict[str, object]],
    *,
    confirmation_seeds: Sequence[int],
) -> list[dict[str, object]]:
    """Build a curated stage without taking a Cartesian product."""
    bases = list(parents) or [_control()]
    candidates: list[dict[str, object]] = []
    for base in bases:
        if stage == 0:
            candidates.append(_control())
        elif stage == 1:
            candidates.extend(
                [
                    _with(base, lora_learning_rate=5e-5),
                    _with(base, lora_learning_rate=2e-4),
                    _with(base, head_learning_rate=4e-4),
                    _with(base, head_learning_rate=1.2e-3),
                    _with(
                        base,
                        warmup_epochs=3,
                        early_stopping_patience=8,
                    ),
                ]
            )
        elif stage == 2:
            candidates.extend(
                _with(base, lora_rank=rank, lora_alpha=float(rank * 2))
                for rank in (4, 8, 16)
            )
        elif stage == 3:
            candidates.extend(
                [
                    dict(base),
                    _with(base, lora_dropout=0.0),
                    _with(base, lora_dropout=0.1),
                    _with(base, dropout=0.25),
                    _with(base, dropout=0.55),
                    _with(base, weight_decay=1e-4),
                    _with(base, weight_decay=1e-3),
                    _with(base, label_smoothing=0.0),
                    _with(base, label_smoothing=0.1),
                    _with(base, lora_dropout=0.1, dropout=0.55),
                    _with(
                        base,
                        weight_decay=1e-3,
                        label_smoothing=0.1,
                    ),
                ]
            )
        elif stage == 4:
            candidates.extend(
                [
                    dict(base),
                    _with(base, frame_dropout=0.125),
                    _with(base, temporal_jitter=1),
                    _with(base, feature_noise_std=0.01),
                ]
            )
        elif stage == 5:
            for seed in confirmation_seeds:
                candidates.append({**base, "seed": seed})
        else:
            raise ValueError(f"Unknown tuning stage: {stage}")
    unique: dict[str, dict[str, object]] = {}
    for candidate in candidates:
        key = json.dumps(candidate, sort_keys=True, separators=(",", ":"))
        unique[key] = candidate
    return list(unique.values())


def stable_trial_id(stage: int, config: dict[str, object]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(f"{stage}:{encoded}".encode()).hexdigest()[:10]
    return f"stage-{stage}-{digest}"


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _search_identity(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: value
        for key, value in vars(args).items()
        if key not in RESUME_CONTROL_KEYS
    }


def _successful(
    records: dict[str, dict[str, object]],
    stage: int,
) -> list[dict[str, object]]:
    return [
        record
        for record in records.values()
        if record["stage"] == stage and record["state"] == "completed"
    ]


def _promote(
    records: dict[str, dict[str, object]],
    stage: int,
    *,
    maximum: int,
    margin: float,
) -> list[dict[str, object]]:
    completed = sorted(
        _successful(records, stage),
        key=lambda item: float(item["validation_accuracy"]),
        reverse=True,
    )
    if not completed:
        raise RuntimeError(f"Stage {stage} has no successful trials to promote")
    best = float(completed[0]["validation_accuracy"])
    return [
        dict(record["config"])
        for record in completed
        if float(record["validation_accuracy"]) >= best - margin
    ][:maximum]


def _recipe(config: dict[str, object]) -> dict[str, object]:
    return {key: config[key] for key in HYPERPARAMETER_KEYS if key in config}


def _select_winner(
    records: dict[str, dict[str, object]],
    final_stage: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in _successful(records, final_stage):
        recipe = _recipe(dict(record["config"]))
        key = json.dumps(recipe, sort_keys=True, separators=(",", ":"))
        grouped.setdefault(key, []).append(record)
    aggregates: list[dict[str, object]] = []
    for key, trials in grouped.items():
        scores = [float(trial["validation_accuracy"]) for trial in trials]
        runtimes = [float(trial["runtime_seconds"]) for trial in trials]
        aggregates.append(
            {
                "config": json.loads(key),
                "validation_mean": statistics.fmean(scores),
                "validation_std": statistics.pstdev(scores),
                "runtime_mean_seconds": statistics.fmean(runtimes),
                "trainable_parameters": trials[0]["trainable_parameters"],
                "seeds": [int(dict(trial["config"])["seed"]) for trial in trials],
                "trial_ids": [str(trial["trial_id"]) for trial in trials],
            }
        )
    if not aggregates:
        raise RuntimeError("No successful final-stage trials")
    aggregates.sort(
        key=lambda item: (
            float(item["validation_mean"]),
            -float(item["validation_std"]),
        ),
        reverse=True,
    )
    return dict(aggregates[0]["config"]), aggregates


def run_tuning(args: argparse.Namespace) -> dict[str, object]:
    if not args.split_manifest:
        raise ValueError("xclip-lora-tune requires --split-manifest")
    if list(args.stages) != sorted(set(args.stages)):
        raise ValueError("--stages must be unique and strictly increasing")
    if args.max_promoted < 1:
        raise ValueError("max_promoted must be at least 1")
    if args.promotion_margin < 0:
        raise ValueError("promotion_margin must be non-negative")
    if not args.confirmation_seeds:
        raise ValueError("confirmation_seeds cannot be empty")
    root = Path(args.artifacts_dir)
    summary_path = root / "search_summary.json"
    if summary_path.exists() and args.resume and args.force_extract:
        raise ValueError("--force-extract cannot be used while resuming")
    if summary_path.exists() and args.resume:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("search_identity") != _search_identity(args):
            raise ValueError(
                "Resume arguments differ from the original tuning search"
            )
        split_digest = hashlib.sha256(
            Path(args.split_manifest).read_bytes()
        ).hexdigest()
        if summary.get("split_sha256") != split_digest:
            raise ValueError("Split manifest changed since the search started")
        if summary.get("state") == "completed":
            return summary
    elif summary_path.exists():
        raise FileExistsError(
            f"{summary_path} already exists; pass --resume or choose another "
            "--artifacts-dir"
        )
    else:
        split_path = Path(args.split_manifest)
        summary = {
            "state": "running",
            "selection_metric": "validation_accuracy",
            "test_evaluated": False,
            "stages": list(args.stages),
            "arguments": vars(args).copy(),
            "search_identity": _search_identity(args),
            "git_revision": xclip_lora._git_revision(),
            "split_manifest": str(split_path),
            "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
            "search_space": {},
            "trials": {},
            "started_at_unix": time.time(),
        }
    records: dict[str, dict[str, object]] = summary["trials"]

    preparation = argparse.Namespace(**vars(args))
    preparation.extract_only = True
    preparation.artifacts_dir = str(root / "cache_preparation")
    preparation_result = xclip_lora.run_pipeline(
        preparation,
        evaluate_test=False,
    )
    cached = preparation_result["_cached_features"]
    print(
        "preloading_shared_cache "
        f"enabled={args.preload_cache} "
        f"samples={sum(len(samples) for samples in cached.values())}"
    )
    prepared_datasets = xclip_lora.build_cached_datasets(
        cached,
        preload=args.preload_cache,
    )
    print("preloading_shared_cache complete")

    new_trials = 0
    last_stage: int | None = None
    for stage in args.stages:
        if last_stage is None:
            parents = [_control()]
        else:
            parents = _promote(
                records,
                last_stage,
                maximum=args.max_promoted,
                margin=args.promotion_margin,
            )
        candidates = stage_candidates(
            stage,
            parents,
            confirmation_seeds=args.confirmation_seeds,
        )
        candidates = [
            {**candidate, "seed": candidate.get("seed", args.seed)}
            for candidate in candidates
        ]
        summary["search_space"][str(stage)] = candidates
        _write_json_atomic(summary_path, summary)
        for config in candidates:
            trial_id = stable_trial_id(stage, config)
            existing = records.get(trial_id)
            if existing:
                if existing["state"] == "completed":
                    continue
                if existing["state"] == "failed" and not args.rerun_failed:
                    continue
            if args.max_trials is not None and new_trials >= args.max_trials:
                summary["state"] = "paused"
                _write_json_atomic(summary_path, summary)
                return summary
            trial_args = argparse.Namespace(**vars(args))
            for key, value in config.items():
                setattr(trial_args, key, value)
            trial_args.extract_only = False
            trial_args.force_extract = False
            trial_args.artifacts_dir = str(root / "trials" / trial_id)
            record: dict[str, object] = {
                "trial_id": trial_id,
                "stage": stage,
                "state": "running",
                "config": config,
                "arguments": vars(trial_args).copy(),
                "artifacts_dir": trial_args.artifacts_dir,
                "started_at_unix": time.time(),
            }
            records[trial_id] = record
            _write_json_atomic(summary_path, summary)
            try:
                result = xclip_lora.run_pipeline(
                    trial_args,
                    evaluate_test=False,
                    cache_read_only=True,
                    prepared_cached=cached,
                    prepared_datasets=prepared_datasets,
                )
                record.update(
                    state="completed",
                    validation_accuracy=result["validation_accuracy"],
                    best_epoch=result["best_epoch"],
                    history=result["history"],
                    trainable_parameters=result["trainable_parameter_audit"][
                        "trainable_parameters"
                    ],
                    runtime_seconds=time.time() - float(record["started_at_unix"]),
                )
            except KeyboardInterrupt:
                record.update(
                    state="interrupted",
                    runtime_seconds=time.time() - float(record["started_at_unix"]),
                )
                summary["state"] = "interrupted"
                _write_json_atomic(summary_path, summary)
                raise
            except Exception as error:
                record.update(
                    state="failed",
                    error=f"{type(error).__name__}: {error}",
                    runtime_seconds=time.time() - float(record["started_at_unix"]),
                )
                _write_json_atomic(summary_path, summary)
                if args.fail_fast:
                    raise
            _write_json_atomic(summary_path, summary)
            new_trials += 1
        last_stage = stage

    assert last_stage is not None
    winner, aggregates = _select_winner(records, last_stage)
    winning_config = {
        **winner,
        "seed": int(aggregates[0]["seeds"][0]),
        "split_manifest": args.split_manifest,
        "videos_dir": args.videos_dir,
        "cache_dir": args.cache_dir,
        "model_name": args.model_name,
        "frames": args.frames,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "loader_workers": args.loader_workers,
        "pin_memory": args.pin_memory,
        "persistent_workers": args.persistent_workers,
        "preload_cache": args.preload_cache,
        "train_eval_frequency": args.train_eval_frequency,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_grad_norm": args.max_grad_norm,
        "amp": args.amp,
    }
    winning_path = root / "winning_config.json"
    _write_json_atomic(winning_path, winning_config)
    summary.update(
        state="winner_selected",
        aggregate_validation=aggregates,
        selected_winner=winner,
        winning_config_path=str(winning_path),
        test_evaluated=False,
    )
    _write_json_atomic(summary_path, summary)

    final_args = argparse.Namespace(**vars(args))
    for key, value in winning_config.items():
        setattr(final_args, key, value)
    final_args.extract_only = False
    final_args.force_extract = False
    final_args.artifacts_dir = str(root / "winner")
    final_started_at = time.time()
    try:
        final_result = xclip_lora.run_pipeline(
            final_args,
            evaluate_test=True,
            cache_read_only=True,
            prepared_cached=cached,
            prepared_datasets=prepared_datasets,
        )
    except Exception as error:
        summary.update(
            state="final_evaluation_failed",
            final_evaluation={
                "state": "failed",
                "error": f"{type(error).__name__}: {error}",
                "runtime_seconds": time.time() - final_started_at,
                "artifacts_dir": final_args.artifacts_dir,
            },
            test_evaluated=False,
        )
        _write_json_atomic(summary_path, summary)
        raise
    summary.update(
        state="completed",
        completed_at_unix=time.time(),
        test_evaluated=True,
        final_evaluation={
            "state": "completed",
            "seed": final_args.seed,
            "best_epoch": final_result["best_epoch"],
            "validation_accuracy": final_result["validation_accuracy"],
            "final_test_accuracy": final_result["final_test_accuracy"],
            "runtime_seconds": time.time() - final_started_at,
            "artifacts_dir": final_args.artifacts_dir,
            "evaluation_results_path": str(
                Path(final_args.artifacts_dir) / "evaluation_results.json"
            ),
        },
    )
    _write_json_atomic(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = xclip_lora.build_parser()
    parser.description = (
        "Run a resumable, staged validation-only XCLIP LoRA search."
    )
    parser.set_defaults(artifacts_dir="artifacts/xclip_lora_tuning")
    parser.set_defaults(preload_cache=True, train_eval_frequency=5)
    parser.add_argument(
        "--stages",
        type=int,
        nargs="+",
        choices=range(6),
        default=[0, 1, 2, 3, 4, 5],
    )
    parser.add_argument(
        "--confirmation-seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
    )
    parser.add_argument("--max-promoted", type=int, default=2)
    parser.add_argument("--promotion-margin", type=float, default=0.005)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    result = run_tuning(build_parser().parse_args(argv))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
