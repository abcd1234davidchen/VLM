"""Gate 1 compatibility smoke for SkeletonCLIP backbone adapters."""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

from vlm.data.skeleton import (
    PHASE1_ACTION_IDS,
    SKELETON_TENSOR_PREPROCESSING_ID,
    load_skeleton_tensor,
)
from vlm.models.skeletonclip import (
    BLOCKGCN_SOURCE_REVISION,
    HYPERFORMER_SOURCE_REVISION,
    SkeletonCheckpointIdentity,
    SkeletonLinearTeacher,
    load_skeleton_checkpoint_strict,
    save_skeleton_checkpoint,
)


SOURCE_REVISIONS = {
    "blockgcn": BLOCKGCN_SOURCE_REVISION,
    "hyperformer": HYPERFORMER_SOURCE_REVISION,
}
EVAL_PARITY_TOLERANCE = 1e-5


def _smoke_backbone(name: str, *, frames: int, seed: int) -> dict[str, object]:
    torch.manual_seed(seed)
    model = SkeletonLinearTeacher(name)  # type: ignore[arg-type]
    parameters = sum(parameter.numel() for parameter in model.parameters())
    inputs = torch.randn(2, 3, frames, 25, 2)
    labels = torch.tensor([0, 1])
    started = time.perf_counter()
    model.train()
    logits = model(inputs)
    loss = nn.functional.cross_entropy(logits.float(), labels)
    loss.backward()
    elapsed = time.perf_counter() - started
    gradients_finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )
    model.eval()
    with torch.no_grad():
        expected = model(inputs)
    identity = SkeletonCheckpointIdentity(
        backbone=name,  # type: ignore[arg-type]
        ordered_class_ids=tuple(f"A{action:03d}" for action in PHASE1_ACTION_IDS),
        preprocessing_id=SKELETON_TENSOR_PREPROCESSING_ID,
        source_revision=SOURCE_REVISIONS[name],
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        checkpoint = Path(temporary_directory) / f"{name}.pt"
        save_skeleton_checkpoint(checkpoint, model, identity)
        reloaded = SkeletonLinearTeacher(name).eval()  # type: ignore[arg-type]
        load_skeleton_checkpoint_strict(checkpoint, reloaded, identity)
        with torch.no_grad():
            actual = reloaded(inputs)
    maximum_reload_difference = (actual - expected).abs().max().item()
    return {
        "cpu_shape_test": list(logits.shape),
        "finite_loss": bool(torch.isfinite(loss).item()),
        "gradients_finite": gradients_finite,
        "strict_checkpoint_reload": maximum_reload_difference == 0.0,
        "maximum_reload_difference": maximum_reload_difference,
        "parameters": parameters,
        "forward_backward_seconds": elapsed,
        "source_revision": SOURCE_REVISIONS[name],
        "implementation": "project_minimal_adapter",
    }


def _load_overfit_batch(manifest_path: Path, *, frames: int) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected: dict[str, dict[str, object]] = {}
    for record in payload["train"]:
        class_id = str(record["class_id"])
        if class_id not in selected:
            selected[class_id] = record
        if len(selected) == 16:
            break
    if len(selected) != 16:
        raise ValueError("Paired manifest must contain at least 16 training classes")
    records = [selected[class_id] for class_id in sorted(selected)]
    inputs = torch.stack(
        [load_skeleton_tensor(Path(str(record["skeleton_path"])), frames=frames) for record in records]
    )
    labels = torch.tensor([int(record["class_index"]) for record in records])
    return inputs, labels, [str(record["sample_id"]) for record in records]


def _gpu_overfit(
    name: str,
    *,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    artifacts_dir: Path,
    seed: int,
    max_steps: int,
) -> dict[str, object]:
    device = torch.device("cuda:0")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = SkeletonLinearTeacher(name).to(device)  # type: ignore[arg-type]
    inputs = inputs.to(device)
    labels = labels.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    final_accuracy = 0.0
    final_loss = float("nan")
    completed_steps = 0
    for step in range(1, max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = nn.functional.cross_entropy(logits.float(), labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{name} produced non-finite loss at step {step}")
        loss.backward()
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise RuntimeError(f"{name} produced non-finite gradients at step {step}")
        optimizer.step()
        completed_steps = step
        final_loss = loss.item()
        if step == 1 or step % 10 == 0:
            model.eval()
            with torch.no_grad():
                evaluation_logits = model(inputs)
                final_accuracy = (
                    evaluation_logits.argmax(dim=1).eq(labels).float().mean().item()
                )
            if final_accuracy >= 0.95:
                break
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    total_memory = torch.cuda.get_device_properties(device).total_memory
    model.eval()
    with torch.no_grad():
        first_logits = model(inputs)
        second_logits = model(inputs)
    repeat_difference = (first_logits - second_logits).abs().max().item()
    identity = SkeletonCheckpointIdentity(
        backbone=name,  # type: ignore[arg-type]
        ordered_class_ids=tuple(f"A{action:03d}" for action in PHASE1_ACTION_IDS),
        preprocessing_id=SKELETON_TENSOR_PREPROCESSING_ID,
        source_revision=SOURCE_REVISIONS[name],
    )
    checkpoint = artifacts_dir / f"{name}_overfit_checkpoint.pt"
    save_skeleton_checkpoint(checkpoint, model, identity)
    reloaded = SkeletonLinearTeacher(name)  # type: ignore[arg-type]
    load_skeleton_checkpoint_strict(checkpoint, reloaded, identity)
    reloaded.to(device).eval()
    with torch.no_grad():
        reloaded_logits = reloaded(inputs)
    reload_difference = (first_logits - reloaded_logits).abs().max().item()
    del reloaded, model, optimizer, inputs, labels
    torch.cuda.empty_cache()
    return {
        "single_gpu_forward_backward": True,
        "overfit_accuracy": final_accuracy,
        "overfit_passed": final_accuracy >= 0.95,
        "steps": completed_steps,
        "final_loss": final_loss,
        "elapsed_seconds": elapsed,
        "examples_per_second": completed_steps * 16 / elapsed,
        "peak_memory_allocated_bytes": peak_allocated,
        "peak_memory_reserved_bytes": peak_reserved,
        "memory_headroom_fraction": 1.0 - peak_reserved / total_memory,
        "memory_headroom_passed": peak_reserved / total_memory <= 0.70,
        "repeat_eval_maximum_difference": repeat_difference,
        "strict_state_dict_reload": True,
        "gpu_reload_within_tolerance": reload_difference <= EVAL_PARITY_TOLERANCE,
        "gpu_reload_maximum_difference": reload_difference,
        "repeat_eval_within_tolerance": repeat_difference <= EVAL_PARITY_TOLERANCE,
        "eval_parity_atol_rtol": EVAL_PARITY_TOLERANCE,
        "checkpoint": str(checkpoint),
    }


def run_gate1_smoke(
    *,
    artifacts_dir: Path,
    manifest_path: Path,
    frames: int = 64,
    seed: int = 42,
    max_steps: int = 500,
    cpu_only: bool = False,
) -> dict[str, object]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    candidates = {
        name: _smoke_backbone(name, frames=frames, seed=seed)
        for name in ("blockgcn", "hyperformer")
    }
    cuda_available = torch.cuda.is_available()
    gpu_results: dict[str, object] = {}
    sample_ids: list[str] = []
    if cuda_available and not cpu_only:
        inputs, labels, sample_ids = _load_overfit_batch(manifest_path, frames=frames)
        for name in ("blockgcn", "hyperformer"):
            gpu_results[name] = _gpu_overfit(
                name,
                inputs=inputs,
                labels=labels,
                artifacts_dir=artifacts_dir,
                seed=seed,
                max_steps=max_steps,
            )
            candidates[name]["gpu"] = gpu_results[name]
    complete = bool(gpu_results) and all(
        result["overfit_passed"]
        and result["memory_headroom_passed"]
        and result["strict_state_dict_reload"]
        and result["gpu_reload_within_tolerance"]
        and result["repeat_eval_within_tolerance"]
        for result in gpu_results.values()
    )
    report = {
        "schema_version": 1,
        "status": "passed" if complete else "incomplete",
        "ordered_class_ids": [f"A{action:03d}" for action in PHASE1_ACTION_IDS],
        "excluded_action_ids": ["A060"],
        "input_contract": ["N", 3, "T", 25, 2],
        "preprocessing_id": SKELETON_TENSOR_PREPROCESSING_ID,
        "paired_manifest": str(manifest_path),
        "overfit_sample_ids": sample_ids,
        "seed": seed,
        "frames": frames,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": cuda_available,
            "torch_cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        },
        "candidates": candidates,
        "pending": [] if complete else [
            "single-GPU forward/backward",
            "16-sample overfit >=95%",
            "GPU peak memory",
            "GPU examples/sec",
        ],
        "blocker": (
            "CUDA is unavailable; host nvidia-smi preflight reported an NVML "
            "driver/library version mismatch (library 580.178)."
            if not cuda_available
            else None
        ),
    }
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "compatibility_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SkeletonCLIP Gate 1 compatibility smoke.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/skeletonclip_gate1"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/skeleton_data/paired_manifest.json"))
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--cpu-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run_gate1_smoke(
        artifacts_dir=args.artifacts_dir,
        manifest_path=args.manifest,
        frames=args.frames,
        seed=args.seed,
        max_steps=args.max_steps,
        cpu_only=args.cpu_only,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
