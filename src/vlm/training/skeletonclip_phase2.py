"""Execute SkeletonCLIP Phase 2 Gates 0-2 without touching the test split."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from vlm.data.skeleton import SKELETON_TENSOR_PREPROCESSING_ID, load_skeleton_tensor
from vlm.models.skeletonclip import SkeletonLinearTeacher
from vlm.skeleton_teacher_cache import (
    TEACHER_CLASS_IDS,
    DistillationAccumulator,
    build_class_mapping,
    canonical_sha256,
    sha256_file,
    validate_class_mapping,
)


SCHEMA_VERSION = 1
EXPECTED_TRAIN_PAIRED = 45_236
EXPECTED_VAL_PAIRED = 4_180


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_revision() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _manifest_body_hash(payload: dict[str, object]) -> str:
    return canonical_sha256({key: value for key, value in payload.items() if key != "manifest_sha256"})


def run_gate0(*, artifacts_dir: Path, teacher_checkpoint: Path, paired_manifest: Path, rgb_manifest: Path, winning_config: Path, specification: Path) -> dict[str, object]:
    paired = json.loads(paired_manifest.read_text(encoding="utf-8"))
    rgb = json.loads(rgb_manifest.read_text(encoding="utf-8"))
    config = json.loads(winning_config.read_text(encoding="utf-8"))
    if paired.get("manifest_sha256") != _manifest_body_hash(paired):
        raise ValueError("Paired manifest embedded hash is stale or invalid")
    if paired.get("ordered_class_ids") != list(TEACHER_CLASS_IDS) or paired.get("excluded_action_ids") != ["A060"]:
        raise ValueError("Paired manifest class contract is not A001-A059 with A060 excluded")
    if len(paired.get("train", [])) != EXPECTED_TRAIN_PAIRED or len(paired.get("val", [])) != EXPECTED_VAL_PAIRED:
        raise ValueError("Paired manifest counts differ from the locked Phase 2 contract")
    split_ids: dict[str, set[str]] = {}
    student_class_ids: set[int] = set()
    for split in ("train", "val", "test"):
        rows = rgb.get(split)
        if not isinstance(rows, list):
            raise ValueError(f"Canonical RGB manifest has no {split} list")
        ids = {Path(str(row["video_path"])).stem.removesuffix("_rgb") for row in rows}
        if len(ids) != len(rows):
            raise ValueError(f"Duplicate identity in RGB {split} split")
        split_ids[split] = ids
        student_class_ids.update(int(row["class_id"]) for row in rows)
    if {name: len(ids) for name, ids in split_ids.items()} != {"train": 80640, "val": 12960, "test": 20880}:
        raise ValueError("Canonical split counts differ from the locked Phase 2 contract")
    if any(split_ids[left] & split_ids[right] for left, right in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Canonical RGB split identities overlap")
    for split in ("train", "val"):
        paired_ids = {str(row["sample_id"]) for row in paired[split]}
        if len(paired_ids) != len(paired[split]) or not paired_ids <= split_ids[split]:
            raise ValueError(f"Paired {split} identities are duplicated or outside canonical split")
    mapping = build_class_mapping(paired["ordered_class_ids"], sorted(student_class_ids))
    checkpoint = torch.load(teacher_checkpoint, map_location="cpu", weights_only=False)
    identity = checkpoint.get("identity", {})
    expected_identity = {
        "backbone": "blockgcn", "head": "linear", "ordered_class_ids": list(TEACHER_CLASS_IDS),
        "excluded_class_ids": ["A060"], "preprocessing_id": SKELETON_TENSOR_PREPROCESSING_ID,
    }
    for key, expected in expected_identity.items():
        if identity.get(key) != expected:
            raise ValueError(f"Teacher checkpoint identity mismatch for {key}")
    if identity.get("manifest_sha256") != sha256_file(paired_manifest):
        raise ValueError("Teacher checkpoint was not trained from this exact paired manifest")
    model = SkeletonLinearTeacher("blockgcn")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    source_hashes = {
        "teacher_checkpoint": {"path": str(teacher_checkpoint), "sha256": sha256_file(teacher_checkpoint)},
        "paired_manifest": {"path": str(paired_manifest), "sha256": sha256_file(paired_manifest)},
        "canonical_rgb_split": {"path": str(rgb_manifest), "sha256": sha256_file(rgb_manifest)},
        "student_winning_config": {"path": str(winning_config), "sha256": sha256_file(winning_config)},
        "phase2_specification": {"path": str(specification), "sha256": sha256_file(specification)},
    }
    mapping.update({"source_hashes": source_hashes, "mapping_sha256": canonical_sha256(mapping)})
    lock = {
        "schema_version": SCHEMA_VERSION, "status": "gate0_passed", "git_revision": _git_revision(),
        "source_hashes": source_hashes, "teacher_identity": identity, "student_recipe": config,
        "split_counts": {name: len(ids) for name, ids in split_ids.items()},
        "paired_counts": {"train": len(paired["train"]), "val": len(paired["val"])},
        "search_space": {"temperatures": [1, 2, 4], "kd_weights": [0.1, 0.5, 1.0]},
        "seed_policy": {"search": 42, "confirmation": 43},
        "decision_threshold_percentage_points": {"advance_accuracy": 0.20, "per_seed_floor": -0.20, "a060_a120_floor": -0.50},
        "test_embargo": "Gate 0-4 commands must not load or evaluate test samples; Gate 5 test-once only.",
    }
    lock["lock_sha256"] = canonical_sha256(lock)
    _write_json(artifacts_dir / "class_mapping.json", mapping)
    _write_json(artifacts_dir / "experiment_lock.json", lock)
    _write_json(artifacts_dir / "identity_audit.json", {"status": "passed", **lock})
    return lock


class _SkeletonRows(Dataset):
    def __init__(self, rows: list[dict[str, object]], frames: int) -> None:
        self.rows, self.frames = rows, frames
    def __len__(self) -> int:
        return len(self.rows)
    def __getitem__(self, index: int) -> tuple[Tensor, str, str]:
        row = self.rows[index]
        path = Path(str(row["skeleton_path"]))
        return load_skeleton_tensor(path, frames=self.frames), str(row["sample_id"]), sha256_file(path)


def _cache_identity(lock: dict[str, object]) -> dict[str, object]:
    hashes = lock["source_hashes"]
    return {
        "schema_version": 1, "checkpoint_sha256": hashes["teacher_checkpoint"]["sha256"],
        "paired_manifest_sha256": hashes["paired_manifest"]["sha256"],
        "ordered_class_ids": list(TEACHER_CLASS_IDS), "preprocessing_id": SKELETON_TENSOR_PREPROCESSING_ID,
        "sampling_policy": "nearest-whole-clip-64", "dtype": "float32", "tensor_shape": [59],
    }


def run_gate1(*, artifacts_dir: Path, cache_root: Path, teacher_checkpoint: Path, paired_manifest: Path, batch_size: int, workers: int, device: str, parity_samples: int) -> dict[str, object]:
    lock = json.loads((artifacts_dir / "experiment_lock.json").read_text())
    identity = _cache_identity(lock)
    cache_dir = cache_root / canonical_sha256(identity)[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "teacher_logits.pt"
    paired = json.loads(paired_manifest.read_text())
    rows = paired["train"]
    if len(rows) != EXPECTED_TRAIN_PAIRED:
        raise ValueError("Gate 1 requires exactly 45,236 paired train rows")
    torch_device = torch.device(device)
    checkpoint = torch.load(teacher_checkpoint, map_location="cpu", weights_only=False)
    model = SkeletonLinearTeacher("blockgcn").to(torch_device).eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    loader = DataLoader(_SkeletonRows(rows, 64), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=torch_device.type == "cuda")
    all_logits, sample_ids, source_hashes = [], [], []
    if cache_path.exists():
        existing = torch.load(cache_path, map_location="cpu", weights_only=True)
        if existing.get("identity") != identity:
            raise ValueError("Existing teacher cache has a stale identity")
        expected_ids = [str(row["sample_id"]) for row in rows]
        if existing.get("sample_ids") != expected_ids:
            raise ValueError("Existing teacher cache membership/order differs from paired manifest")
        current_hashes = [sha256_file(Path(str(row["skeleton_path"]))) for row in rows]
        if existing.get("source_sha256") != current_hashes:
            raise ValueError("Existing teacher cache source skeleton fingerprint mismatch")
        logits = existing["logits"].float(); sample_ids = expected_ids; source_hashes = current_hashes
        print(f"gate1 reused validated cache samples={len(sample_ids)}", flush=True)
    else:
        shards_dir = cache_dir / "shards"; shards_dir.mkdir(exist_ok=True)
        with torch.inference_mode():
            for batch_index, (inputs, ids, hashes) in enumerate(loader, start=1):
                shard_path = shards_dir / f"{batch_index:06d}.pt"
                shard_identity = {"cache_identity": identity, "sample_ids": list(ids), "source_sha256": list(hashes)}
                if shard_path.exists():
                    shard = torch.load(shard_path, map_location="cpu", weights_only=True)
                    if shard.get("identity") != shard_identity:
                        raise ValueError(f"Stale interrupted cache shard: {shard_path}")
                    batch_logits = shard["logits"]
                else:
                    batch_logits = model(inputs.to(torch_device, non_blocking=True)).float().cpu()
                    torch.save({"identity": shard_identity, "logits": batch_logits}, shard_path)
                if batch_logits.shape != (len(ids), 59) or batch_logits.dtype != torch.float32 or not torch.isfinite(batch_logits).all():
                    raise RuntimeError(f"Invalid teacher output in batch {batch_index}")
                all_logits.append(batch_logits); sample_ids.extend(ids); source_hashes.extend(hashes)
                if batch_index % 100 == 0:
                    print(f"gate1 batches={batch_index}/{len(loader)} samples={len(sample_ids)}", flush=True)
        logits = torch.cat(all_logits)
    if len(sample_ids) != EXPECTED_TRAIN_PAIRED or len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError("Teacher cache coverage or uniqueness failed")
    payload_fingerprints = [canonical_sha256({"sample_id": sample_id, "source_sha256": source_hash, "logits": row.tolist()}) for sample_id, source_hash, row in zip(sample_ids, source_hashes, logits)]
    torch.save({"identity": identity, "sample_ids": sample_ids, "source_sha256": source_hashes, "payload_fingerprints": payload_fingerprints, "logits": logits}, cache_path)
    rng = random.Random(42)
    parity_indices = sorted(rng.sample(range(len(rows)), min(parity_samples, len(rows))))
    parity_dataset = _SkeletonRows([rows[index] for index in parity_indices], 64)
    parity_loader = DataLoader(parity_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    online = []
    with torch.inference_mode():
        for inputs, _ids, _hashes in parity_loader:
            online.append(model(inputs.to(torch_device)).float().cpu())
    parity_error = float((torch.cat(online) - logits[parity_indices]).abs().max().item())
    first = load_skeleton_tensor(Path(str(rows[0]["skeleton_path"])), frames=64).unsqueeze(0).to(torch_device)
    with torch.inference_mode():
        determinism_error = float((model(first).float() - model(first).float()).abs().max().item())
    probabilities = logits.softmax(dim=1)
    report = {
        "schema_version": 1, "status": "passed" if parity_error <= 1e-5 and determinism_error == 0 else "failed",
        "cache_path": str(cache_path), "cache_sha256": sha256_file(cache_path), "cache_identity": identity, "samples": len(sample_ids), "missing": 0, "extra": 0, "duplicates": 0,
        "dtype": str(logits.dtype).removeprefix("torch."), "shape": list(logits.shape), "finite": bool(torch.isfinite(logits).all()),
        "logit_min": float(logits.min()), "logit_max": float(logits.max()),
        "mean_entropy": float((-(probabilities * probabilities.clamp_min(1e-12).log()).sum(1)).mean()),
        "argmax_distribution": dict(sorted(Counter(logits.argmax(1).tolist()).items())),
        "online_parity_samples": len(parity_indices), "online_parity_max_abs_diff": parity_error,
        "determinism_max_abs_diff": determinism_error,
    }
    _write_json(cache_dir / "manifest.json", {**report, "payload_fingerprints": payload_fingerprints})
    _write_json(artifacts_dir / "cache_validation.json", report)
    if report["status"] != "passed": raise RuntimeError("Gate 1 validation failed")
    return report


class _SmokeStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__(); self.base = nn.Linear(7, 7); self.base.requires_grad_(False); self.lora = nn.Linear(7, 7, bias=False); self.head = nn.Linear(7, 120)
    def forward(self, inputs: Tensor) -> Tensor: return self.head(torch.relu(self.base(inputs) + self.lora(inputs)))


def _update(inputs: Tensor, labels: Tensor, teacher: Tensor, mask: Tensor, mapping: Tensor, splits: list[slice], kd_weight: float) -> tuple[dict[str, Tensor], dict[str, float]]:
    torch.manual_seed(7); model = _SmokeStudent(); model.eval(); optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.01)
    accumulator = DistillationAccumulator()
    for part in splits:
        accumulator.add(model(inputs[part]), labels[part], teacher[part], mask[part], mapping, temperature=2.0, label_smoothing=0.05)
    ce, kd, total = accumulator.losses(kd_weight=kd_weight); total.backward(); optimizer.step()
    if any(p.grad is not None for p in model.base.parameters()): raise RuntimeError("Frozen base received gradients")
    return {name: value.detach().clone() for name, value in model.state_dict().items()}, {"ce": float(ce.detach()), "kd": float(kd.detach()), "total": float(total.detach()), "paired": accumulator.paired_count, "samples": accumulator.total_count}


def run_gate2(*, artifacts_dir: Path) -> dict[str, object]:
    mapping_payload = json.loads((artifacts_dir / "class_mapping.json").read_text())
    mapping = torch.tensor(mapping_payload["student_indices"], dtype=torch.long)
    torch.manual_seed(11); inputs = torch.randn(8, 7); labels = torch.tensor([0, 60, 1, 119, 59, 2, 80, 3]); teacher = torch.randn(8, 59); mask = torch.tensor([1, 0, 1, 0, 0, 1, 0, 1], dtype=torch.bool)
    full_state, full_metrics = _update(inputs, labels, teacher, mask, mapping, [slice(None)], 0.5)
    split_state, split_metrics = _update(inputs, labels, teacher, mask, mapping, [slice(0, 3), slice(3, 5), slice(5, 8)], 0.5)
    accumulation_error = max(float((full_state[key] - split_state[key]).abs().max()) for key in full_state)
    zero_state, zero_metrics = _update(inputs, labels, teacher, mask, mapping, [slice(None)], 0.0)
    ce_state, ce_metrics = _update(inputs, labels, teacher, torch.zeros_like(mask), mapping, [slice(None)], 0.0)
    lambda_zero_error = max(float((zero_state[key] - ce_state[key]).abs().max()) for key in zero_state)
    unpaired_state, unpaired_metrics = _update(inputs, labels, teacher, torch.zeros_like(mask), mapping, [slice(0, 4), slice(4, 8)], 1.0)
    bad_mapping_failed = False
    shuffled_mapping = dict(mapping_payload)
    shuffled_mapping["student_indices"] = mapping_payload["student_indices"][1:] + mapping_payload["student_indices"][:1]
    try: validate_class_mapping(shuffled_mapping)
    except ValueError: bad_mapping_failed = True
    checkpoint = artifacts_dir / "smoke_student.pt"; torch.save({"state_dict": full_state, "teacher_provenance": mapping_payload["source_hashes"]}, checkpoint)
    reloaded = _SmokeStudent(); reloaded.load_state_dict(torch.load(checkpoint, weights_only=True)["state_dict"], strict=True)
    reload_error = max(float((full_state[key] - reloaded.state_dict()[key]).abs().max()) for key in full_state)
    report = {
        "schema_version": 1, "status": "passed", "lambda_zero_update_max_abs_diff": lambda_zero_error,
        "accumulation_update_max_abs_diff": accumulation_error, "strict_reload_max_abs_diff": reload_error,
        "mapping_shuffle_fails_fast": bad_mapping_failed, "mixed_batch": full_metrics, "split_mixed_batch": split_metrics,
        "lambda_zero": zero_metrics, "ce_reference": ce_metrics, "all_unpaired": unpaired_metrics,
        "trainable_parameter_contract": ["lora.weight", "head.weight", "head.bias"], "rgb_only_checkpoint": str(checkpoint),
    }
    if lambda_zero_error > 1e-6 or accumulation_error > 1e-6 or reload_error != 0 or not bad_mapping_failed or unpaired_metrics["kd"] != 0:
        report["status"] = "failed"
    _write_json(artifacts_dir / "integration_smoke.json", report)
    if report["status"] != "passed": raise RuntimeError("Gate 2 integration smoke failed")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SkeletonCLIP Phase 2 Gates 0-2")
    parser.add_argument("--through-gate", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/skeletonclip_phase2"))
    parser.add_argument("--cache-root", type=Path, default=Path("cache/skeleton_teacher_logits"))
    parser.add_argument("--teacher-checkpoint", type=Path, default=Path("artifacts/skeletonclip_gate3/linear_seed42/best_checkpoint.pt"))
    parser.add_argument("--paired-manifest", type=Path, default=Path("artifacts/skeleton_data/paired_manifest.json"))
    parser.add_argument("--rgb-manifest", type=Path, default=Path("artifacts/xclip/selected_split.json"))
    parser.add_argument("--winning-config", type=Path, default=Path("artifacts/xclip_lora_tuning/winning_config.json"))
    parser.add_argument("--specification", type=Path, default=Path("docs/skeletonclip-distillation-phase2.md"))
    parser.add_argument("--device", default="cuda"); parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--workers", type=int, default=4); parser.add_argument("--parity-samples", type=int, default=128)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    lock = run_gate0(artifacts_dir=args.artifacts_dir, teacher_checkpoint=args.teacher_checkpoint, paired_manifest=args.paired_manifest, rgb_manifest=args.rgb_manifest, winning_config=args.winning_config, specification=args.specification)
    print(f"Gate 0: {lock['status']}", flush=True)
    if args.through_gate >= 1:
        report = run_gate1(artifacts_dir=args.artifacts_dir, cache_root=args.cache_root, teacher_checkpoint=args.teacher_checkpoint, paired_manifest=args.paired_manifest, batch_size=args.batch_size, workers=args.workers, device=args.device, parity_samples=args.parity_samples); print(f"Gate 1: {report['status']}", flush=True)
    if args.through_gate >= 2:
        report = run_gate2(artifacts_dir=args.artifacts_dir); print(f"Gate 2: {report['status']}", flush=True)


if __name__ == "__main__": main()
