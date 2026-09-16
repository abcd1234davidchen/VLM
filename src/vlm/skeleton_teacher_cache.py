"""Phase 2 frozen-teacher cache and masked-KL contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from vlm.data.skeleton import parse_skeleton_identity


TEACHER_CLASS_IDS = tuple(f"A{index:03d}" for index in range(1, 60))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_class_mapping(
    teacher_class_ids: Sequence[str], student_class_ids: Sequence[int]
) -> dict[str, object]:
    expected = list(TEACHER_CLASS_IDS)
    if list(teacher_class_ids) != expected:
        raise ValueError("Teacher class order must be exactly A001-A059")
    if len(set(student_class_ids)) != len(student_class_ids):
        raise ValueError("Student class IDs must be unique")
    index_by_id = {f"A{class_id:03d}": index for index, class_id in enumerate(student_class_ids)}
    missing = [class_id for class_id in expected if class_id not in index_by_id]
    if missing:
        raise ValueError("Student classes are missing teacher IDs: " + ", ".join(missing))
    indices = [index_by_id[class_id] for class_id in expected]
    if len(set(indices)) != 59 or any(student_class_ids[index] != offset + 1 for offset, index in enumerate(indices)):
        raise ValueError("Teacher/student class mapping failed round-trip validation")
    return {
        "schema_version": 1,
        "teacher_class_ids": expected,
        "student_class_ids": list(student_class_ids),
        "student_indices": indices,
        "mapping": dict(zip(expected, indices)),
    }


def validate_class_mapping(payload: dict[str, object]) -> Tensor:
    expected = build_class_mapping(
        payload.get("teacher_class_ids", ()),
        payload.get("student_class_ids", ()),
    )
    if any(
        payload.get(key) != expected[key]
        for key in (
            "teacher_class_ids",
            "student_class_ids",
            "student_indices",
            "mapping",
        )
    ) or payload.get("mapping_sha256") != canonical_sha256(expected):
        raise ValueError("Class mapping artifact failed identity/round-trip validation")
    return torch.tensor(expected["student_indices"], dtype=torch.long)


def masked_kl_sum(
    student_logits: Tensor,
    teacher_logits: Tensor,
    paired_mask: Tensor,
    class_indices: Tensor,
    *,
    temperature: float,
) -> tuple[Tensor, int]:
    """Return the unnormalised paired KL sum and paired row count in FP32."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if student_logits.ndim != 2 or teacher_logits.ndim != 2:
        raise ValueError("Student and teacher logits must be rank-2")
    if paired_mask.shape != (student_logits.shape[0],):
        raise ValueError("paired_mask does not match batch size")
    if class_indices.shape != (teacher_logits.shape[1],):
        raise ValueError("class mapping length does not match teacher logits")
    if class_indices.unique().numel() != class_indices.numel():
        raise ValueError("class mapping contains duplicate student indices")
    if class_indices.min().item() < 0 or class_indices.max().item() >= student_logits.shape[1]:
        raise ValueError("class mapping contains an out-of-range student index")
    paired_count = int(paired_mask.sum().item())
    if paired_count == 0:
        return student_logits.sum() * 0.0, 0
    selected_teacher = teacher_logits[paired_mask].detach().float()
    if selected_teacher.shape[0] != paired_count or not torch.isfinite(selected_teacher).all():
        raise ValueError("Paired teacher logits are missing or non-finite")
    selected_student = student_logits[paired_mask].float().index_select(1, class_indices)
    teacher_probabilities = torch.softmax(selected_teacher / temperature, dim=1)
    student_log_probabilities = torch.log_softmax(selected_student / temperature, dim=1)
    per_row = torch.sum(
        teacher_probabilities
        * (torch.log(teacher_probabilities.clamp_min(1e-12)) - student_log_probabilities),
        dim=1,
    )
    return per_row.sum() * temperature**2, paired_count


@dataclass
class DistillationAccumulator:
    """Accumulate loss numerators so micro-batch boundaries do not reweight rows."""

    ce_sum: Tensor | None = None
    kd_sum: Tensor | None = None
    total_count: int = 0
    paired_count: int = 0

    def add(
        self,
        student_logits: Tensor,
        labels: Tensor,
        teacher_logits: Tensor,
        paired_mask: Tensor,
        class_indices: Tensor,
        *,
        temperature: float,
        label_smoothing: float = 0.0,
    ) -> None:
        ce = nn.functional.cross_entropy(
            student_logits.float(), labels, reduction="sum", label_smoothing=label_smoothing
        )
        kd, paired = masked_kl_sum(
            student_logits, teacher_logits, paired_mask, class_indices, temperature=temperature
        )
        self.ce_sum = ce if self.ce_sum is None else self.ce_sum + ce
        self.kd_sum = kd if self.kd_sum is None else self.kd_sum + kd
        self.total_count += labels.numel()
        self.paired_count += paired

    def losses(self, *, kd_weight: float) -> tuple[Tensor, Tensor, Tensor]:
        if self.ce_sum is None or self.kd_sum is None or self.total_count == 0:
            raise ValueError("No micro-batches were accumulated")
        ce = self.ce_sum / self.total_count
        kd = self.kd_sum / self.paired_count if self.paired_count else self.kd_sum
        return ce, kd, ce + kd_weight * kd


class TeacherLogitCache:
    """Fail-closed in-memory lookup for a validated consolidated cache."""

    def __init__(self, path: Path, *, expected_identity: dict[str, object]) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("identity") != expected_identity:
            raise ValueError(f"Teacher cache identity mismatch: {path}")
        sample_ids = payload.get("sample_ids")
        logits = payload.get("logits")
        if not isinstance(sample_ids, list) or len(sample_ids) != len(set(sample_ids)):
            raise ValueError("Teacher cache sample IDs are invalid or duplicated")
        if not isinstance(logits, Tensor) or logits.dtype != torch.float32 or logits.shape != (len(sample_ids), 59):
            raise ValueError("Teacher cache logits must be FP32 [N, 59]")
        if not torch.isfinite(logits).all():
            raise ValueError("Teacher cache contains non-finite logits")
        self._index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
        self._logits = logits

    def batch(self, sample_ids: Sequence[str]) -> tuple[Tensor, Tensor]:
        rows, mask = [], []
        for sample_id in sample_ids:
            index = self._index.get(sample_id)
            mask.append(index is not None)
            rows.append(self._logits[index] if index is not None else torch.zeros(59))
        return torch.stack(rows), torch.tensor(mask, dtype=torch.bool)


class DistillationDataset(Dataset):
    """Attach immutable sample identities and cached teacher logits to RGB features."""

    def __init__(self, base: Dataset, samples: Sequence[object], cache: TeacherLogitCache) -> None:
        if len(base) != len(samples):
            raise ValueError("RGB dataset and sample metadata lengths differ")
        self.base, self.samples, self.cache = base, samples, cache

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, str, Tensor, Tensor]:
        features, label = self.base[index]
        video_path = Path(str(getattr(self.samples[index], "video_path")))
        sample_id = parse_skeleton_identity(video_path).sample_id
        logits, mask = self.cache.batch([sample_id])
        return features, label, sample_id, logits[0], mask[0]
