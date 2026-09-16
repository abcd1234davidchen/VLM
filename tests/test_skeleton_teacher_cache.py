from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from vlm.skeleton_teacher_cache import (
    DistillationAccumulator,
    TeacherLogitCache,
    build_class_mapping,
    canonical_sha256,
    masked_kl_sum,
    validate_class_mapping,
)


class SkeletonTeacherCacheTest(unittest.TestCase):
    def test_class_mapping_is_explicit_and_round_trips(self) -> None:
        mapping = build_class_mapping(
            [f"A{index:03d}" for index in range(1, 60)],
            list(range(1, 121)),
        )
        self.assertEqual(mapping["student_indices"], list(range(59)))
        mapping["mapping_sha256"] = canonical_sha256(mapping)
        self.assertTrue(torch.equal(validate_class_mapping(mapping), torch.arange(59)))
        mapping["student_indices"] = mapping["student_indices"][1:] + [0]
        with self.assertRaises(ValueError):
            validate_class_mapping(mapping)
        with self.assertRaises(ValueError):
            build_class_mapping(["A002", "A001"], list(range(1, 121)))

    def test_empty_pairing_is_differentiable_zero(self) -> None:
        student = torch.randn(3, 120, requires_grad=True)
        value, count = masked_kl_sum(
            student, torch.randn(3, 59), torch.zeros(3, dtype=torch.bool),
            torch.arange(59), temperature=2,
        )
        value.backward()
        self.assertEqual(count, 0)
        self.assertEqual(value.item(), 0)
        self.assertTrue(torch.equal(student.grad, torch.zeros_like(student)))

    def test_accumulation_partition_has_identical_loss_and_gradients(self) -> None:
        torch.manual_seed(4)
        original = torch.randn(7, 120)
        labels = torch.tensor([0, 1, 70, 2, 90, 3, 4])
        teacher = torch.randn(7, 59)
        mask = torch.tensor([1, 1, 0, 1, 0, 0, 1], dtype=torch.bool)
        gradients = []
        for slices in ([slice(None)], [slice(0, 2), slice(2, 6), slice(6, 7)]):
            student = original.clone().requires_grad_()
            accumulator = DistillationAccumulator()
            for part in slices:
                accumulator.add(student[part], labels[part], teacher[part], mask[part], torch.arange(59), temperature=4)
            losses = accumulator.losses(kd_weight=0.5)
            losses[-1].backward(); gradients.append((tuple(float(loss.detach()) for loss in losses), student.grad))
        self.assertTrue(torch.allclose(gradients[0][1], gradients[1][1], atol=1e-7))
        self.assertEqual(gradients[0][0], gradients[1][0])

    def test_cache_identity_and_duplicate_ids_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.pt"
            identity = {"checkpoint": "abc"}
            torch.save({"identity": identity, "sample_ids": ["one"], "logits": torch.zeros(1, 59)}, path)
            cache = TeacherLogitCache(path, expected_identity=identity)
            logits, mask = cache.batch(["one", "missing"])
            self.assertEqual(logits.shape, (2, 59)); self.assertEqual(mask.tolist(), [True, False])
            with self.assertRaises(ValueError): TeacherLogitCache(path, expected_identity={"checkpoint": "changed"})
            torch.save({"identity": identity, "sample_ids": ["one", "one"], "logits": torch.zeros(2, 59)}, path)
            with self.assertRaises(ValueError): TeacherLogitCache(path, expected_identity=identity)


if __name__ == "__main__": unittest.main()
