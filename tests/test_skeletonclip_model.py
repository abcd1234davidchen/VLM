from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from vlm.models.skeletonclip import (
    BLOCKGCN_SOURCE_REVISION,
    SkeletonCheckpointIdentity,
    SkeletonLinearTeacher,
    load_skeleton_checkpoint_strict,
    save_skeleton_checkpoint,
)


class SkeletonCLIPModelTest(unittest.TestCase):
    def test_backbones_share_batch_and_logit_contract(self) -> None:
        batch = torch.randn(2, 3, 8, 25, 2)
        for backbone_name in ("blockgcn", "hyperformer"):
            with self.subTest(backbone=backbone_name):
                model = SkeletonLinearTeacher(backbone_name)
                model.eval()
                with torch.no_grad():
                    logits = model(batch)
                self.assertEqual(logits.shape, (2, 59))
                self.assertTrue(torch.isfinite(logits).all())

    def test_checkpoint_reload_is_strict_and_reproducible(self) -> None:
        torch.manual_seed(7)
        batch = torch.randn(2, 3, 8, 25, 2)
        model = SkeletonLinearTeacher("blockgcn").eval()
        identity = SkeletonCheckpointIdentity(
            backbone="blockgcn",
            ordered_class_ids=tuple(f"A{index:03d}" for index in range(1, 60)),
            preprocessing_id="ntu-rgbd-parser-v1",
            source_revision=BLOCKGCN_SOURCE_REVISION,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "checkpoint.pt"
            with torch.no_grad():
                expected = model(batch)
            save_skeleton_checkpoint(path, model, identity)
            reloaded = SkeletonLinearTeacher("blockgcn").eval()
            load_skeleton_checkpoint_strict(path, reloaded, identity)
            with torch.no_grad():
                actual = reloaded(batch)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_batch_contract_rejects_non_finite_input(self) -> None:
        model = SkeletonLinearTeacher("blockgcn")
        batch = torch.zeros(1, 3, 8, 25, 2)
        batch[0, 0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            model(batch)


if __name__ == "__main__":
    unittest.main()
