"""Guard the Phase 4 supervision mask and patch geometry contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from vlm.spatial_joint_targets import build_motion_spatial_targets, build_spatial_targets, spatial_attention_loss, spatial_region_loss


class SpatialJointTargetsTest(unittest.TestCase):
    def test_motion_removes_translation_and_scale_and_masks_static_frames(self) -> None:
        def sequence(path: Path, moving: bool, transformed: bool) -> None:
            lines = ["3"]
            for frame in range(3):
                lines.extend(["1", "123 0 0 0 0 0 0 0 0 2", "25"])
                for joint in range(25):
                    x = frame * 0.1 if moving and joint == 7 else 0.0
                    y = joint / 20
                    scale = 1 + frame if transformed else 1
                    shift = 5 * frame if transformed else 0
                    color_x = 208 if joint == 7 else 16
                    lines.append(f"{x * scale + shift} {y * scale + shift} {shift} 0 0 {color_x} 16 1 0 0 0 2")
            path.write_text("\n".join(lines) + "\n")

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "motion.skeleton"
            sequence(path, True, False)
            reference = build_motion_spatial_targets(path, [0, 1, 2], source_width=224, source_height=224)
            sequence(path, True, True)
            transformed = build_motion_spatial_targets(path, [0, 1, 2], source_width=224, source_height=224)
            sequence(path, False, True)
            static = build_motion_spatial_targets(path, [0, 1, 2], source_width=224, source_height=224)
        self.assertEqual(reference.frame_mask.tolist(), [True, True, True])
        self.assertTrue(torch.allclose(reference.distribution, transformed.distribution))
        self.assertEqual(reference.distribution.argmax(-1).tolist(), [6, 6, 6])
        self.assertTrue(torch.allclose(reference.body_part_mass.sum((1, 2)), reference.distribution))
        self.assertFalse(bool(static.frame_mask.any()))
        self.assertEqual(float(static.distribution.sum()), 0)
        logits = torch.randn(1, 3, 49, requires_grad=True)
        loss = spatial_attention_loss(logits, reference.distribution.unsqueeze(0), reference.frame_mask.unsqueeze(0))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        zero = spatial_attention_loss(logits, static.distribution.unsqueeze(0), static.frame_mask.unsqueeze(0))
        self.assertEqual(float(zero.detach()), 0)

    def test_tracked_joints_map_to_patches_and_missing_frame_has_no_loss(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "S001C001P001R001A001.skeleton"
            lines = ["2"]
            for frame in range(2):
                lines.extend(["1", "123 0 0 0 0 0 0 0 0 2", "25"])
                for joint in range(25):
                    x, y = (223, 223) if joint == 10 else (16, 16)
                    state = 2 if frame == 0 and joint <= 10 else 0
                    lines.append(f"0 0 0 0 0 {x} {y} 1 0 0 0 {state}")
            path.write_text("\n".join(lines) + "\n")
            target = build_spatial_targets(path, [0, 1], source_width=224, source_height=224)

        self.assertEqual(target.body_ids, (123,))
        self.assertEqual(target.tracked_in_crop, (11, 0))
        self.assertEqual(target.frame_mask.tolist(), [True, False])
        self.assertEqual(target.occupancy[0].nonzero().flatten().tolist(), [0, 48])
        self.assertEqual(int(target.occupancy[1].sum()), 0)
        self.assertEqual(int(target.joint_validity[0].sum()), 11)
        self.assertEqual(int(target.joint_validity[1].sum()), 0)

        logits = torch.zeros(1, 2, 49, requires_grad=True)
        loss = spatial_region_loss(logits, target.occupancy.unsqueeze(0), target.frame_mask.unsqueeze(0))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(logits.grad[0, 0].abs().sum()), 0)
        self.assertEqual(float(logits.grad[0, 1].abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
