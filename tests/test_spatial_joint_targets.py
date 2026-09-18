"""Guard the Phase 4 supervision mask and patch geometry contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from vlm.spatial_joint_targets import build_spatial_targets, spatial_region_loss


class SpatialJointTargetsTest(unittest.TestCase):
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
