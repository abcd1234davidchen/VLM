from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from vlm.data.skeleton import (
    SkeletonFormatError,
    audit_skeleton_data,
    load_skeleton_tensor,
    parse_skeleton_file,
    parse_skeleton_identity,
    safe_extract_skeleton_archive,
)


def _write_skeleton(path: Path, *, joints: int = 25, value: str = "0") -> None:
    lines = ["1", "1", "72057594037931777 0 0 0 0 0 0 0 0 2", str(joints)]
    lines.extend(f"{value} 0 0 0 0 0 0 1 0 0 0 2" for _ in range(joints))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class SkeletonDataTest(unittest.TestCase):
    def test_identity_requires_complete_name(self) -> None:
        identity = parse_skeleton_identity(
            "S017C003P040R002A060_rgb.avi"
        )
        self.assertEqual(identity.sample_id, "S017C003P040R002A060")
        with self.assertRaises(ValueError):
            parse_skeleton_identity("P040A060.skeleton")

    def test_parser_validates_joint_count_and_non_finite_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            valid = root / "S001C001P001R001A001.skeleton"
            _write_skeleton(valid)
            metadata = parse_skeleton_file(valid)
            self.assertEqual(metadata.frame_count, 1)
            self.assertEqual(metadata.joint_observations, 25)

            wrong_count = root / "S001C001P001R001A002.skeleton"
            _write_skeleton(wrong_count, joints=24)
            with self.assertRaisesRegex(SkeletonFormatError, "expected 25"):
                parse_skeleton_file(wrong_count)

            non_finite = root / "S001C001P001R001A003.skeleton"
            _write_skeleton(non_finite, value="nan")
            with self.assertRaisesRegex(SkeletonFormatError, "non-finite"):
                parse_skeleton_file(non_finite)

            tensor = load_skeleton_tensor(valid, frames=4)
            self.assertEqual(tuple(tensor.shape), (3, 4, 25, 2))
            self.assertTrue(tensor.isfinite().all())

    def test_safe_extract_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive_path = root / "unsafe.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../S001C001P001R001A001.skeleton", "0\n")
            with self.assertRaisesRegex(ValueError, "Unsafe archive member"):
                safe_extract_skeleton_archive(archive_path, root / "output")

    def test_audit_pairs_train_and_val_without_reading_test_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            skeleton_dir = root / "skeletons"
            skeleton_dir.mkdir()
            train_id = "S001C001P001R001A001"
            val_id = "S001C001P002R001A001"
            _write_skeleton(skeleton_dir / f"{train_id}.skeleton")
            _write_skeleton(skeleton_dir / f"{val_id}.skeleton")
            rgb_manifest = root / "rgb.json"
            rgb_manifest.write_text(
                json.dumps(
                    {
                        "train": [{"video_path": f"{train_id}_rgb.avi", "class_id": 1}],
                        "val": [{"video_path": f"{val_id}_rgb.avi", "class_id": 1}],
                        "test": [{"intentionally": "not a video record"}],
                    }
                ),
                encoding="utf-8",
            )
            missing = root / "missing.txt"
            missing.write_text(
                "Official list\nS017C003P040R002A060\n", encoding="utf-8"
            )
            artifacts = root / "artifacts"

            coverage = audit_skeleton_data(
                skeleton_dir=skeleton_dir,
                rgb_manifest_path=rgb_manifest,
                missing_skeletons_path=missing,
                artifacts_dir=artifacts,
                expected_file_count=2,
            )

            self.assertEqual(coverage["splits"]["train"]["paired_samples"], 1)
            self.assertEqual(coverage["splits"]["val"]["paired_samples"], 1)
            self.assertEqual(
                coverage["splits"]["train"]["paired_fraction_of_eligible"],
                1.0,
            )
            paired = json.loads((artifacts / "paired_manifest.json").read_text())
            self.assertNotIn("test", paired)
            self.assertEqual(len(paired["ordered_class_ids"]), 59)
            self.assertEqual(paired["excluded_action_ids"], ["A060"])


if __name__ == "__main__":
    unittest.main()
