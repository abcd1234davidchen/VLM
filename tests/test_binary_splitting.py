from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vlm.data import parse_participant_id
from vlm.training.binary import build_binary_split


def _video_name(participant_id: int, class_id: int) -> str:
    return (
        f"S001C001P{participant_id:03d}R001"
        f"A{class_id:03d}_rgb.mp4"
    )


class BinaryParticipantSplitTest(unittest.TestCase):
    def test_binary_split_keeps_participants_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            videos_dir = Path(temporary_directory)
            for participant_id in range(1, 7):
                for class_id in (1, 42):
                    (videos_dir / _video_name(participant_id, class_id)).touch()

            split = build_binary_split(
                videos_dir,
                test_fraction=1 / 6,
                val_fraction=1 / 6,
                seed=23,
                max_videos_per_category=6,
            )

            participants = {
                split_name: {
                    parse_participant_id(sample.video_path)
                    for sample in samples
                }
                for split_name, samples in split.items()
            }
            self.assertTrue(
                participants["train"].isdisjoint(participants["val"])
            )
            self.assertTrue(
                participants["train"].isdisjoint(participants["test"])
            )
            self.assertTrue(
                participants["val"].isdisjoint(participants["test"])
            )
            self.assertEqual(
                {name: len(samples) for name, samples in split.items()},
                {"train": 8, "val": 2, "test": 2},
            )


if __name__ == "__main__":
    unittest.main()
