from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vlm.data import (
    build_split,
    load_split_manifest,
    load_split_manifest_metadata,
    parse_participant_id,
    save_split_manifest,
    select_split_classes,
)


def _video_name(
    participant_id: int,
    class_id: int,
    repetition: int = 1,
) -> str:
    return (
        f"S001C001P{participant_id:03d}R{repetition:03d}"
        f"A{class_id:03d}_rgb.mp4"
    )


class ParticipantDisjointSplitTest(unittest.TestCase):
    def test_build_split_is_deterministic_and_participant_disjoint(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            videos_dir = Path(temporary_directory)
            for participant_id in range(1, 7):
                for class_id in (42, 43):
                    (videos_dir / _video_name(participant_id, class_id)).touch()

            first = build_split(
                videos_dir,
                class_ids=[42, 43],
                videos_per_split=1,
                seed=17,
            )
            second = build_split(
                videos_dir,
                class_ids=[42, 43],
                videos_per_split=1,
                seed=17,
            )

            self.assertEqual(first, second)
            participants = {
                split_name: {
                    parse_participant_id(sample.video_path)
                    for sample in samples
                }
                for split_name, samples in first.items()
            }
            self.assertEqual(
                set(participants),
                {"train", "val", "test"},
            )
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
                {name: len(samples) for name, samples in first.items()},
                {"train": 8, "val": 2, "test": 2},
            )

    def test_default_uses_all_videos_and_optional_cap_limits_each_class(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            videos_dir = Path(temporary_directory)
            for participant_id in range(1, 7):
                for class_id in (42, 43):
                    for repetition in (1, 2):
                        (
                            videos_dir
                            / _video_name(
                                participant_id,
                                class_id,
                                repetition,
                            )
                        ).touch()

            all_videos = build_split(
                videos_dir,
                class_ids=[42, 43],
                seed=17,
            )
            capped = build_split(
                videos_dir,
                class_ids=[42, 43],
                seed=17,
                max_videos_per_class=6,
            )

            self.assertEqual(
                sum(len(samples) for samples in all_videos.values()),
                24,
            )
            self.assertEqual(
                sum(len(samples) for samples in capped.values()),
                12,
            )

    def test_manifest_records_strategy_seed_and_participants(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for participant_id in range(1, 7):
                (root / _video_name(participant_id, 42)).touch()
            split = build_split(
                root,
                class_ids=[42],
                videos_per_split=1,
                seed=5,
            )
            manifest_path = root / "artifacts" / "selected_split.json"

            save_split_manifest(
                split,
                manifest_path,
                seed=5,
                sample_fps=6.0,
                max_seq_len=32,
            )

            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["seed"], 5)
            self.assertEqual(payload["sample_fps"], 6.0)
            self.assertEqual(
                payload["frame_sampling_strategy"],
                "time_based_uniform_cap",
            )
            self.assertEqual(
                payload["split_strategy"],
                "participant_disjoint",
            )
            self.assertEqual(
                set(payload["participants"]),
                {"train", "val", "test"},
            )
            reloaded = load_split_manifest(manifest_path)
            self.assertEqual(split, reloaded)
            metadata = load_split_manifest_metadata(manifest_path)
            self.assertEqual(metadata["seed"], 5)
            self.assertEqual(
                metadata["split_strategy"],
                "participant_disjoint",
            )

    def test_loaded_manifest_can_be_filtered_and_reindexed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            videos_dir = Path(temporary_directory)
            for participant_id in range(1, 7):
                for class_id in (42, 43):
                    (
                        videos_dir
                        / _video_name(participant_id, class_id)
                    ).touch()
            split = build_split(
                videos_dir,
                class_ids=[42, 43],
                seed=9,
            )

            selected = select_split_classes(split, [43])

            self.assertTrue(
                all(
                    sample.class_id == 43
                    and sample.class_index == 0
                    for samples in selected.values()
                    for sample in samples
                )
            )

    def test_manifest_can_record_xclip_sampling_without_fps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for participant_id in range(1, 7):
                (root / _video_name(participant_id, 42)).touch()
            split = build_split(
                root,
                class_ids=[42],
                videos_per_split=1,
                seed=5,
            )
            manifest_path = root / "artifacts" / "selected_split.json"

            save_split_manifest(
                split,
                manifest_path,
                seed=5,
                sample_fps=None,
                max_seq_len=8,
                frame_sampling_strategy="whole_video_uniform",
            )

            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertIsNone(payload["sample_fps"])
            self.assertEqual(payload["max_seq_len"], 8)
            self.assertEqual(
                payload["frame_sampling_strategy"],
                "whole_video_uniform",
            )


if __name__ == "__main__":
    unittest.main()
