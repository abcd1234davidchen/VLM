from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from vlm.data import VideoSample
from vlm.embeddings import (
    build_cache_metadata,
    get_embedding_cache_path,
    is_cache_compatible,
)


class CachePathTest(unittest.TestCase):
    def test_same_filename_in_different_directories_does_not_collide(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_dir = Path(temporary_directory) / "cache"
            filename = "S001C001P001R001A001_rgb.mp4"
            first = VideoSample(
                video_path=Path("dataset/first") / filename,
                class_id=1,
                class_index=0,
                split="train",
            )
            second = VideoSample(
                video_path=Path("dataset/second") / filename,
                class_id=1,
                class_index=0,
                split="train",
            )

            first_path = get_embedding_cache_path(
                cache_dir,
                first,
                model_name="openai/clip-vit-base-patch32",
                sample_fps=6.0,
                max_seq_len=32,
            )
            second_path = get_embedding_cache_path(
                cache_dir,
                second,
                model_name="openai/clip-vit-base-patch32",
                sample_fps=6.0,
                max_seq_len=32,
            )

            self.assertNotEqual(first_path, second_path)
            self.assertIn("__path_", first_path.name)
            self.assertIn("__path_", second_path.name)

    def test_variable_length_cache_requires_exact_valid_length(self) -> None:
        sample = VideoSample(
            video_path=Path("dataset/S001C001P001R001A001_rgb.mp4"),
            class_id=1,
            class_index=0,
            split="train",
        )
        metadata = build_cache_metadata(
            sample,
            model_name="openai/clip-vit-base-patch32",
            sample_fps=6.0,
            max_seq_len=32,
        )
        compatible = {
            "embeddings": torch.zeros(11, 512),
            "valid_length": 11,
            "cache_metadata": metadata,
        }
        fixed_length = {
            **compatible,
            "embeddings": torch.zeros(32, 512),
        }

        self.assertTrue(
            is_cache_compatible(
                compatible,
                sample,
                model_name="openai/clip-vit-base-patch32",
                sample_fps=6.0,
                max_seq_len=32,
            )
        )
        self.assertFalse(
            is_cache_compatible(
                fixed_length,
                sample,
                model_name="openai/clip-vit-base-patch32",
                sample_fps=6.0,
                max_seq_len=32,
            )
        )


if __name__ == "__main__":
    unittest.main()
