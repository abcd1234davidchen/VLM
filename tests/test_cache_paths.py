from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vlm.data import VideoSample
from vlm.embeddings import get_embedding_cache_path


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
                frame_stride=5,
                max_seq_len=32,
            )
            second_path = get_embedding_cache_path(
                cache_dir,
                second,
                model_name="openai/clip-vit-base-patch32",
                frame_stride=5,
                max_seq_len=32,
            )

            self.assertNotEqual(first_path, second_path)
            self.assertIn("__path_", first_path.name)
            self.assertIn("__path_", second_path.name)


if __name__ == "__main__":
    unittest.main()
