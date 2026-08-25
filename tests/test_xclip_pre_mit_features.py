from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from vlm.data import VideoSample
from vlm.xclip_features import get_xclip_feature_cache_path
from vlm.xclip_pre_mit_features import (
    build_xclip_pre_mit_cache_metadata,
    extract_or_load_xclip_pre_mit_features,
    extract_xclip_pre_mit_feature_batch,
    get_xclip_pre_mit_feature_cache_path,
    is_xclip_pre_mit_cache_compatible,
)


class XCLIPPreMITFeatureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = VideoSample(
            video_path=Path("dataset/video.mp4"),
            class_id=1,
            class_index=0,
            split="train",
        )

    def test_cache_path_is_isolated_from_post_mit_cache(self) -> None:
        pre_mit = get_xclip_pre_mit_feature_cache_path(
            Path("cache"),
            self.sample,
            model_name="microsoft/xclip-base-patch32",
            num_frames=8,
        )
        post_mit = get_xclip_feature_cache_path(
            Path("cache"),
            self.sample,
            model_name="microsoft/xclip-base-patch32",
            num_frames=8,
        )

        self.assertNotEqual(pre_mit, post_mit)
        self.assertIn("xclip_pre_mit_features", str(pre_mit))
        self.assertIn("pre_mit_projected_frames", str(pre_mit))

    def test_cache_requires_projected_frame_matrix(self) -> None:
        metadata = build_xclip_pre_mit_cache_metadata(
            self.sample,
            model_name="microsoft/xclip-base-patch32",
            num_frames=8,
            feature_dim=512,
            dtype="float16",
        )
        compatible = {
            "feature": torch.zeros(8, 512, dtype=torch.float16),
            "cache_metadata": metadata,
        }
        post_mit_vector = {
            **compatible,
            "feature": torch.zeros(512, dtype=torch.float16),
        }

        self.assertTrue(
            is_xclip_pre_mit_cache_compatible(
                compatible,
                self.sample,
                model_name="microsoft/xclip-base-patch32",
                num_frames=8,
                feature_dim=512,
                dtype="float16",
            )
        )
        self.assertFalse(
            is_xclip_pre_mit_cache_compatible(
                post_mit_vector,
                self.sample,
                model_name="microsoft/xclip-base-patch32",
                num_frames=8,
                feature_dim=512,
                dtype="float16",
            )
        )

    def test_extractor_stops_before_mit(self) -> None:
        class Processor:
            def __call__(self, *, images, return_tensors):
                return {
                    "pixel_values": torch.zeros(2, 8, 3, 4, 4)
                }

        class Projection:
            def __call__(self, features):
                return features + 1

        class Model:
            visual_projection = Projection()

            def vision_model(self, *, pixel_values, return_dict):
                self.pixel_values = pixel_values
                self.return_dict = return_dict
                return SimpleNamespace(
                    pooler_output=torch.zeros(16, 512)
                )

        frames = np.zeros((8, 4, 4, 3), dtype=np.uint8)
        model = Model()

        features = extract_xclip_pre_mit_feature_batch(
            [frames, frames],
            processor=Processor(),
            model=model,
            device=torch.device("cpu"),
        )

        self.assertEqual(model.pixel_values.shape, (16, 3, 4, 4))
        self.assertTrue(model.return_dict)
        self.assertEqual([item.shape for item in features], [(8, 512)] * 2)
        self.assertTrue(torch.equal(features[0], torch.ones(8, 512)))

    def test_read_only_cache_rejects_missing_entries(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "read-only"):
                extract_or_load_xclip_pre_mit_features(
                    {"train": [self.sample]},
                    cache_dir=Path(directory),
                    model_name="microsoft/xclip-base-patch32",
                    num_frames=8,
                    device=torch.device("cpu"),
                    force_extract=False,
                    read_only=True,
                )


if __name__ == "__main__":
    unittest.main()
