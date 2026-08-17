from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from vlm.data import VideoSample
from vlm.models import XCLIPFeatureClassifier
from vlm.xclip_features import (
    build_xclip_cache_metadata,
    extract_xclip_feature,
    get_xclip_feature_cache_path,
    is_xclip_cache_compatible,
)


class XCLIPFeatureTest(unittest.TestCase):
    def test_cache_path_separates_same_filename_by_source_path(self) -> None:
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

        first_path = get_xclip_feature_cache_path(
            Path("cache"),
            first,
            model_name="microsoft/xclip-base-patch32",
            num_frames=8,
        )
        second_path = get_xclip_feature_cache_path(
            Path("cache"),
            second,
            model_name="microsoft/xclip-base-patch32",
            num_frames=8,
        )

        self.assertNotEqual(first_path, second_path)
        self.assertIn("whole_video_uniform", str(first_path))
        self.assertIn("get_video_features", str(first_path))

    def test_cache_requires_fixed_vector_and_matching_metadata(self) -> None:
        sample = VideoSample(
            video_path=Path("dataset/video.mp4"),
            class_id=1,
            class_index=0,
            split="train",
        )
        metadata = build_xclip_cache_metadata(
            sample,
            model_name="microsoft/xclip-base-patch32",
            num_frames=8,
        )
        compatible = {
            "feature": torch.zeros(512),
            "cache_metadata": metadata,
        }
        wrong_shape = {
            **compatible,
            "feature": torch.zeros(8, 512),
        }

        self.assertTrue(
            is_xclip_cache_compatible(
                compatible,
                sample,
                model_name="microsoft/xclip-base-patch32",
                num_frames=8,
            )
        )
        self.assertFalse(
            is_xclip_cache_compatible(
                wrong_shape,
                sample,
                model_name="microsoft/xclip-base-patch32",
                num_frames=8,
            )
        )
        self.assertFalse(
            is_xclip_cache_compatible(
                compatible,
                sample,
                model_name="microsoft/xclip-base-patch32",
                num_frames=16,
            )
        )

    def test_classifier_maps_video_feature_to_class_logits(self) -> None:
        model = XCLIPFeatureClassifier(
            feature_dim=512,
            num_classes=120,
            dropout=0.4,
        )

        logits = model(torch.zeros(3, 512))

        self.assertEqual(logits.shape, (3, 120))
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            62_584,
        )

    def test_extractor_uses_image_processor_and_pooler_output(self) -> None:
        sample = VideoSample(
            video_path=Path("dataset/video.mp4"),
            class_id=1,
            class_index=0,
            split="train",
        )

        class Processor:
            def __call__(self, *, images, return_tensors):
                self.images = images
                self.return_tensors = return_tensors
                return {
                    "pixel_values": torch.zeros(1, 8, 3, 4, 4)
                }

        class Model:
            def get_video_features(self, *, pixel_values):
                self.pixel_values = pixel_values
                return SimpleNamespace(
                    pooler_output=torch.ones(1, 512)
                )

        processor = Processor()
        model = Model()
        with patch(
            "vlm.xclip_features.read_uniform_video_frames",
            return_value=np.zeros((8, 4, 4, 3), dtype=np.uint8),
        ):
            feature = extract_xclip_feature(
                sample,
                processor=processor,
                model=model,
                device=torch.device("cpu"),
                num_frames=8,
            )

        self.assertEqual(len(processor.images), 8)
        self.assertEqual(processor.return_tensors, "pt")
        self.assertEqual(model.pixel_values.shape, (1, 8, 3, 4, 4))
        self.assertEqual(feature.shape, (512,))


if __name__ == "__main__":
    unittest.main()
