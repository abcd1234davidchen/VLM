"""Processed RGB cache must preserve pixels and reject changed sources."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from vlm.processed_rgb_cache import ProcessedRGBCache


class _ImageProcessor:
    def to_dict(self):
        return {"resize": 224}


class _Processor:
    image_processor = _ImageProcessor()

    def __init__(self):
        self.calls = 0

    def __call__(self, *, images, return_tensors):
        self.calls += 1
        return {"pixel_values": torch.full((1, 8, 3, 224, 224), float(self.calls))}


class ProcessedRGBCacheTest(unittest.TestCase):
    def test_hit_and_source_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "clip.mp4"
            source.write_bytes(b"first")
            processor = _Processor()
            cache = ProcessedRGBCache(Path(directory) / "cache", processor, model_name="xclip", model_revision="rev")
            with patch("vlm.processed_rgb_cache.read_uniform_video_frames", return_value=np.zeros((8, 2, 2, 3), dtype=np.uint8)) as decode:
                first = cache.load(source)
                torch.testing.assert_close(cache.load(source), first, rtol=0, atol=0)
                self.assertEqual(decode.call_count, 1)
                source.write_bytes(b"changed source")
                second = cache.load(source)
                self.assertEqual(decode.call_count, 2)
                self.assertFalse(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
