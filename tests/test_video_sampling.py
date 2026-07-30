from __future__ import annotations

import unittest

import numpy as np
import torch

from vlm.data import cap_embedding_sequence, time_based_frame_indices
from vlm.embeddings import collate_cached_embeddings
from vlm.experiments.xclip import build_parser as build_xclip_parser
from vlm.experiments.xclip_vision import (
    build_parser as build_xclip_vision_parser,
)
from vlm.experiments.video_sampling import uniform_frame_indices


class VideoSamplingTest(unittest.TestCase):
    def test_six_fps_uses_real_time_and_caps_across_full_video(self) -> None:
        short_indices = time_based_frame_indices(
            total_frames=66,
            source_fps=30.0,
            target_fps=6.0,
            max_frames=32,
        )
        long_indices = time_based_frame_indices(
            total_frames=248,
            source_fps=30.0,
            target_fps=6.0,
            max_frames=32,
        )

        self.assertEqual(short_indices, list(range(0, 66, 5)))
        self.assertEqual(len(long_indices), 32)
        self.assertEqual(long_indices[0], 0)
        self.assertEqual(long_indices[-1], 245)
        self.assertEqual(long_indices, sorted(set(long_indices)))

    def test_short_embedding_sequences_are_not_pre_padded(self) -> None:
        sequence = torch.randn(11, 8)

        resized, valid_length = cap_embedding_sequence(sequence, 32)

        self.assertEqual(resized.shape, (11, 8))
        self.assertEqual(valid_length, 11)

    def test_collate_pads_only_to_batch_maximum(self) -> None:
        batch = [
            (
                torch.ones(8, 4),
                torch.tensor(8),
                torch.tensor(0),
            ),
            (
                torch.ones(11, 4),
                torch.tensor(11),
                torch.tensor(1),
            ),
        ]

        embeddings, lengths, labels = collate_cached_embeddings(batch)

        self.assertEqual(embeddings.shape, (2, 11, 4))
        self.assertTrue(torch.equal(lengths, torch.tensor([8, 11])))
        self.assertTrue(torch.equal(labels, torch.tensor([0, 1])))
        self.assertEqual(embeddings[0, 8:].count_nonzero().item(), 0)

    def test_xclip_uniform_sampling_is_deterministic_and_fixed_size(
        self,
    ) -> None:
        full_video = uniform_frame_indices(8, 100)
        short_video = uniform_frame_indices(8, 3)

        np.testing.assert_array_equal(
            full_video,
            np.array([0, 14, 28, 42, 57, 71, 85, 99]),
        )
        self.assertEqual(len(short_video), 8)
        self.assertEqual(short_video[0], 0)
        self.assertEqual(short_video[-1], 2)
        self.assertEqual(build_xclip_parser().parse_args([]).frames, 8)
        self.assertEqual(
            build_xclip_vision_parser().parse_args([]).frames,
            8,
        )


if __name__ == "__main__":
    unittest.main()
