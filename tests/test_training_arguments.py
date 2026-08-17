from __future__ import annotations

import unittest

from vlm.training.binary import build_parser as build_binary_parser
from vlm.training.multiclass import (
    build_parser as build_multiclass_parser,
)
from vlm.training.prototype import (
    build_parser as build_prototype_parser,
)
from vlm.training.xclip import build_parser as build_xclip_parser


class TrainingArgumentTest(unittest.TestCase):
    def test_default_artifact_directories_are_isolated(self) -> None:
        multiclass = build_multiclass_parser().parse_args([])
        prototype = build_prototype_parser().parse_args([])
        binary = build_binary_parser().parse_args([])
        xclip = build_xclip_parser().parse_args([])

        self.assertEqual(
            {
                multiclass.artifacts_dir,
                prototype.artifacts_dir,
                binary.artifacts_dir,
                xclip.artifacts_dir,
            },
            {
                "artifacts/multiclass",
                "artifacts/prototype",
                "artifacts/binary",
                "artifacts/xclip",
            },
        )

    def test_every_pipeline_accepts_a_split_manifest(self) -> None:
        for parser in (
            build_multiclass_parser(),
            build_prototype_parser(),
            build_binary_parser(),
            build_xclip_parser(),
        ):
            arguments = parser.parse_args(
                ["--split-manifest", "artifacts/shared_split.json"]
            )
            self.assertEqual(
                arguments.split_manifest,
                "artifacts/shared_split.json",
            )

    def test_every_pipeline_defaults_to_six_fps_and_32_frames(self) -> None:
        for parser in (
            build_multiclass_parser(),
            build_prototype_parser(),
            build_binary_parser(),
        ):
            arguments = parser.parse_args([])
            self.assertEqual(arguments.sample_fps, 6.0)
            self.assertEqual(arguments.max_seq_len, 32)

    def test_xclip_defaults_to_native_uniform_eight_frames(self) -> None:
        arguments = build_xclip_parser().parse_args([])
        self.assertEqual(
            arguments.model_name,
            "microsoft/xclip-base-patch32",
        )
        self.assertEqual(arguments.frames, 8)
        self.assertEqual(arguments.feature_batch_size, 1)
        self.assertEqual(arguments.decode_workers, 1)


if __name__ == "__main__":
    unittest.main()
