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
from vlm.training.xclip_lora import (
    build_parser as build_xclip_lora_parser,
)


class TrainingArgumentTest(unittest.TestCase):
    def test_default_artifact_directories_are_isolated(self) -> None:
        multiclass = build_multiclass_parser().parse_args([])
        prototype = build_prototype_parser().parse_args([])
        binary = build_binary_parser().parse_args([])
        xclip = build_xclip_parser().parse_args([])
        xclip_lora = build_xclip_lora_parser().parse_args([])

        self.assertEqual(
            {
                multiclass.artifacts_dir,
                prototype.artifacts_dir,
                binary.artifacts_dir,
                xclip.artifacts_dir,
                xclip_lora.artifacts_dir,
            },
            {
                "artifacts/multiclass",
                "artifacts/prototype",
                "artifacts/binary",
                "artifacts/xclip",
                "artifacts/xclip_lora_ce",
            },
        )

    def test_every_pipeline_accepts_a_split_manifest(self) -> None:
        for parser in (
            build_multiclass_parser(),
            build_prototype_parser(),
            build_binary_parser(),
            build_xclip_parser(),
            build_xclip_lora_parser(),
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

        lora_arguments = build_xclip_lora_parser().parse_args([])
        self.assertEqual(lora_arguments.frames, 8)
        self.assertEqual(lora_arguments.feature_batch_size, 1)
        self.assertEqual(lora_arguments.decode_workers, 1)
        self.assertEqual(lora_arguments.epochs, 60)
        self.assertEqual(lora_arguments.lora_rank, 8)
        self.assertEqual(lora_arguments.lora_alpha, 16.0)
        self.assertEqual(lora_arguments.lora_dropout, 0.05)
        self.assertEqual(lora_arguments.lora_learning_rate, 1e-4)
        self.assertEqual(lora_arguments.head_learning_rate, 8e-4)
        self.assertEqual(lora_arguments.gradient_accumulation_steps, 1)
        self.assertTrue(lora_arguments.amp)
        self.assertIsNone(lora_arguments.action_taxonomy)
        reduced = build_xclip_lora_parser().parse_args(
            ["--action-taxonomy", "reduced-v1"]
        )
        self.assertEqual(reduced.action_taxonomy, "reduced-v1")


if __name__ == "__main__":
    unittest.main()
