from __future__ import annotations

import unittest

from vlm.training.binary import build_parser as build_binary_parser
from vlm.training.multiclass import (
    build_parser as build_multiclass_parser,
)
from vlm.training.prototype import (
    build_parser as build_prototype_parser,
)


class TrainingArgumentTest(unittest.TestCase):
    def test_default_artifact_directories_are_isolated(self) -> None:
        multiclass = build_multiclass_parser().parse_args([])
        prototype = build_prototype_parser().parse_args([])
        binary = build_binary_parser().parse_args([])

        self.assertEqual(
            {
                multiclass.artifacts_dir,
                prototype.artifacts_dir,
                binary.artifacts_dir,
            },
            {
                "artifacts/multiclass",
                "artifacts/prototype",
                "artifacts/binary",
            },
        )

    def test_every_pipeline_accepts_a_split_manifest(self) -> None:
        for parser in (
            build_multiclass_parser(),
            build_prototype_parser(),
            build_binary_parser(),
        ):
            arguments = parser.parse_args(
                ["--split-manifest", "artifacts/shared_split.json"]
            )
            self.assertEqual(
                arguments.split_manifest,
                "artifacts/shared_split.json",
            )


if __name__ == "__main__":
    unittest.main()
