from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from vlm import cli


class CLITest(unittest.TestCase):
    def test_embedding_help_uses_registered_pipeline_choices(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            cli.main(["embeddings", "--help"])

        self.assertIn("xclip-lora", output.getvalue())
        self.assertNotIn("xclip-lora-tune", output.getvalue())

    def test_xclip_lora_training_dispatch(self) -> None:
        with patch("vlm.cli._invoke") as invoke:
            cli.main(["train", "xclip-lora", "--help"])

        invoke.assert_called_once_with(
            "vlm.training.xclip_lora",
            ["--help"],
            prog="vlm train xclip-lora",
        )

    def test_xclip_lora_tuning_dispatch(self) -> None:
        with patch("vlm.cli._invoke") as invoke:
            cli.main(["train", "xclip-lora-tune", "--help"])

        invoke.assert_called_once_with(
            "vlm.training.xclip_lora_tune",
            ["--help"],
            prog="vlm train xclip-lora-tune",
        )

    def test_xclip_lora_embedding_dispatch_adds_extract_only(self) -> None:
        with patch("vlm.cli._invoke") as invoke:
            cli.main(
                [
                    "embeddings",
                    "extract",
                    "--pipeline",
                    "xclip-lora",
                    "--split-manifest",
                    "shared.json",
                ]
            )

        invoke.assert_called_once_with(
            "vlm.training.xclip_lora",
            ["--split-manifest", "shared.json", "--extract-only"],
            prog="vlm embeddings extract",
        )


if __name__ == "__main__":
    unittest.main()
