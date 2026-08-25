from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vlm.training import xclip_lora
from vlm.training import xclip_lora_tune as tuning


class XCLIPLoRATuningTest(unittest.TestCase):
    def test_optimization_stage_reuses_stage_zero_control(self) -> None:
        control = tuning._control()

        candidates = tuning.stage_candidates(
            1,
            [control],
            confirmation_seeds=[42],
        )

        self.assertEqual(len(candidates), 5)
        self.assertNotIn(control, candidates)

        records = {
            "control": {
                "stage": 0,
                "state": "completed",
                "validation_accuracy": 0.6,
                "config": control,
            },
            "candidate": {
                "stage": 1,
                "state": "completed",
                "validation_accuracy": 0.5,
                "config": candidates[0],
            },
        }
        promoted = tuning._promote(
            records,
            1,
            maximum=1,
            margin=0.0,
        )

        self.assertEqual(promoted, [control])

    def test_capacity_stage_preserves_alpha_over_rank_scaling(self) -> None:
        candidates = tuning.stage_candidates(
            2,
            [tuning._control()],
            confirmation_seeds=[42],
        )

        self.assertEqual([item["lora_rank"] for item in candidates], [4, 8, 16])
        self.assertTrue(
            all(item["lora_alpha"] / item["lora_rank"] == 2 for item in candidates)
        )

    def test_trial_ids_are_stable_and_stage_specific(self) -> None:
        config = tuning._control()

        self.assertEqual(
            tuning.stable_trial_id(1, config),
            tuning.stable_trial_id(1, dict(reversed(list(config.items())))),
        )
        self.assertNotEqual(
            tuning.stable_trial_id(1, config),
            tuning.stable_trial_id(2, config),
        )

    def test_winning_config_loads_in_regular_training_parser(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "winner.json"
            path.write_text(
                json.dumps(
                    {
                        "lora_rank": 16,
                        "lora_alpha": 32.0,
                        "warmup_epochs": 3,
                    }
                ),
                encoding="utf-8",
            )

            args = xclip_lora.parse_args(["--config", str(path)])

        self.assertEqual(args.lora_rank, 16)
        self.assertEqual(args.lora_alpha, 32.0)
        self.assertEqual(args.warmup_epochs, 3)

    def test_tuning_trials_never_evaluate_test_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            split = Path(directory) / "split.json"
            split.write_text("{}", encoding="utf-8")
            artifacts = Path(directory) / "artifacts"
            args = tuning.build_parser().parse_args(
                [
                    "--split-manifest",
                    str(split),
                    "--artifacts-dir",
                    str(artifacts),
                    "--stages",
                    "0",
                    "--epochs",
                    "1",
                ]
            )

            def fake_run(
                namespace,
                *,
                evaluate_test=True,
                cache_read_only=False,
                prepared_cached=None,
                prepared_datasets=None,
                print_results=True,
                print_run_details=True,
            ):
                if namespace.extract_only:
                    self.assertFalse(evaluate_test)
                    self.assertFalse(cache_read_only)
                    return {
                        "test_evaluated": False,
                        "_cached_features": {
                            "train": [],
                            "val": [],
                            "test": [],
                        },
                    }
                self.assertTrue(cache_read_only)
                self.assertIsNotNone(prepared_cached)
                self.assertIsNotNone(prepared_datasets)
                self.assertFalse(print_results)
                self.assertFalse(print_run_details)
                result = {
                    "validation_accuracy": 0.5,
                    "best_epoch": 1,
                    "history": [],
                    "trainable_parameter_audit": {
                        "trainable_parameters": 10,
                    },
                    "test_evaluated": evaluate_test,
                }
                if evaluate_test:
                    result["final_test_accuracy"] = 0.4
                return result

            with patch.object(tuning.xclip_lora, "run_pipeline", side_effect=fake_run) as run:
                first = tuning.run_tuning(args)
                first_call_count = run.call_count
                second = tuning.run_tuning(args)

        self.assertEqual(first["state"], "completed")
        self.assertTrue(first["test_evaluated"])
        self.assertEqual(first["final_evaluation"]["final_test_accuracy"], 0.4)
        self.assertEqual(first_call_count, 3)
        self.assertEqual(run.call_count, first_call_count)
        self.assertEqual(first["selected_winner"], second["selected_winner"])


if __name__ == "__main__":
    unittest.main()
