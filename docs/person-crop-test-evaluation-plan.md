# Person crop final configuration: test evaluation plan

Date: 2026-10-14. Status: completed.

## 1. Goal and scope

Evaluate one locked final configuration once on the 20,880-sample participant-disjoint test split (participants 2, 10, 11, 16, 42, 43, 53, 57, 59, 66, 67, 68, 74, 77, 81, 83, 84, 101), which no earlier experiment has read. Seed 42 checkpoints only; nothing is trained, tuned, or selected after the test split is read.

## 2. Current issue to solve

All results so far are validation results, and the final configuration was chosen after looking at validation predictions:

| Validation (12,960) | Passes | Correct | Accuracy | A042–A048 correct / 504 |
| --- | ---: | ---: | ---: | ---: |
| A center_zoom70 | 1 | 10,680 | 82.41% | 403 |
| A whole + center_zoom70 | 2 | 10,742 | 82.89% | 404 |
| A person crop | 1 | 11,077 | 85.47% | 413 |
| B person crop | 1 | 11,069 | 85.41% | 430 |
| **A center_zoom70 + B person crop** | 2 | **11,289** | **87.11%** | **432** |

A is the crop-aware epoch 20 checkpoint (`artifacts/patch16_crop_aware/seed42/final_merged.pt`); B is the person-crop-trained epoch 20 checkpoint (`artifacts/patch16_person_crop/seed42/final_merged.pt`). The 1.64-point gain of the ensemble over A person crop is a post-hoc selection, so its size on unseen participants is unknown.

## 3. Hypothesis

The ensemble gain comes from two different models looking at two different crops of the same clip, not from fitting validation noise. On the test split, A center_zoom70 + B person crop should beat A person crop by a clear margin and keep the A042–A048 gain.

## 4. Final configuration

- **Views**: center_zoom70 and person crop exactly as in `docs/person-aware-crop-stage0-plan.md` section 4: the same uniform 8 frames, pose model, thresholds, square rule, and center_zoom70 fallback.
- **Prediction**: the mean of A's softmax on center_zoom70 and B's softmax on the person crop, then argmax. FP32 inference.
- **Reference rows** (same pass, reported only): A whole, A center_zoom70, A whole + center_zoom70, A person crop, B whole, B person crop.

## 5. Implementation

- **`person_crop_localization.py`**: accept `--subset test` and localize the 20,880 test samples from RGB frames only, with the same lock and resumable manifest format.
- **`src/vlm/training/person_crop_test_eval.py` (GPU)**: loads A and B strictly into RGB-only `XCLIPVisionClassifier`s after checking each checkpoint identity against its run lock. Before reading test labels, it reproduces the saved validation logits of both models on the first 64 Stage 0 validation samples (maximum difference at most `1e-3`, identical argmax). It then runs A on whole, center_zoom70, and person crop and B on whole and person crop for every test sample, with test labels read from the split.
- **Outputs** in `artifacts/person_crop_test/evaluation/`: `experiment_lock.json` (SHA-256 of the runner, split, crop manifest, both checkpoints, and run locks), `test_predictions.pt`, and `report.json` with accuracy, correct, macro-F1, NLL, A042–A048 correct, and per-event correct for every row, plus fixes and regressions of the final configuration against A person crop. The runner refuses to run if `report.json` already exists.

## 6. Results

Completed 2026-10-14. Validation parity passed for both models. Artifacts are in `artifacts/person_crop_test/`.

| Test (20,880) | Passes | Correct | Accuracy | Macro-F1 | NLL | A042–A048 correct / 714 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A whole + center_zoom70 | 2 | 17,688 | 84.71% | 86.00% | 0.555 | 595 |
| A person crop | 1 | 18,035 | 86.37% | 87.60% | 0.473 | 623 |
| B person crop | 1 | 18,071 | 86.55% | 87.88% | 0.465 | 641 |
| **Final: A center_zoom70 + B person crop** | 2 | **18,365** | **87.95%** | **89.20%** | **0.437** | 636 |

| Test event (correct / 102) | A042 | A043 | A044 | A045 | A046 | A047 | A048 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A whole + center_zoom70 | 92 | 101 | 83 | 74 | 93 | 70 | 82 |
| A person crop | 87 | 101 | 87 | 88 | 93 | 82 | 85 |
| B person crop | 92 | 101 | 95 | 83 | 92 | 88 | 90 |
| Final | 95 | 101 | 94 | 84 | 92 | 82 | 88 |

- The final configuration is 1.58 points above A person crop, matching the 1.64-point validation gain, so the selection did not overfit validation.
- At the same two-pass cost, it is 3.24 points above A whole + center_zoom70.
- B person crop alone has the most A042–A048 correct.
