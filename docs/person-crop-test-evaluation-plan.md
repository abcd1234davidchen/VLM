# Person crop final configuration: test evaluation plan

Date: 2026-10-14. Status: plan only; not yet implemented or run.

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
