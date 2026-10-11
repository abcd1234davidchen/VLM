# Person-aware crop Stage 0: inference diagnostic plan

Date: 2026-10-11. Status: plan only; not yet implemented or run.

## 1. Goal and scope

Evaluate the crop-aware epoch 20 checkpoint (`artifacts/patch16_crop_aware/seed42/final_merged.pt`) on the full 12,960-sample canonical validation split with center_zoom70 replaced by a crop placed from RGB human pose. The result decides whether a Stage 1 person-aware training run is worth doing.

## 2. Current issue to solve

The center 70% crop has a fixed position and scale. People who are off-center or lying on the floor after a fall can be cut off, and people far from the camera are not magnified enough. Crop-aware training raised the overall accuracy, yet A043 (falling) did not improve and A048 (nausea/vomiting) regressed in the zoom view:

| View | Correct | Accuracy | A042–A048 correct / 504 |
| --- | ---: | ---: | ---: |
| Whole (single pass) | 9,763 | 75.33% | 380 |
| Center zoom70 (single pass) | 10,680 | 82.41% | 403 |
| Whole + zoom probability mean (two passes) | 10,742 | 82.89% | 404 |

| Event (correct / 72) | A042 | A043 | A044 | A045 | A046 | A047 | A048 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Center zoom70 | 57 | 69 | 71 | 53 | 63 | 32 | 58 |
| Whole + zoom | 60 | 69 | 70 | 50 | 62 | 33 | 60 |

## 3. Hypothesis

A square crop that covers every detected person over the whole clip keeps the full body, including the post-fall position, while still magnifying the person. Replacing center_zoom70 with it in the two-pass ensemble should match or beat whole + zoom overall and recover A042–A048 errors caused by cropping. Because the checkpoint never saw person crops in training, a result that matches center crops here suggests training on them can do better.

## 4. Person crop definition

- **Landmarks**: MediaPipe PoseLandmarker lite on the same uniform 8 frames the classifier sees, `num_poses=2` (two-person actions are 22.8% of validation), detection and presence confidence 0.5. In each frame, keep every landmark from every detected person whose visibility and presence are both at least 0.5 and whose coordinates are inside the frame.
- **Valid frame**: at least 8 usable landmarks in that frame, counted across all detected people.
- **Box**: the bounding box of the union of usable landmarks over all valid frames. All 8 frames share one fixed box.
- **Square**: side `round(1.4 × max(box width, box height))`, clamped to `[round(0.5 × short edge), short edge]`.
- **Position**: centered on the box center, shifted to lie fully inside the frame, top-left corner floored.
- **Fallback**: with fewer than 4 valid frames, use the center_zoom70 box and record the reason. Every sample stays in the evaluation.
- The 8 cropped frames go through the same processor as whole and zoom.

## 5. Implementation

- **`src/vlm/training/person_crop_localization.py` (CPU)**: localizes all validation samples from RGB frames only. Writes `experiment_lock.json` (SHA-256 of the split, pose model, runner, and crop definition; MediaPipe version and thresholds), `crop_manifest.json` (per sample: frame indices, per-frame landmarks, box, square crop, fallback reason, source video SHA-256), and `progress.json`, and resumes from `partial_crop_manifest.json`.
- **`src/vlm/training/person_crop_audit.py` (GPU)**: loads `final_merged.pt` into an RGB-only `XCLIPVisionClassifier` (strict load) and runs FP32 inference on the person crop view. Whole and center_zoom70 logits come from `final_view_predictions.pt`. Writes `view_predictions.pt`, `report.json`, and `case_changes.json` (per-sample fixes and regressions relative to center_zoom70 and whole + zoom).
- **Report**: accuracy, correct, macro-F1, NLL, A042–A048 correct, and per-event correct for whole, center_zoom70, person_crop, whole + zoom, whole + person, and whole + zoom + person.
- **Geometry statistics**: fallback count by class; IoU between the person square and the center_zoom70 box; the fraction of samples whose person box extends beyond the center_zoom70 box, overall and for A043, A048, and two-person classes; square side relative to the short edge.
- **Input review**: for samples pre-selected by sample-ID hash (3 each from A043, A048, A045, and two-person classes, plus the first 6 fallbacks), save the source frame with both boxes drawn and a strip of the 8 frames fed to the model.

## 6. Pre-registered decision rules

The primary comparison is whole + person against whole + zoom (10,742 correct), at the same two-pass cost. 0.20 points is about 26 samples.

- **Go (write a Stage 1 training plan)**: whole + person correct is at least 10,742, and A042–A048 whole + person correct is at least 404.
- **No-go**: whole + person correct is below 10,716.
- **Inconclusive**: anything else. Review case changes and input strips before deciding.
