# Person-aware crop Stage 1: training plan

Date: 2026-10-12. Status: completed; decision no-go (section 7).

## 1. Goal and scope

Train `microsoft/xclip-base-patch16` with the patch16 crop-aware recipe, replacing the center_zoom70 training view with the Stage 0 person crop, and compare the epoch 20 checkpoint with the crop-aware checkpoint evaluated on person crops. Seed 42, canonical train and validation splits.

## 2. Current issue to solve

Stage 0 (`docs/person-aware-crop-stage0-plan.md`) showed that the crop-aware checkpoint does better on person crops than on the center crop it was trained with, but the model never saw person crops in training:

| Crop-aware epoch 20, evaluated on | Correct | Accuracy | A042–A048 correct / 504 |
| --- | ---: | ---: | ---: |
| Center zoom70 (training view) | 10,680 | 82.41% | 403 |
| Person crop (unseen view) | 11,077 | 85.47% | 413 |
| Whole + person | 10,995 | 84.84% | 413 |

Event-level gains are uneven: A047 rose from 32 to 48 but A042 fell from 57 to 52. The model is also tuned to the fixed 70% center scale, while person crops range from 50% to 100% of the short edge (median 66%).

## 3. Hypothesis

Training on person crops lets the model learn the variable scale and position of person-centered input. The single-pass person crop accuracy should rise above the 85.47% the crop-aware checkpoint reaches without seeing person crops, and the A042 drop should shrink once staggering seen through a tight crop is part of training.

## 4. Training view definition

- **Person crop**: exactly the Stage 0 definition (plan section 4) and the same localization runner, constants, and pose model. Validation reuses the Stage 0 manifest; train gets a new manifest from the same runner. Samples that fall back use the center_zoom70 box.
- **View mix**: each training sample sees whole or person_crop each epoch, chosen by the same `sha256("patch16-crop-aware:{seed}:{epoch}:{sample_id}")` bit as crop-aware. Every sample therefore gets the crop view in exactly the epochs it got center_zoom70 in the crop-aware run, so the only difference from that run is the crop box.
- **Recipe**: identical to crop-aware, read from the patch16 pilot `run_lock.json`: 60-epoch cosine schedule stopped after epoch 20, batch 16, AMP, upper-2 vision plus MIT q/v LoRA, fresh initialization, and the same shuffle order.

## 5. Implementation

- **`person_crop_localization.py`**: add a `--subset {train,val}` option (default `val`, so the Stage 0 lock is unchanged) and localize the 80,640 train samples.
- **`src/vlm/training/patch16_person_crop.py`**: a runner built on `patch16_crop_aware.py` that reuses its sampler, view hash, model, optimizer, checkpointing, and resume logic. It adds a `PersonCropUInt8Cache` that reads each sample's box from the train or validation manifest, keyed by the manifest SHA-256 so it never collides with the whole or center_zoom70 caches, and fills on miss.
- **Validation**: every epoch on the full 12,960 samples in FP32, reporting whole, person_crop, and whole + person: accuracy, correct, macro-F1, NLL, A042–A048 correct, and per-event correct.
- **Outputs**: `run_lock.json`, `train.log`, `progress.json`, resumable `last_checkpoint.pt`, and at epoch 20 `final_adapter.pt`, `final_merged.pt` (with merge and RGB-only reload parity checks), `final_view_predictions.pt`, and `run_report.json`. The report includes per-view fixes and regressions against the Stage 0 predictions of the crop-aware checkpoint.

## 6. Pre-registered decision rules

All judgments use epoch 20 with no epoch selection. The baseline is the crop-aware checkpoint on person crops (Stage 0): 11,077 correct and 413 A042–A048 correct. 0.20 points is about 26 samples.

- **Early stop (automatic)**: person_crop accuracy at epoch 5 below 78.13% (the crop-aware center_zoom70 accuracy at epoch 5, 79.13%, minus 1.0 point) stops the run as no-go.
- **Promising**: person_crop single-pass correct is at least 11,103, and A042–A048 person_crop correct is at least 413.
- **No-go**: person_crop single-pass correct is below 11,077.
- **Inconclusive**: anything else.
- **Also reported (not gates)**: A042 and A047 per event, whole + person, whole single pass, and the epoch curve against crop-aware center_zoom70.

## 7. Results

Completed 2026-10-13: 20 epochs, no early stop (epoch 5 person_crop 81.65%), test unread. Initial logits matched the pilot, merge and RGB-only reload parity passed (max difference `5.4e-6`), and 193,656 trainable parameters. Artifacts are in `artifacts/patch16_person_crop/seed42/`.

| Epoch 20 view | Crop-aware (Stage 0) correct | Person-trained correct | Person-trained accuracy | Macro-F1 | NLL | A042–A048 correct / 504 | Fixes / regressions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Whole | 9,763 | 9,653 | 74.48% | 75.90% | 0.898 | 395 (was 380) | 645 / 755 |
| Person crop | 11,077 | 11,069 | 85.41% | 86.29% | 0.506 | 430 (was 413) | 482 / 490 |
| Whole + person | 10,995 | 10,984 | 84.75% | 85.63% | 0.572 | 428 (was 413) | 488 / 499 |

| Person crop event (correct / 72) | A042 | A043 | A044 | A045 | A046 | A047 | A048 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Crop-aware (Stage 0) | 52 | 70 | 71 | 52 | 63 | 48 | 57 |
| Person-trained | 60 | 71 | 68 | 58 | 64 | 50 | 59 |

Person crop accuracy by epoch: 81.65% at 5, 84.25% at 10, 84.89% at 15, 85.52% at 19, 85.41% at 20. It rose 3.8 points from epoch 5 to 20, against 3.3 points for crop-aware center_zoom70.

### Decision

**No-go** under the pre-registered rules: person_crop single-pass correct is 11,069, 8 below the 11,077 baseline.

- Overall accuracy is a tie (482 fixes, 490 regressions, −0.06 points). Training on person crops adds nothing overall beyond using person crops at inference with the crop-aware checkpoint.
- A042–A048 person_crop correct rose from 413 to 430, and A042 recovered from 52 to 60, as the hypothesis predicted. This is an event-level gain, not a pre-registered gate, and comes with a slightly lower whole view (−110 correct).
