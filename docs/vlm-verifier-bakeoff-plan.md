# VLM verifier bake-off plan

Date: 2026-10-14. Status: completed 2026-10-15; rules were locked before any model run.

## 1. Goal and scope

Choose the local vision-language model that double-checks stage-1 alerts on the Mac that will run the demo (Apple M4 Pro, 24 GB), and decide whether the verifier is allowed to lower alert priority at all. Stage 1 is the locked final configuration from `docs/person-crop-test-evaluation-plan.md`: the mean of checkpoint A's softmax on center_zoom70 and checkpoint B's softmax on the person crop. Validation only; the test split is not read.

Candidates, all 4-bit MLX builds that leave memory for stage 1:

| Candidate | Family | Size |
| --- | --- | --- |
| `mlx-community/Qwen3.5-9B-MLX-4bit` | Qwen | 9B |
| `mlx-community/Qwen3.5-4B-MLX-4bit` | Qwen | 4B |
| `mlx-community/gemma-4-12B-it-qat-4bit` | Gemma | 12B |
| `mlx-community/gemma-4-e4b-it-4bit` | Gemma | E4B |

## 2. Current issue to solve

The verifier has its own error rate. As a hard gate it multiplies recall (stage-1 recall times verifier recall), and because it sees the same frames as stage 1, its errors on look-alike actions may be correlated with stage-1 errors. An earlier draft scored the verifier on clips picked by class, which does not match what the verifier receives in deployment: only the clips stage 1 has already flagged.

The attention-group score (sum of A042–A048 probabilities) of the final configuration, thresholded at 95% recall on the shared calibration half, flags 352 of 6,480 evaluation-half clips: 246 true alerts out of 252 positives (97.6% recall) and 106 false alerts. The largest false-alert sources are A004 brushing hair (21), A027 jump up (10), A037 wipe face (10), A026 hopping (8), and A041 sneeze/cough (8). Falling has no false alerts at its own threshold, while nausea/vomiting is mostly confused with A045 chest pain and A035 nod head/bow.

## 3. Hypothesis

A general VLM with scene context can recognise ordinary grooming, jumping, and face-touching that stage 1 confuses with distress poses, so it can dismiss a meaningful share of false alerts while almost never dismissing a true alert. A smaller model may match a larger one on this coarse decision at lower latency.

## 4. Verifier definition

- **Input**: 8 uniformly sampled, uncropped RGB frames of the flagged clip, long side resized to at most 768 pixels, in time order.
- **Prompt**: one multiple-choice question with seven alert options (fall, vomiting, staggering, and holding head, chest or stomach, back, or neck in pain), five normal-activity options (handling objects or shoes; sitting, standing, squatting, jumping, or hopping; grooming or touching the face or hair; exercising or stretching; other normal activity), and an explicit "cannot tell" option. The answer is JSON with the choice, whether the person is on the floor at the end, a confidence, and one sentence of evidence. Greedy decoding, thinking disabled, at most 160 new tokens.
- **Tier**: an alert option is `confirmed`, a normal-activity option is `dismissed`, and "cannot tell" or an unparseable answer is `uncertain`. Only `dismissed` may lower an alert's priority; `confirmed` and `uncertain` keep the alert at high priority. No answer ever deletes an alert.
- **Evaluation set**: from the 352 flagged evaluation-half clips, every one of the 106 false alerts plus 8 true alerts per event (56) chosen by a fixed sample-ID hash, 162 clips in total.

## 5. Implementation

- **`src/vlm/training/alert_false_positive_audit.py`**: loads the A and B validation predictions, checks that sample IDs and labels align, audits false-alert sources for the final configuration and both single person-crop views, and exports the flagged evaluation-half clips.
- **`src/vlm/experiments/vlm_verifier.py`**: `select` samples the evaluation set from that export; `run` asks one candidate to verify every clip in an isolated mlx-vlm environment, appends one resumable JSON row per clip, and writes a report with the tier rates below, per-event dismissals and exact-event matches, false-alert tiers by source action, median and 90th-percentile latency, and peak memory.
- **Outputs**: `artifacts/vlm_verifier/` holds the flagged export, the manifest, and one directory per candidate with `rows.jsonl` and `report.json`.

## 6. Pre-registered decision rules

Rates are measured on the 162-clip evaluation set. The retained rate is the share of true alerts not dismissed; the dismissed rate is the share of false alerts dismissed.

- **Eligibility**: valid JSON for at least 95% of clips, median latency at most 15 seconds per clip, and peak memory at most 12 GB.
- **Recall guard**: a candidate must retain at least 95% of true alerts, that is, dismiss at most 2 of the 56.
- **Winner**: among candidates passing both, the highest false-alert dismissed rate. Candidates within 3 clips of the best are tied; ties go to the higher true-alert confirmed rate, then the lower median latency.
- **Verifier role**: if the winner dismisses at least 30% of false alerts (32 of 106), it may lower alert priority in the pipeline. Otherwise, or if no candidate passes, the verifier only writes the alert description and never changes priority.
- **Also reported (not gates)**: per-event dismissals, uncertain rate, the false-alert sources each candidate dismisses, and the cascade estimate for the full evaluation half (stage-1 recall times the retained rate, and the expected number of high-priority false alerts).

## 7. Results

Completed 2026-10-15 on all 162 clips for every candidate with `mlx-vlm` 0.7.8; test unread. Every candidate returned valid JSON for every clip and never chose "cannot tell", so no answer was `uncertain`. Artifacts are in `artifacts/vlm_verifier/`.

| Candidate | Median / p90 latency (s) | Peak memory (GB) | Eligible | True alerts dismissed / 56 | True alerts confirmed / 56 | False alerts dismissed / 106 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| Qwen3.5-9B | 12.9 / 13.3 | 7.7 | yes | 14 | 42 | 73 |
| Qwen3.5-4B | 7.4 / 7.5 | 4.9 | yes | 18 | 38 | 83 |
| Gemma 4 12B | 16.7 / 17.0 | 12.8 | no (latency, memory) | 56 | 0 | 106 |
| Gemma 4 E4B | 6.1 / 6.2 | 8.4 | yes | 45 | 11 | 81 |

| True alerts dismissed (of 8 per event) | A042 | A043 | A044 | A045 | A046 | A047 | A048 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-9B | 2 | 6 | 1 | 0 | 2 | 3 | 0 |
| Qwen3.5-4B | 3 | 6 | 3 | 0 | 4 | 2 | 0 |
| Gemma 4 12B | 8 | 8 | 8 | 8 | 8 | 8 | 8 |
| Gemma 4 E4B | 8 | 8 | 3 | 7 | 8 | 6 | 5 |

Used as a hard gate on the full evaluation half, the best recall (Qwen3.5-9B) would cut stage-1 attention-group recall from 97.6% (246 / 252) to 73.2%, while high-priority false alerts would fall from 106 to 33.

### Decision

**No winner; the verifier role is description only.** Every candidate fails the recall guard (at most 2 of 56 true alerts dismissed), so under the pre-registered rules no verifier may lower alert priority. It may still write the alert description sent to relatives.

- **Falls are dismissed because NTU falls are acted as controlled descents.** The Qwen models describe six of eight falls as "standing, then squatting, then sitting on the floor", which is what the acted clips show, and they answer `person_on_floor_at_end: true` for all of them. The verifier reads the motion correctly; the acted data does not look like an uncontrolled fall.
- **The remaining dismissals repeat the stage-1 confusion.** Headache, back pain, and neck pain poses are described as grooming or touching the face or hair, the same look-alikes that cause stage-1 false alerts, so verifier and stage-1 errors are correlated on exactly the hardest events.
- **The Gemma models are biased towards normal activity.** They describe the frames sensibly but Gemma 4 12B never chose an alert option and neither Gemma model ever reported the person on the floor at the end.
- **The Qwen models do reject the main look-alike.** Both dismiss about 17 to 18 of 21 A004 brushing-hair false alerts, the largest false-alert source.

### Post-hoc analysis (not pre-registered)

Keeping a dismissal as `uncertain` when the same answer reports the person on the floor at the end changes only the true-alert side:

| Candidate | True alerts dismissed / 56 | Falls dismissed / 8 | False alerts dismissed / 106 |
| --- | ---: | ---: | ---: |
| Qwen3.5-9B, floor guard | 8 | 0 | 73 |
| Qwen3.5-4B, floor guard | 13 | 1 | 83 |

This removes the fall dismissals but still dismisses 8 to 13 pain-pose alerts, so it would also fail the recall guard. Any revised rule (floor guard, a prompt without the "people often bend or touch their face" priming sentence, or allowing dismissal only for specific stage-1 events) needs its own pre-registered run on clips not used here, such as the flagged calibration-half clips.
