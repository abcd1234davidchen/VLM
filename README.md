# VLM-Based Home Care System

This project explores video understanding for home-care scenarios with CLIP,
XCLIP, temporal GRU classifiers, and cached video-frame embeddings.

The active implementation is organized as an installable `src/vlm` package
with one unified `vlm` command.

## Setup

```bash
uv sync
```

This installs the local package and its `vlm` console command.

## Commands

Use the unified CLI for active development:

```bash
uv run vlm train multiclass
uv run vlm train prototype
uv run vlm train binary
uv run vlm train xclip
uv run vlm train xclip-lora
uv run vlm train xclip-lora-tune --split-manifest artifacts/xclip/selected_split.json

uv run vlm embeddings extract --pipeline multiclass
uv run vlm embeddings extract --pipeline prototype
uv run vlm embeddings extract --pipeline binary
uv run vlm embeddings extract --pipeline xclip
uv run vlm embeddings extract --pipeline xclip-lora

uv run vlm experiment clip
uv run vlm experiment xclip
uv run vlm experiment xclip-vision
uv run vlm experiment skeleton-gate1
```

Run the SkeletonCLIP Phase 1 Gate 0 audit against the canonical RGB split with:

```bash
uv run vlm data skeleton-audit \
  --skeleton-dir RGB_vids/nturgb+d_skeletons \
  --rgb-manifest artifacts/xclip/selected_split.json \
  --missing-skeletons /path/to/NTU_RGBD_samples_with_missing_skeletons.txt
```

The missing/incomplete sample list must come from the [NTU RGB+D authors' repository](https://github.com/shahroudy/NTURGB-D/blob/master/Matlab/NTU_RGBD_samples_with_missing_skeletons.txt). The command fully validates all 56,880 NTU60 skeleton files, pairs only canonical train/validation records by complete `S/C/P/R/A` identity, never creates test pairings, and writes `coverage.json`, `paired_manifest.json`, and `excluded_samples.json` under `artifacts/skeleton_data/`. Following Gate 0 review, A060 is explicitly excluded and the Phase 1 teacher contract uses ordered classes A001–A059. Pass one or more `--archive` arguments when safe extraction is needed; existing files are never overwritten.

`vlm experiment skeleton-gate1` runs CPU contract checks and then sequential single-GPU compatibility smoke for the BlockGCN and Hyperformer adapters. It uses 16 real paired training samples, requires at least 95% overfit accuracy, enforces finite loss/gradients, verifies strict checkpoint reload and eval parity, records throughput and peak VRAM, and writes `artifacts/skeletonclip_gate1/compatibility_report.json`. It does not start the Gate 2 bake-off.

Gate 3 uses an adaptive full-run funnel. Complete the BlockGCN linear baseline with seeds 42 and 43, then run the frozen XCLIP text-prototype head with seed 42 first. Run XCLIP seed 43 only when seed 42 is within the predeclared 2 percentage-point macro-F1 quality gate; the OpenAI CLIP reference is a conditional diagnostic rather than a required run. This policy was locked before any XCLIP full-run result was observed. Example commands:

```bash
uv run vlm train skeleton-gate3 --heads linear --seeds 42 43
uv run vlm train skeleton-gate3 --heads xclip --seeds 42
```

Gate 3 is complete. The two-seed linear and XCLIP heads reached 86.47% and 86.16% mean validation macro-F1 respectively. XCLIP improved ECE from 0.0741 to 0.0596 but worsened NLL from 0.6358 to 0.6523, so it did not pass the predeclared rule requiring one calibration metric to improve without worsening the other. The selected first-pass distillation teacher is the BlockGCN linear seed-42 best checkpoint; XCLIP prototypes remain an independent semantic-alignment result. The next phase freezes that teacher, caches raw 59-class logits for valid paired training samples, and integrates explicit A001–A059 masked KL into the existing 120-class XCLIP MIT LoRA student.

Phase 2 Gates 0–2 are complete. Revalidate the immutable experiment lock, the 45,236-entry FP32 teacher-logit cache, and the masked-KL integration smoke with `uv run vlm train skeleton-phase2 --through-gate 2 --batch-size 64 --workers 4`. This command does not evaluate the test split. Gate 3 CE parity and Gate 4 validation-only search are also complete; see `docs/skeletonclip-distillation-phase2.md` for the locked contract and exact results.

Phase 2 Gate 3 subsequently passed CE parity at 52.1065% validation accuracy (`-0.0154` percentage points versus the historical seed-42 control). Gate 4 completed the locked nine-candidate validation-only KD search with a no-go result: the best candidate (`T=1, λ=0.1`) reached 51.9907%, below control by 0.1157 points. No test evaluation was performed and no seed-43 confirmation is planned for this first class-logit KD branch. See the Phase 2 document and `artifacts/skeletonclip_phase2/search_summary.json` for complete results.

`xclip-lora` trains project-owned LoRA adapters on the XCLIP MIT q/v attention projections plus a new classification head. `uv run vlm embeddings extract --pipeline xclip-lora` builds only the isolated pre-MIT cache, while `uv run vlm train xclip-lora` performs cache preparation and CE-only training.

XCLIP cache extraction can overlap CPU video decoding with batched GPU inference. For an 11 GB GPU, start with:

```bash
uv run vlm embeddings extract --pipeline xclip \
  --device cuda --feature-batch-size 4 --decode-workers 4
```

Increase `--feature-batch-size` only while GPU memory has headroom. Existing compatible per-video cache files are still reused, so interrupted extraction can resume without `--force-extract`.

The equivalent `uv run python -m vlm ...` form is also supported. Run any
command with `--help` to see its dataset, cache, artifact, model, sampling, and
training options.

## Live Alert Demo

The live demo runs the locked stage-1 configuration (A on center_zoom70 plus B on the person crop) as the detector on a sliding window from the webcam or a local video file, raises an alert when the falling (A043) or nausea/vomiting (A048) probability reaches its own calibrated threshold in one step (`--persist`), and asks the verifier, a local VLM, to check the alert window together with 4 frames from the window before it and 4 from the window after it, so it sees what led up to the alert and how it ended; the first check therefore goes out one window (2.5 seconds) after the alert. Staggering and the body-pain poses (A042, A044–A047) are not alerted because their acted NTU versions are not reliable enough:

```bash
uv run --with mediapipe python -m vlm.demo.server
```

Open `http://127.0.0.1:8642`. It needs `artifacts/patch16_crop_aware/seed42/final_merged.pt`, `artifacts/patch16_person_crop/seed42/final_merged.pt`, and `artifacts/hand_roi_models/pose_landmarker_lite.task`. The verifier runs `mlx-vlm` in its own isolated environment and needs Apple Silicon; pass `--verifier-model none` to run the detector alone. The page fits one screen: the camera and the falling and vomiting scores stay in view, and `j`/`k` or the arrow keys switch the side pane between the latest verifier check, the frames fed to each model, the detector outputs, the alert history, the step log, and the settings. When the verifier sees a normal activity such as exercise, the alert is held rather than deleted: the verifier rechecks the newest 8-frame window every 15 seconds (`--recheck`), any recheck that is not a normal activity notifies, and three normal rechecks in a row (`--rechecks`) clear the alert, which stays in the log. If there is no new video to recheck, for example because a video file ended, the alert ends without notifying and stays in the log. This hold rule is a user decision recorded in `docs/vlm-verifier-bakeoff-plan.md` and is not yet validated. Alert frames and verifier answers are saved under `cache/demo_alerts/`.

Phone notifications are optional. Install the ntfy app on each phone and subscribe to one hard-to-guess topic, install `cloudflared` (`brew install cloudflared`), and put the topic in `demo_local.json` (copy `src/vlm/demo/demo_local.example.json`; the file is ignored by git, and `--ntfy-topic` overrides it). When the server starts, a first push asks each phone to register a name. Every notified alert is then pushed through ntfy.sh with only the title "Possible falling detected!!!" and a link to an alert page that loops the frames the verifier saw. Opening the page marks the alert as seen by that phone's registered name, and the page has a demo-only Call 911 button that places no call but records who pressed it; both responses show on the demo page. The page also shows the person's name, address and notes from `demo_local.json`. Nothing is uploaded: a separate local server answers only paths under random tokens, alert links for 24 hours (`--share-hours`), with `Cache-Control: private` so Cloudflare keeps no copy, and a Cloudflare Quick Tunnel (no account or domain) makes it reachable. The tunnel address changes on every start, so links in old notifications stop working and phones register again after a restart. `--no-tunnel` sends text only, and Settings has a test button.

## Training Pipelines

The classifier training commands follow this high-level flow:

```text
discover videos
→ construct or reuse a deterministic participant-disjoint split
→ create or reuse the pipeline's frozen feature cache
→ train the selected classifier
→ choose the best validation checkpoint
→ evaluate test once
→ write checkpoint, JSON results, split manifest, and accuracy plot
```

They share split construction, validation checkpoint selection, evaluation helpers, and artifact writers while retaining model-specific feature caches and evaluation rules.

| Command | Model | Current split | Evaluation behavior |
|---|---|---|---|
| `vlm train multiclass` | `GRUVideoClassifier` | All discovered videos, split into participant-disjoint train/validation/test | Selects the best checkpoint by validation accuracy, then evaluates test once |
| `vlm train prototype` | `PrototypeVideoClassifier` | Same all-video, participant-disjoint train/validation/test split | Selects the best checkpoint by validation accuracy, then evaluates test once |
| `vlm train binary` | Safe/danger `GRUVideoClassifier` | Participant-disjoint; approximately 2/3 train, 1/6 validation, 1/6 test | Uses validation macro F1, balanced accuracy, and accuracy to select the best checkpoint, then evaluates test once |
| `vlm train xclip` | Frozen XCLIP video encoder plus `LayerNorm → Dropout → Linear` head | Same all-video, participant-disjoint train/validation/test split | Selects the best checkpoint by validation accuracy, then evaluates test once |
| `vlm train xclip-lora` | XCLIP MIT q/v LoRA plus classification head | Reuses a canonical participant-disjoint manifest and pre-MIT cache | Selects by validation accuracy, then evaluates test once |
| `vlm train xclip-lora-tune` | Staged orchestrator over the ordinary LoRA model | Requires one canonical manifest for every trial | Ranks and confirms recipes using validation only, then evaluates the locked winner on test once |

In `PrototypeVideoClassifier`, “prototype” refers to learnable normalized class
prototypes used to score a GRU-encoded video. It is distinct from the older use
of “prototype” to mean the project's initial proof of concept.

The NTU participant identifier (`P###`) is assigned to exactly one split, so
different actions, repetitions, and camera views from the same participant
cannot leak across train, validation, and test. The split manifest records the
seed, strategy, selected videos, and participant IDs for each split.

Multiclass and prototype use all discovered videos by default. Set
`--max-videos-per-class N` to cap the total number per class for a faster
experiment. The older `--videos-per-split N` option remains as a deprecated
compatibility alias for a `6N` total per-class budget. Use `--val-fraction`
and `--test-fraction` to change the participant allocation ratios.

At startup, both commands print the total video count, participant count, and
per-class count for each split. The same summary is stored in the evaluation
results JSON.

### Frame Sampling and Embedding Cache

Training defaults to `--sample-fps 6` and `--max-seq-len 32`:

1. read the source frame rate and select frames on a deterministic six-frames-per-second time grid;
2. if more than 32 frames are selected, retain 32 evenly across the complete video;
3. encode the selected frames with `openai/clip-vit-base-patch32` and L2-normalize each embedding;
4. store only the real embedding sequence in cache, without fixed-length zero padding;
5. when a batch is assembled, pad only to that batch's longest sequence and retain each video's true length for packed GRU processing and masked pooling.

Compatible embedding tensors are reused when the video, model, sampling strategy, sample FPS, maximum sequence length, storage format, and augmentation identity match. Binary training additionally creates deterministic augmented cache entries for the minority training class using horizontal flip, small rotation, color jitter, and random crop by default. Validation and test videos are not augmented.

Cache filenames include a hash of the source video path, so videos with the
same filename in different directories cannot collide.

The XCLIP training pipeline and standalone XCLIP experiments deterministically select 8 frames uniformly across the complete video, matching the pretrained checkpoint's native frame count. Short videos repeat selected real frames rather than adding zero frames. The frozen training pipeline calls `XCLIPModel.get_video_features()`, caches one 512-dimensional video representation, and trains only a `LayerNorm → Dropout → Linear` classification head. The standalone per-frame CLIP experiment still uses its separate seeded random contiguous-clip strategy.

## Models and Experiments Maintained

- `GRUVideoClassifier`;
- `PrototypeVideoClassifier`;
- binary safe/danger GRU classification;
- shared CLIP frame-embedding extraction and cache;
- train-only binary frame augmentation and optional class weights;
- multiclass and binary evaluation metrics;
- checkpoint, JSON, manifest, and accuracy-plot output;
- per-frame CLIP image/text similarity;
- XCLIP video/text similarity;
- XCLIP vision feature extraction;
- frozen XCLIP video-feature classification.

Default pretrained models remain:

- `openai/clip-vit-base-patch32`;
- `microsoft/xclip-base-patch32`.

### Reusing a Split

Each training command can reuse an existing split manifest:

```bash
uv run vlm train multiclass \
  --split-manifest artifacts/shared/selected_split.json
```

Loaded manifests are checked for duplicate videos, mismatched split labels,
and participants appearing in more than one split. Multiclass and prototype
derive contiguous action indices from the selected classes; binary derives its
safe/danger labels from the same video records.

Default artifacts are isolated under `artifacts/multiclass`, `artifacts/prototype`, `artifacts/binary`, `artifacts/xclip`, `artifacts/xclip_lora_ce`, and `artifacts/xclip_lora_tuning`.

### Frozen XCLIP Baseline

Run the frozen XCLIP training pipeline with all discovered classes and videos:

```bash
uv run vlm train xclip
```

For a faster cache-only server preparation or smoke run:

```bash
uv run vlm embeddings extract --pipeline xclip \
  --max-videos-per-class 6
```

The first run downloads `microsoft/xclip-base-patch32`, samples eight deterministic whole-video frames per video, and writes frozen features under `cache/xclip_features/`. Training writes `selected_split.json`, `xclip_classifier.pt`, `evaluation_results.json`, and `accuracy_graph.png` under `artifacts/xclip/`. Pass the same canonical manifest used by another pipeline through `--split-manifest`; video membership remains identical while the XCLIP artifact records its own eight-frame sampling policy.

### XCLIP MIT LoRA Cache Foundation

Prepare the LoRA pipeline's projected per-frame features with the same split and sampling policy:

```bash
uv run vlm embeddings extract --pipeline xclip-lora \
  --split-manifest artifacts/xclip/selected_split.json
```

This command stores `[8, 512]` projected frame tensors under `cache/xclip_pre_mit_features/` and writes `cache/xclip_pre_mit_feature_manifest.json`. It does not read or overwrite the frozen post-MIT `[512]` cache.

Run the CE-only MIT LoRA control experiment with:

```bash
uv run vlm train xclip-lora \
  --split-manifest artifacts/xclip/selected_split.json
```

The default run trains rank-8 LoRA adapters on every MIT self-attention `q_proj` and `v_proj` plus the `LayerNorm → Dropout → Linear` head for 60 epochs. It selects the checkpoint by validation accuracy, evaluates test once after restoring the best epoch, and writes adapter and merged checkpoints, `run_config.json`, `evaluation_results.json`, and `accuracy_graph.png` under `artifacts/xclip_lora_ce/`.

Run the staged hyperparameter search with:

```bash
uv run vlm train xclip-lora-tune \
  --split-manifest artifacts/xclip/selected_split.json
```

The tuning command prepares the shared pre-MIT cache once, runs curated optimization, LoRA-capacity, regularization, feature-augmentation, and multi-seed confirmation stages, and selects only from validation accuracy. Stable trial IDs and `artifacts/xclip_lora_tuning/search_summary.json` make interrupted searches resumable; individual trials never evaluate the test split. After the winner is irrevocably selected, the orchestrator automatically runs that recipe once through the ordinary `xclip-lora` path, evaluates test, and writes the final artifacts under `artifacts/xclip_lora_tuning/winner/`. The reusable recipe remains available as `winning_config.json`.

Feature augmentation is sampled after cached tensors are loaded and is training-only. `--frame-dropout`, `--temporal-jitter`, and `--feature-noise-std` therefore leave the canonical cache unchanged; `--warmup-epochs` and `--early-stopping-patience` support optimization-stage trials.

The tuning orchestrator verifies the shared cache once, preloads its feature tensors into RAM once, and reuses the same in-memory datasets across every trial. This avoids repeated scans and per-sample `.pt` reads; use `--no-preload-cache` only when host memory is constrained. Validation still runs every epoch for checkpoint selection, while tuning records full train-set accuracy every five epochs by default through `--train-eval-frequency 5`.

Tuning uses compact SSH-friendly logging by default: batch and cache progress bars are disabled, per-trial configuration and result JSON remain in artifacts instead of stdout, and epoch metrics print every five epochs. These logging controls do not change optimizer updates, validation frequency, checkpoint selection, or stored histories.

### Recorded XCLIP LoRA Result

The completed 2026-08-26 search selected rank-16／alpha-32 MIT q/v LoRA with LoRA learning rate `2e-4`, head learning rate `8e-4`, head dropout `0.25`, and Gaussian feature noise `0.01`. On the fixed 80,640／12,960／20,880 train／validation／test split, the validation-selected epoch-49 checkpoint reached 52.12% validation accuracy and 54.52% test accuracy, improving over the recorded frozen XCLIP baseline by 12.85 and 11.98 percentage points respectively. The final model trains 95,352 parameters, or 0.0485% of full XCLIP, and its merged checkpoint passed parity with maximum absolute error `1.91e-6`.

The final artifacts are stored under `artifacts/xclip_lora_tuning/winner/`; `evaluation_results.json` contains the complete history and 20,880 test predictions, while `run_config.json` records the exact recipe and Git revision `9f1d5618f226fbef31c3f316`. This tuned CE-only model is the fixed RGB student control for the next SkeletonCLIP distillation experiment; future teacher-specific hyperparameters must still be selected without using test results.

### Reduced Action Taxonomy

The `reduced-v1` taxonomy retains 103 named NTU120 actions and maps 17 source actions into three internal irrelevant groups, which collapse to one public `irrelevant` label. Use `vlm experiment action-taxonomy` to aggregate an existing 120-class probability file without retraining, or pass `--action-taxonomy reduced-v1` to `vlm train xclip-lora` and `vlm train skeleton-phase3-gate1` for the direct 106-output controls.

On the canonical validation split, inference-time aggregation of the existing 120-class upper-vision checkpoint reached 65.8565% exact and 67.8395% public accuracy. A direct 106-output upper-vision retrain reached 61.9830% exact accuracy, so the current deployment candidate remains the 120-class checkpoint plus probability aggregation; the direct checkpoint is retained only as a training control. The mapping, metrics, retained-action checks, and experiment decisions are documented in `docs/action-scope-review.md`.

Motion-guided spatial pooling completed two 120-class pilots: joint adaptation reached 58.11% accuracy versus the 58.92% baseline; attention-only training with a frozen baseline reached 58.94%, only three additional correct validation samples. Neither is selected as a winner. Those runs and checkpoints are preserved; see [results](docs/spatial-attention-v2-results.md). After the user's pause and subsequent request to try the SKI approach, work resumed on a distinct [SKI-style semantic experiment](docs/ski-semantic-v1-plan.md), not another motion-pooling variant.

The SKI-style semantic smoke passed and its single bounded pilot completed in tmux: supervised120 accuracy reached 58.98% (+0.0617 percentage points, eight additional correct validation samples). This partial59 online semantic adaptation is inconclusive, not an official zero-shot reproduction or a selected winner; logs and the full result are recorded in the [semantic experiment report](docs/ski-semantic-v1-plan.md).

Following read-only checkpoint analysis and the user's approval, [direct decision distillation with RGB retention](docs/decision-distillation-v1-plan.md) completed its smoke and single tmux pilot: 58.96% accuracy, five additional correct validation samples overall (paired -15, unpaired +20). It remains inconclusive; unpaired output drift was smaller, but skeleton guidance did not establish a paired classification gain. This frozen-teacher supervised adaptation is not official SKI reproduction.

The user's requested [RGB-input and alert audit](docs/rgb-input-and-alert-audit.md) completed without training: on a fixed480 validation subset, whole-video input reached 60.83%, center zoom70 reached 68.75%, and their equal-probability mean reached 72.29%. These are subset diagnostics, not full-validation improvements or deployment approval. Full canonical validation subsequently confirmed 58.92% whole, 63.91% zoom, and 66.94% whole-plus-zoom accuracy across 12,960 samples; the ensemble requires two forwards. A short [crop-aware pilot](docs/crop-aware-v1-plan.md) reached 67.14% ensemble accuracy but reduced attention-event correct predictions from 349 to 343, so it remains inconclusive and is not recommended as a replacement. The audit also lists the 20 most confused raw120 actions (including previously ignored ones) and exploratory thresholds for A042–A048 attention-associated poses; those scores are not clinical danger probabilities. Thresholds and the current 24-action product exclusion scope remain offline analysis and product decisions, with no runtime alarm or output filtering enabled; see [current status](docs/whole-zoom-alert-future-work.md).

A separate [native attention supervision plan](docs/native-attention-supervision-v1-plan.md) guides the original XCLIP CLS-to-patch attention in its final two vision layers using training-only skeleton targets. It preserves RGB-only inference. The 472-sample audit and eight-step GPU smoke passed their execution contracts. The full-data two-epoch pilot completed at 58.94% raw120 accuracy, only three additional correct samples; the retained 96-class scope fell from 68.13% to 67.79%. Attention guidance changed as intended, but this recipe remains inconclusive and is not recommended as a replacement. This is distinct from the earlier added pooling head.

## Archive and Legacy Code

The complete pre-restructure standalone implementations are frozen under
`archive/legacy_scripts/`. Active code must not import or execute code from
that directory.

The former standalone script names are historical references, not maintained
entry points. Their workflows remain available through the `vlm` Python
package and unified CLI; only the old filename-based calls and imports have
been removed. Use `vlm.data`, `vlm.embeddings`, `vlm.models`, and the other
package modules for Python imports.

Use the unified `vlm` commands for active workflows. New features and fixes
belong in `src/vlm/`. See
[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for module responsibilities,
legacy command equivalents, and default output locations.

## Research Resources

- [LLAVIDAL](https://github.com/ADL-X/LLAVIDAL)
- [ADL-X Evals dataset](https://huggingface.co/datasets/dreilly/ADL-X-Evals/viewer/default/traina)
- [NTU RGB+D action-recognition dataset](https://rose1.ntu.edu.sg/dataset/actionRecognition/)

## Roadmap

### Dataset Splits and Evaluation

- [x] Split train, validation, and test by participant.
- [x] Add validation splits to the multiclass and prototype pipelines.
- [x] Select their best checkpoints using validation metrics and evaluate test
      only once.
- [ ] Decide whether all training pipelines should share early stopping.
- [x] Keep split generation deterministic and record selected videos, seeds,
      strategy, and participants in split manifests.

### Video and Frame Sampling

- [x] Use deterministic whole-video uniform 8-frame sampling for XCLIP experiments.
- [x] Add a frozen XCLIP video-feature classification pipeline.
- [ ] Decide whether the standalone per-frame CLIP experiment should retain a random contiguous clip or cover the entire video.
- [ ] Consider aggregating multiple temporal windows for long videos.
- [ ] Expose experiment sampling strategy and frame sample rate as CLI options.

### Training Improvements

- [ ] Explore random frame dropout, temporal jitter, or small Gaussian noise on
      embeddings.
- [ ] Consider a pooled-feature branch using temporal mean, max, and standard
      deviation alongside the GRU output.
- [ ] Increase the prototype dataset beyond the current small baseline when the
      experimental scope allows it.

A [hand/context crop feasibility audit](docs/hand-roi-v1-audit.md) now compares the existing patch32 classifier on fixed local views. The [RGB localization results](docs/rgb-hand-roi-v1-results.md) use the same 176 validation clips: a fixed whole+zoom+interaction mean improves from 104 to 121 correct, with 21 fixes and four regressions. Coughing and eating show gains; fanning and neck touching do not improve in the combined view. This is a small diagnostic, not full 86-class accuracy or a new trained model; baseline remains unchanged.

The [expanded hand ROI follow-up](docs/hand-roi-followup-v1.md) evaluates 1,228 fixed validation clips and trains a separate local expert for one pass on 2,280 train clips. Compared with the untrained local-view ensemble, the trained ensemble adds 26 correct overall and 21 within retained classes. Eating improves from 35 to 45/72 and coughing from 37 to 49/72 versus original whole+zoom; neck touching remains worse at 20 versus 24/72. The original whole/zoom checkpoint stays unchanged. This stratified diagnostic is not full 86-class accuracy, and the candidate is not adopted.

The [full retained86 RGB interaction validation](docs/hand-roi-full86-v1-results.md) subsequently completed on all 8,640 canonical validation clips in scope: original whole+zoom reached 79.9769%, and the fixed ensemble with the trained local expert reached 84.2245% (+4.2477 percentage points; 505 fixes, 138 regressions). Outside the prior diagnostic subset, 7,548 clips gained 283 correct. All 120 output logits are retained; there was no additional training or test evaluation. Six retained classes declined, including neck touching (24 to 20/72); evaluation does not automatically replace deployment models.

The [XCLIP RGB model path and vision LoRA plan](docs/xclip-rgb-vision-lora-path.md) records the pretrained-to-current checkpoint lineage, historical tuning, scope differences, and the completed expansion trial of the local expert from the final two to four vision blocks. The upper4 runner reuses the Phase 3 initialization and existing 570-step budget; strict initialization, CPU checks and GPU smoke passed, and the 570-step pilot completed. The [full retained86 upper4 evaluation](docs/hand-roi-upper4-v1-results.md) reached 84.5486% versus upper2 84.2245% (48 fixes, 20 regressions, net +28); outside the diagnostic subset it gained 15 correct. This is a limited single-seed gain; neck touching remains below original whole+zoom and deployment is unchanged. `python -m vlm.training.hand_roi_upper4 --artifacts-dir <fresh-directory>` runs smoke, pilot and full retained86 evaluation with cached RGB localization and paired upper2 comparisons.

The [middle4 layer-position trial](docs/hand-roi-middle4-v1-plan.md) trains vision blocks6–9 while preserving frozen baseline adapters in blocks10–11. It keeps the upper4 trainable parameter count and 570-step budget; CPU checks, 14 existing LoRA tests and GPU smoke passed. The [full middle4 evaluation](docs/hand-roi-middle4-v1-results.md) completed at 84.3866% (7,291/8,640): net +14 versus upper2, net −14 versus upper4. It is not recommended or extended; neck touching stays at 20/72. The fixed stages run through `python -m vlm.training.hand_roi_upper4 --vision-layout middle4 --artifacts-dir <fresh-directory>`.

The [mixed4 trial](docs/hand-roi-mixed4-v1-plan.md) trains blocks6/7/10/11, retaining adaptation of the existing tail adapters while moving the two new adapters from blocks8/9 to 6/7. It keeps rank16, the upper4 parameter budget and 570 steps; CPU checks, 14 existing LoRA tests and GPU smoke passed, including actual tail updates. The [full mixed4 evaluation](docs/hand-roi-mixed4-v1-results.md) completed at 84.3981% (7,292/8,640): net +15 versus upper2, −13 versus upper4 and +1 versus middle4. All 325 independent artifact checks passed. This recipe is not adopted or extended; neck touching stays at 20/72. Run all fixed stages with `python -m vlm.training.hand_roi_upper4 --vision-layout mixed4 --artifacts-dir <fresh-directory>`.

The [RGB improvement series](docs/hand-roi-improvement-series-v1-plan.md) sequentially tests cached fusion calibration, a fresh upper4 expert on 9,120 train videos for 2,280 steps, two eight-frame temporal samples and a 1.25x context crop. A fixed participant split separates calibration from confirmation within development validation; settings are selected only on calibration, and no test inference occurs. The [completed results](docs/hand-roi-improvement-series-v1-results.md) reached 85.4630% (7,384/8,640) for the calibration-selected final configuration, net +79 versus upper4 and +25 on confirmation. The highest observed full-validation score was 85.5440% for combined inputs with locked fusion; it does not override calibration selection. Neck touching improved from 22 to 29/72, while A042–A048 declined by two correct predictions overall. Seven new contract tests, 14 existing LoRA tests, GPU smoke/reload checks, 117 artifact checks and an independent NumPy reconstruction passed; no deployment model is automatically replaced. The independent entry point is `python -m vlm.training.hand_roi_improvement_series --artifacts-dir <fresh-directory>`; localization uses the existing isolated MediaPipe environment described in the plan.
