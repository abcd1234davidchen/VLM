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

The next skeleton-guided vision candidate keeps the 120-class CE objective and tests motion-aware spatial attention that directly participates in frame pooling. Taxonomy aggregation remains an evaluation and inference operation so representation gains are not confused with label merging; see `docs/skeleton-guided-spatial-supervision-plan.md`.

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
