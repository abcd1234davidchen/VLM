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

uv run vlm embeddings extract --pipeline multiclass
uv run vlm embeddings extract --pipeline prototype
uv run vlm embeddings extract --pipeline binary
uv run vlm embeddings extract --pipeline xclip
uv run vlm embeddings extract --pipeline xclip-lora

uv run vlm experiment clip
uv run vlm experiment xclip
uv run vlm experiment xclip-vision
```

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

The four training commands follow the same high-level flow:

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

Default artifacts are isolated under `artifacts/multiclass`, `artifacts/prototype`, `artifacts/binary`, and `artifacts/xclip`.

### Frozen XCLIP Baseline

Run the fourth training pipeline with all discovered classes and videos:

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
