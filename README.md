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

uv run vlm embeddings extract --pipeline multiclass
uv run vlm embeddings extract --pipeline prototype
uv run vlm embeddings extract --pipeline binary

uv run vlm experiment clip
uv run vlm experiment xclip
uv run vlm experiment xclip-vision
```

The equivalent `uv run python -m vlm ...` form is also supported. Run any
command with `--help` to see its dataset, cache, artifact, model, sampling, and
training options.

## Training Pipelines

All three training commands follow the same high-level flow:

```text
discover videos
→ construct deterministic splits
→ create or reuse CLIP frame-embedding cache
→ load fixed-length embedding sequences
→ train the selected temporal classifier
→ evaluate
→ write checkpoint, JSON results, split manifest, and accuracy plot
```

They share data loading, CLIP extraction, caching, training utilities,
evaluation helpers, and artifact writers, while retaining task-specific
models and evaluation rules.

| Command | Model | Current split | Evaluation behavior |
|---|---|---|---|
| `vlm train multiclass` | `GRUVideoClassifier` | All discovered videos, split into participant-disjoint train/validation/test | Selects the best checkpoint by validation accuracy, then evaluates test once |
| `vlm train prototype` | `PrototypeVideoClassifier` | Same all-video, participant-disjoint train/validation/test split | Selects the best checkpoint by validation accuracy, then evaluates test once |
| `vlm train binary` | Safe/danger `GRUVideoClassifier` | Participant-disjoint; approximately 2/3 train, 1/6 validation, 1/6 test | Uses validation macro F1, balanced accuracy, and accuracy to select the best checkpoint, then evaluates test once |

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

The standalone XCLIP and XCLIP-vision experiments deterministically select 8 frames uniformly across the complete video, matching the pretrained checkpoint's native frame count. The standalone per-frame CLIP experiment still uses its separate seeded random contiguous-clip strategy. Experiment sampling does not affect the three training pipelines.

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
- XCLIP vision feature extraction.

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

Default artifacts are isolated under `artifacts/multiclass`,
`artifacts/prototype`, and `artifacts/binary`.

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
