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
| `vlm train multiclass` | `GRUVideoClassifier` | Per class: deterministic 5:1 train/test; default `50` train and `10` test videos | Reports train and test accuracy each epoch |
| `vlm train prototype` | `PrototypeVideoClassifier` | Same deterministic 5:1 train/test split | Reports train and test accuracy each epoch |
| `vlm train binary` | Safe/danger `GRUVideoClassifier` | Per binary label: approximately 2/3 train, 1/6 validation, 1/6 test | Uses validation macro F1, balanced accuracy, and accuracy to select the best checkpoint, then evaluates test |

In `PrototypeVideoClassifier`, “prototype” refers to learnable normalized class
prototypes used to score a GRU-encoded video. It is distinct from the older use
of “prototype” to mean the project's initial proof of concept.

Multiclass and prototype currently have no validation split and inspect test
accuracy each epoch. Adding validation and reserving test for one final
evaluation is tracked in the roadmap below.

### Frame Sampling and Embedding Cache

Training defaults to `--frame-stride 5` and `--max-seq-len 32`:

1. decode the full video and keep frames `0, 5, 10, 15, ...`;
2. encode each retained frame with `openai/clip-vit-base-patch32`;
3. L2-normalize each frame embedding;
4. if more than 32 embeddings remain, select 32 evenly across the sequence;
5. if fewer than 32 remain, zero-pad and retain the true sequence length.

Compatible embedding tensors are reused when the video, model, frame stride,
sequence length, and augmentation identity match. Binary training additionally
creates deterministic augmented cache entries for the minority training class
using horizontal flip, small rotation, color jitter, and random crop by
default. Validation and test videos are not augmented.

The standalone CLIP/XCLIP experiments use a separate strategy: they choose a
seeded random contiguous clip and currently read 8 frames for CLIP/XCLIP or 32
frames for XCLIP vision. This experiment sampling does not affect training.

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

- [ ] Add validation splits to the multiclass and prototype pipelines.
- [ ] Select their best checkpoints using validation metrics and evaluate test
      only once.
- [ ] Decide whether all training pipelines should share early stopping.
- [ ] Keep split generation deterministic and record selected videos and seeds
      in split manifests.

### Video and Frame Sampling

- [ ] Decide whether experiments should represent a random contiguous clip or
      the entire video.
- [ ] If whole-video coverage is intended, sample uniformly across the full
      duration.
- [ ] Consider aggregating multiple temporal windows for long videos.
- [ ] Expose experiment sampling strategy and frame sample rate as CLI options.

### Training Improvements

- [ ] Explore random frame dropout, temporal jitter, or small Gaussian noise on
      embeddings.
- [ ] Consider a pooled-feature branch using temporal mean, max, and standard
      deviation alongside the GRU output.
- [ ] Increase the prototype dataset beyond the current small baseline when the
      experimental scope allows it.
