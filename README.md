# VLM-Based Home Care System

This project explores video understanding for home-care scenarios with CLIP,
XCLIP, temporal GRU classifiers, and cached video-frame embeddings.

The active implementation is organized as an installable `src/vlm` package
with one unified `vlm` command. Existing script calls and Python imports remain
available through compatibility wrappers.

## Models and Experiments

The project retains:

- multiclass CLIP embeddings with `GRUVideoClassifier`;
- multiclass CLIP embeddings with `PrototypeVideoClassifier`;
- binary safe/danger GRU classification;
- per-frame CLIP image/text similarity experiments;
- XCLIP video/text similarity experiments;
- XCLIP vision feature extraction;
- shared embedding caching, augmentation, evaluation, checkpoints, and plots.

## Setup

```bash
uv sync
```

This installs the local package and its `vlm` console command.

## Commands

The primary interface is:

```bash
uv run vlm train multiclass
uv run vlm train prototype
uv run vlm train binary

uv run vlm embeddings extract --pipeline multiclass

uv run vlm experiment clip
uv run vlm experiment xclip
uv run vlm experiment xclip-vision
```

The equivalent `uv run python -m vlm ...` form is also supported.

Existing commands such as the following continue to work through thin
compatibility wrappers:

```bash
python train_video_classifier.py
python video_embeddings_classifier.py
python train_binary_video_classifier.py
```

## Project Documentation

- [Project structure and command guide](PROJECT_STRUCTURE.md)
- [Prototype pipeline notes](README_PROTOTYPE.md)
- [Restructure design](docs/superpowers/specs/2026-07-24-vlm-project-restructure-design.md)

The complete pre-restructure scripts will be retained under
`archive/legacy_scripts/`. Active implementation lives under `src/vlm/`.

## Development Stages

- [ ] Data collection
  - [ ] Charades/LEMMA dataset
  - [ ] Toyota Smart Home dataset
  - [ ] NTU RGB+D dataset
- [ ] Data preprocessing
- [ ] Model training
- [ ] Model evaluation
- [ ] Deployment
