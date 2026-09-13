# VLM Project Structure and Command Guide

> Status: implemented. Update this document whenever a command, module boundary,
> compatibility wrapper, or output location changes.

## Current Structure

```text
.
├── README.md
├── PROJECT_STRUCTURE.md
├── pyproject.toml
├── _compat.py                              # src-layout bootstrap for wrappers
├── src/
│   └── vlm/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── config.py
│       ├── xclip_features.py
│       ├── xclip_pre_mit_features.py
│       ├── skeleton_data.py
│       ├── data/
│       │   ├── discovery.py
│       │   ├── skeleton.py
│       │   ├── splitting.py
│       │   └── video_io.py
│       ├── embeddings/
│       │   ├── augmentation.py
│       │   ├── cache.py
│       │   ├── dataset.py
│       │   └── extractor.py
│       ├── models/
│       │   ├── gru.py
│       │   ├── prototype.py
│       │   ├── temporal.py
│       │   ├── xclip.py
│       │   ├── lora.py
│       │   ├── skeletonclip.py
│       │   └── xclip_lora.py
│       ├── training/
│       │   ├── arguments.py
│       │   ├── binary.py
│       │   ├── engine.py
│       │   ├── multiclass.py
│       │   ├── prototype.py
│       │   ├── xclip.py
│       │   ├── xclip_lora.py
│       │   └── xclip_lora_tune.py
│       ├── evaluation/
│       │   ├── binary.py
│       │   └── multiclass.py
│       ├── reporting/
│       │   ├── artifacts.py
│       │   └── plots.py
│       └── experiments/
│           ├── clip_video.py
│           ├── video_sampling.py
│           ├── skeleton_gate1.py
│           ├── xclip.py
│           └── xclip_vision.py
├── archive/
│   └── legacy_scripts/                    # frozen pre-restructure sources
│       ├── main.py
│       ├── train_video_classifier.py
│       ├── video_embeddings_classifier.py
│       ├── train_binary_video_classifier.py
│       ├── video_dataset.py
│       ├── video_embedding_cache.py
│       ├── clipModelTest.py
│       ├── XclipModelTest.py
│       └── XCVisionTest.py
└── main.py                                # compatibility entry point
```

The complete pre-restructure contents of the old Python files move into
`archive/legacy_scripts/` after local work and the latest `origin/main` have
been integrated. Those archived files are frozen historical references.

The archive is inert: active code and compatibility wrappers must never import
from it, and new features or fixes belong in `src/vlm/`. Git history and the
active package are authoritative; archived files exist only to inspect the
pre-restructure implementations.

The former repository-root script wrappers have been removed. Active
implementation belongs in `src/vlm/`, and nothing in active code may import
from `archive/`. `main.py` remains as a minimal compatibility entry point for
the original default multiclass command.

In other words:

```text
old full implementation  → archive/legacy_scripts/<old filename>
new implementation       → src/vlm/<responsibility-specific module>
old root filename         → removed; use the unified CLI
```

Each package directory under `src/vlm/` also contains its corresponding
`__init__.py`.

`README.md` and this file are the project's only maintained Markdown
documents. `README.md` owns setup, commands, pipeline behavior, sampling,
compatibility policy, research links, and the roadmap. This file owns the
directory tree, module boundaries, legacy mappings, and output locations.

## Installation

```bash
uv sync
```

This installs the editable `src/vlm` package and `vlm` console command.

## Unified Commands

The package supports:

```bash
uv run vlm <command>
uv run python -m vlm <command>
```

### Training

```bash
uv run vlm train multiclass
uv run vlm train prototype
uv run vlm train binary
uv run vlm train xclip
uv run vlm train xclip-lora
uv run vlm train xclip-lora-tune
```

Existing training arguments such as dataset path, cache path, artifact path,
model identifier, epochs, batch sizes, learning rate, sequence length, device,
and force-extraction options remain available on the relevant command.

The `xclip-lora` pipeline uses the same command surface for isolated pre-MIT cache extraction and MIT LoRA + CE training. Its adapter and merged checkpoints remain separate from frozen XCLIP artifacts.

The `xclip-lora-tune` orchestrator verifies and preloads the shared cache once, reuses its in-memory datasets across validation-only trials, promotes a bounded set of candidates between stages, records failed or interrupted states, and exports a reusable winning configuration. After selection is fixed, it delegates one final winner run with test evaluation to the ordinary LoRA pipeline; model, cache, checkpoint, and evaluation contracts remain there.

### Embedding extraction

```bash
uv run vlm embeddings extract --pipeline multiclass
uv run vlm embeddings extract --pipeline prototype
uv run vlm embeddings extract --pipeline binary
uv run vlm embeddings extract --pipeline xclip
uv run vlm embeddings extract --pipeline xclip-lora
```

### Experiments

```bash
uv run vlm experiment clip
uv run vlm experiment xclip
uv run vlm experiment xclip-vision
uv run vlm experiment skeleton-gate1
uv run vlm data skeleton-audit --help
```

## Legacy Command Equivalents

Use these unified commands in place of the former standalone scripts:

| Former command | Unified equivalent |
|---|---|
| `python main.py` | `uv run vlm train multiclass` |
| `python train_video_classifier.py` | `uv run vlm train multiclass` |
| `python video_embeddings_classifier.py` | `uv run vlm train prototype` |
| `python train_binary_video_classifier.py` | `uv run vlm train binary` |
| `python clipModelTest.py` | `uv run vlm experiment clip` |
| `python XclipModelTest.py` | `uv run vlm experiment xclip` |
| `python XCVisionTest.py` | `uv run vlm experiment xclip-vision` |

The archived scripts are not active entry points and should not be used for new
runs.

Former compatibility imports map to the active package:

| Former import surface | Active package |
|---|---|
| `video_dataset` | `vlm.data` |
| `video_embedding_cache` | `vlm.embeddings` |
| `train_video_classifier` | `vlm.training.multiclass`, `vlm.models.gru`, and `vlm.evaluation.multiclass` |
| `video_embeddings_classifier` | `vlm.training.prototype` and `vlm.models.prototype` |
| `train_binary_video_classifier` | `vlm.training.binary`, `vlm.models.gru`, and `vlm.evaluation.binary` |
| CLIP/XCLIP test scripts | `vlm.experiments` |

The underlying models, data helpers, cache operations, training pipelines,
evaluation functions, and experiments remain available. Only imports through
the removed repository-root wrapper modules no longer resolve.

## Models and Experiments

The following remain available:

- `GRUVideoClassifier`;
- `PrototypeVideoClassifier`;
- binary safe/danger GRU classification;
- frozen XCLIP video-feature classification;
- pre-MIT XCLIP projected-frame feature extraction for MIT LoRA training;
- MIT q/v LoRA + CE classification and merged RGB-only MIT inference;
- per-frame CLIP image/text similarity;
- XCLIP video/text similarity;
- XCLIP vision feature extraction.

Default pretrained identifiers remain:

- `openai/clip-vit-base-patch32`;
- `microsoft/xclip-base-patch32`.

## Outputs

Legacy defaults remain supported:

| Output | Default location |
|---|---|
| Embedding tensors | `cache/embeddings/` |
| Embedding manifest | `cache/embedding_manifest.json` |
| Multiclass artifacts | `artifacts/multiclass/` |
| Prototype artifacts | `artifacts/prototype/` |
| Binary split | `artifacts/binary/selected_split.json` |
| Binary checkpoint | `artifacts/binary/binary_video_classifier.pt` |
| Binary results | `artifacts/binary/binary_evaluation_results.json` |
| Binary accuracy plot | `artifacts/binary/accuracy_graph.png` |
| XCLIP frozen features | `cache/xclip_features/` |
| XCLIP feature manifest | `cache/xclip_feature_manifest.json` |
| XCLIP pre-MIT features | `cache/xclip_pre_mit_features/` |
| XCLIP pre-MIT manifest | `cache/xclip_pre_mit_feature_manifest.json` |
| XCLIP split | `artifacts/xclip/selected_split.json` |
| XCLIP checkpoint | `artifacts/xclip/xclip_classifier.pt` |
| XCLIP results | `artifacts/xclip/evaluation_results.json` |
| XCLIP accuracy plot | `artifacts/xclip/accuracy_graph.png` |
| XCLIP LoRA CE artifacts | `artifacts/xclip_lora_ce/` |
| XCLIP LoRA tuning summary and trials | `artifacts/xclip_lora_tuning/` |
| SkeletonCLIP Gate 0 audit | `artifacts/skeleton_data/` |
| SkeletonCLIP Gate 1 compatibility report | `artifacts/skeletonclip_gate1/` |
