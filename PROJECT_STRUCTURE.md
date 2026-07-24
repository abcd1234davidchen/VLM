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
│       ├── data/
│       │   ├── discovery.py
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
│       │   └── temporal.py
│       ├── training/
│       │   ├── arguments.py
│       │   ├── binary.py
│       │   ├── engine.py
│       │   ├── multiclass.py
│       │   └── prototype.py
│       ├── evaluation/
│       │   ├── binary.py
│       │   └── multiclass.py
│       ├── reporting/
│       │   ├── artifacts.py
│       │   └── plots.py
│       └── experiments/
│           ├── clip_video.py
│           ├── video_sampling.py
│           ├── xclip.py
│           └── xclip_vision.py
├── archive/
│   ├── README.md
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
├── main.py                                # compatibility wrapper only
├── train_video_classifier.py              # compatibility wrapper only
├── video_embeddings_classifier.py         # compatibility wrapper only
├── train_binary_video_classifier.py       # compatibility wrapper only
├── video_dataset.py                       # compatibility re-exports only
├── video_embedding_cache.py               # compatibility re-exports only
├── clipModelTest.py                       # compatibility wrapper only
├── XclipModelTest.py                      # compatibility wrapper only
└── XCVisionTest.py                        # compatibility wrapper only
```

The complete pre-restructure contents of the old Python files move into
`archive/legacy_scripts/` after local work and the latest `origin/main` have
been integrated. Those archived files are frozen historical references.

The same filenames at the repository root are newly written, minimal
compatibility wrappers. They contain no training, model, cache, evaluation, or
experiment implementation. Active implementation belongs in `src/vlm/`, and
nothing in active code may import from `archive/`.

In other words:

```text
old full implementation  → archive/legacy_scripts/<old filename>
new implementation       → src/vlm/<responsibility-specific module>
old root filename         → tiny compatibility wrapper only
```

Each package directory under `src/vlm/` also contains its corresponding
`__init__.py`.

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
```

Existing training arguments such as dataset path, cache path, artifact path,
model identifier, epochs, batch sizes, learning rate, sequence length, device,
and force-extraction options remain available on the relevant command.

### Embedding extraction

```bash
uv run vlm embeddings extract --pipeline multiclass
uv run vlm embeddings extract --pipeline prototype
uv run vlm embeddings extract --pipeline binary
```

### Experiments

```bash
uv run vlm experiment clip
uv run vlm experiment xclip
uv run vlm experiment xclip-vision
```

## Legacy Command Mapping

These commands remain valid after restructuring:

| Existing command | Unified equivalent |
|---|---|
| `python main.py` | `uv run vlm train multiclass` |
| `python train_video_classifier.py` | `uv run vlm train multiclass` |
| `python video_embeddings_classifier.py` | `uv run vlm train prototype` |
| `python train_binary_video_classifier.py` | `uv run vlm train binary` |
| `python clipModelTest.py` | `uv run vlm experiment clip` |
| `python XclipModelTest.py` | `uv run vlm experiment xclip` |
| `python XCVisionTest.py` | `uv run vlm experiment xclip-vision` |

Arguments supplied to an existing training script are forwarded to the
corresponding unified pipeline.

## Models and Experiments

The following remain available:

- `GRUVideoClassifier`;
- `PrototypeVideoClassifier`;
- binary safe/danger GRU classification;
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
| Multiclass/prototype split | `artifacts/selected_split.json` |
| Multiclass/prototype checkpoint | `artifacts/video_classifier.pt` |
| Multiclass/prototype results | `artifacts/evaluation_results.json` |
| Multiclass/prototype accuracy plot | `artifacts/accuracy_graph.png` |
| Binary split | `artifacts/binary/selected_split.json` |
| Binary checkpoint | `artifacts/binary/binary_video_classifier.pt` |
| Binary results | `artifacts/binary/binary_evaluation_results.json` |
| Binary accuracy plot | `artifacts/binary/accuracy_graph.png` |
