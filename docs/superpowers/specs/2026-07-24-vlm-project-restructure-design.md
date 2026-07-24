# VLM Project Restructure Design

## Objective

Replace the current collection of overlapping executable Python scripts with a
cohesive `vlm` package and one unified command-line interface while preserving
all existing models, experiments, callable Python symbols, script entry points,
arguments, and generated artifacts.

The current working tree is the functional baseline. Before restructuring, the
latest `origin/main` changes must be integrated with the local uncommitted work.
Both sides' functionality must be retained. No commits or pushes are permitted
unless the user explicitly authorizes them later.

## Scope

The restructure includes:

- multiclass CLIP-embedding GRU training;
- multiclass prototype-similarity training;
- binary safe/danger GRU training;
- shared CLIP embedding extraction and caching;
- frame augmentation and class weighting;
- multiclass and binary evaluation;
- checkpoint, JSON, manifest, and accuracy-plot output;
- CLIP, XCLIP, and XCLIP vision experiments;
- MP4 and AVI video discovery and decoding;
- a unified `vlm` CLI;
- compatibility wrappers for every existing executable script and public
  import path.

No existing feature, model, experiment, public callable, script, command-line
argument, output filename, or output format will be intentionally removed.
Existing implementation mistakes are not compatibility requirements; they will
be corrected while preserving the intended behavior.

## Explicit Constraints

- Do not commit or push.
- Do not remove or overwrite the user's uncommitted work.
- Fetch and integrate the latest `origin/main` before restructuring.
- Resolve integration conflicts by retaining both local and remote behavior.
- Do not add a persistent automated test suite or test files.
- Verification may use syntax checks, imports, CLI help, and small temporary
  smoke executions that do not remain in the repository.
- Avoid downloading models or running full training solely for verification.

## Architecture

```text
src/
└── vlm/
    ├── __init__.py
    ├── __main__.py
    ├── cli.py
    ├── config.py
    ├── data/
    │   ├── __init__.py
    │   ├── discovery.py
    │   ├── splitting.py
    │   └── video_io.py
    ├── embeddings/
    │   ├── __init__.py
    │   ├── augmentation.py
    │   ├── cache.py
    │   ├── dataset.py
    │   └── extractor.py
    ├── models/
    │   ├── __init__.py
    │   ├── gru.py
    │   └── prototype.py
    ├── training/
    │   ├── __init__.py
    │   ├── engine.py
    │   ├── multiclass.py
    │   ├── prototype.py
    │   └── binary.py
    ├── evaluation/
    │   ├── __init__.py
    │   ├── multiclass.py
    │   └── binary.py
    ├── reporting/
    │   ├── __init__.py
    │   ├── artifacts.py
    │   └── plots.py
    └── experiments/
        ├── __init__.py
        ├── clip_video.py
        ├── xclip.py
        └── xclip_vision.py

archive/
├── README.md
└── legacy_scripts/
    ├── main.py
    ├── train_video_classifier.py
    ├── video_embeddings_classifier.py
    ├── train_binary_video_classifier.py
    ├── video_dataset.py
    ├── video_embedding_cache.py
    ├── clipModelTest.py
    ├── XclipModelTest.py
    └── XCVisionTest.py

PROJECT_STRUCTURE.md
```

The `src` layout keeps the repository root distinct from the installable
`vlm` Python package and prevents accidental imports from the working directory.

Classes are used where state and a stable interface are useful:

- `GRUVideoClassifier`;
- `PrototypeVideoClassifier`;
- configuration dataclasses;
- embedding-cache services when they own cache configuration and paths.

Pure functions are used for stateless behavior:

- video discovery and class-ID parsing;
- split construction;
- frame sampling and sequence resizing;
- metric calculation;
- plotting and artifact serialization.

## Unified CLI

The package exposes both a console command and a module entry point:

```bash
vlm ...
python -m vlm ...
```

Primary commands:

```bash
vlm train multiclass
vlm train prototype
vlm train binary

vlm embeddings extract --pipeline multiclass
vlm embeddings extract --pipeline prototype
vlm embeddings extract --pipeline binary

vlm experiment clip
vlm experiment xclip
vlm experiment xclip-vision
```

Each training command owns only pipeline-specific arguments. Shared training,
embedding, device, and artifact arguments are defined once and reused.

`--extract-only` remains supported by legacy wrappers and may also remain
available on the corresponding unified training command for compatibility.

## Preserved Models and Experiments

### Multiclass GRU

`GRUVideoClassifier` consumes cached per-frame CLIP embeddings, applies a
bidirectional GRU, performs masked temporal mean/max pooling, and produces class
logits through its classifier head.

### Prototype classifier

`PrototypeVideoClassifier` uses the same bidirectional GRU and masked temporal
pooling, normalizes the pooled video representation, and compares it against
learnable normalized class prototypes with a learnable logit scale.

The class was accidentally removed by an unfinished local cache extraction
refactor while its call site remained. Its committed implementation has been
restored and must remain part of the new package and legacy exports.

### Binary classifier

The safe/danger pipeline retains danger-class mapping, stratified
train/validation/test splitting, minority-class frame augmentation, optional
class weights, checkpoint selection, and binary metrics.

### Experiments

The following remain independently executable:

- per-frame CLIP image/text similarity;
- XCLIP video/text similarity;
- XCLIP vision feature extraction.

The default pretrained model identifiers remain:

- `openai/clip-vit-base-patch32`;
- `microsoft/xclip-base-patch32`.

## Data Flow

```text
CLI or compatibility script
→ parse arguments
→ construct typed pipeline configuration
→ discover videos and construct split
→ write split manifest
→ resolve or generate CLIP embedding cache
→ create cached embedding datasets and loaders
→ construct selected model
→ run shared training engine
→ run pipeline-specific evaluation
→ save checkpoint, JSON results, manifests, and plots
```

The three training pipelines share infrastructure but retain independent model
construction, split rules, evaluation rules, and result schemas.

## Compatibility Layer

The following files remain:

- `main.py`;
- `train_video_classifier.py`;
- `video_embeddings_classifier.py`;
- `train_binary_video_classifier.py`;
- `video_dataset.py`;
- `video_embedding_cache.py`;
- `clipModelTest.py`;
- `XclipModelTest.py`;
- `XCVisionTest.py`.

They become thin wrappers and re-export the public names that existing Python
callers may import. For example:

```python
from train_video_classifier import GRUVideoClassifier, train_epoch
from video_embeddings_classifier import PrototypeVideoClassifier
from video_embedding_cache import extract_or_load_embeddings
```

Existing direct script execution continues to work. The wrappers translate
legacy arguments to the unified package without emitting deprecation warnings
or maintaining separate copies of pipeline logic.

## Archive Strategy

The directory is named `archive/` using the standard English spelling.

After local changes and the latest `origin/main` are integrated, the complete
pre-restructure standalone implementations are copied to
`archive/legacy_scripts/` for historical reference. The root files are then
replaced with minimal compatibility wrappers. Archived source is inert:

- the active `vlm` package must never import from `archive/`;
- compatibility wrappers must never execute archived implementations;
- archived files are not an alternate implementation to maintain;
- `archive/README.md` explains that Git history and the new package are the
  authoritative sources.

Top-level legacy filenames remain as small wrappers because moving them outright
would break existing commands and Python imports. This preserves calls such as:

```bash
python train_video_classifier.py
```

and:

```python
from train_video_classifier import GRUVideoClassifier
```

while removing the old full implementations from the active project surface.

## Living Structure and Command Guide

The repository root contains `PROJECT_STRUCTURE.md`. It is a maintained guide,
not a one-time migration note. It records:

- the current active directory tree;
- the purpose of every top-level package and compatibility wrapper;
- unified `vlm` command examples;
- mappings from every old script command to its unified replacement;
- model and experiment availability;
- artifact and cache locations.

It must be updated in the same change whenever a command, module boundary,
wrapper, or output location changes.

The root `README.md` remains in place as the project landing page. It summarizes
the project, supported models and experiments, primary unified commands,
compatibility policy, and links to `PROJECT_STRUCTURE.md`. It must reflect the
implemented state and must not advertise commands as available before they
exist.

## Cache Compatibility

Cache identity includes only inputs that affect embedding tensors, such as:

- model identifier;
- video path;
- frame stride;
- maximum sequence length;
- augmentation identity and deterministic seed.

Labels and split membership do not unnecessarily invalidate embeddings.
Compatible existing tensors are reused. Missing or incompatible tensors are
regenerated. Existing cache paths and manifest formats remain readable.

The inconsistent `base_seed` call in the unfinished local refactor will be
reconciled with the actual shared augmentation-seed interface.

## Error Handling

- Missing dataset directories produce a clear path-specific error.
- Empty class selections or insufficient samples produce actionable split
  errors.
- MP4 and AVI are both discovered through one supported-extension definition.
- Unsupported filename patterns identify the offending video.
- Video decoding errors include the source video path.
- Empty frame samples and malformed embedding tensors fail near their source.
- Cache incompatibility triggers regeneration rather than silent reuse.
- Model-loading failures retain their original exception and identify the model
  name.
- Invalid CLI combinations are rejected by `argparse` before a pipeline starts.

## Output Compatibility

Existing output locations and schemas remain available, including:

- selected split manifests;
- embedding manifests and cached tensors;
- multiclass and binary checkpoints;
- evaluation JSON;
- accuracy graphs.

The new package may centralize artifact-writing code, but compatibility wrappers
must preserve each legacy script's defaults and filenames.

## Verification Without Test Files

The repository will not gain a persistent test suite. Verification consists of:

- parsing every Python source file;
- importing the new package and all compatibility modules without unintended
  execution;
- checking unified and legacy `--help` output;
- constructing each model with small synthetic tensors and checking forward
  output shape and finite values;
- exercising pure split, metric, and cache-key functions with temporary data;
- confirming Git diff contains no unintended deletion of models, experiments,
  arguments, or compatibility exports.

Temporary verification files, if required, must be created outside the
repository or removed before handoff.

## Git Integration

Before restructuring:

1. inspect the working tree and latest remote refs;
2. fetch the latest `origin`;
3. preserve local uncommitted changes;
4. integrate `origin/main`;
5. resolve conflicts by retaining both local changes and remote
   MP4/AVI-compatibility behavior;
6. verify that no local work disappeared.

No commit, push, history rewrite, or destructive reset is allowed.

## Completion Criteria

The restructure is complete when:

- one unified `vlm` CLI exposes all training, extraction, and experiment flows;
- all legacy scripts remain executable;
- all legacy public imports remain resolvable;
- all models and experiments are present;
- local and latest remote functionality are integrated;
- duplicate pipeline, cache, evaluation, and plotting implementations have
  been consolidated;
- pre-restructure full implementations are historical-only under `archive/`
  while top-level compatibility wrappers remain active;
- `PROJECT_STRUCTURE.md` accurately documents the final tree and every unified
  and legacy command;
- verification checks pass without adding a persistent test suite;
- no commit or push has occurred.
