"""Compatibility wrapper for prototype-similarity video training."""

from _compat import ensure_src_path

ensure_src_path()

from vlm.data import build_split, scan_all_classes
from vlm.embeddings import (
    CachedEmbeddingDataset,
    CachedSample,
    extract_or_load_embeddings,
)
from vlm.evaluation.multiclass import evaluate
from vlm.models.prototype import PrototypeVideoClassifier
from vlm.training.engine import resolve_device, set_seed, train_epoch
from vlm.training.prototype import (
    build_parser,
    main,
    run_pipeline,
)

__all__ = [
    "CachedEmbeddingDataset",
    "CachedSample",
    "PrototypeVideoClassifier",
    "build_split",
    "build_parser",
    "evaluate",
    "extract_or_load_embeddings",
    "main",
    "resolve_device",
    "run_pipeline",
    "scan_all_classes",
    "set_seed",
    "train_epoch",
]


if __name__ == "__main__":
    main()
