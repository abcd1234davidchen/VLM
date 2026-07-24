"""Compatibility wrapper for binary safe/danger training."""

from _compat import ensure_src_path

ensure_src_path()

from vlm.data import VideoSample, parse_class_id, save_split_manifest
from vlm.embeddings import (
    CachedEmbeddingDataset,
    extract_or_load_embeddings,
)
from vlm.models.gru import GRUVideoClassifier
from vlm.training.binary import *
from vlm.training.binary import __all__ as _binary_exports
from vlm.training.engine import resolve_device, set_seed, train_epoch

__all__ = [
    *_binary_exports,
    "CachedEmbeddingDataset",
    "GRUVideoClassifier",
    "VideoSample",
    "extract_or_load_embeddings",
    "parse_class_id",
    "resolve_device",
    "save_split_manifest",
    "set_seed",
    "train_epoch",
]


if __name__ == "__main__":
    main()
