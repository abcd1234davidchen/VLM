"""CLIP embedding extraction, deterministic augmentation, and disk caching."""

from .augmentation import (
    FrameAugmentation,
    apply_frame_augmentation,
    stable_augmentation_seed,
)
from .cache import (
    build_cache_metadata,
    cache_config_key,
    extract_or_load_embeddings,
    get_embedding_cache_path,
    is_cache_compatible,
    write_cache_manifest,
)
from .dataset import CachedEmbeddingDataset, CachedSample
from .extractor import embed_video

__all__ = [
    "CachedEmbeddingDataset",
    "CachedSample",
    "FrameAugmentation",
    "apply_frame_augmentation",
    "build_cache_metadata",
    "cache_config_key",
    "embed_video",
    "extract_or_load_embeddings",
    "get_embedding_cache_path",
    "is_cache_compatible",
    "stable_augmentation_seed",
    "write_cache_manifest",
]
