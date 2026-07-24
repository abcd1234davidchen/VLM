"""Multiclass and binary evaluation."""

from .binary import (
    BINARY_CLASS_NAMES,
    binary_class_name,
    compute_metrics,
    evaluate_binary,
    safe_divide,
    validation_selection_score,
)
from .multiclass import evaluate

__all__ = [
    "BINARY_CLASS_NAMES",
    "binary_class_name",
    "compute_metrics",
    "evaluate",
    "evaluate_binary",
    "safe_divide",
    "validation_selection_score",
]
