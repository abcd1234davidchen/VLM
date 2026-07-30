"""Trainable video classifiers."""

from .gru import GRUVideoClassifier
from .prototype import PrototypeVideoClassifier
from .xclip import XCLIPFeatureClassifier

__all__ = [
    "GRUVideoClassifier",
    "PrototypeVideoClassifier",
    "XCLIPFeatureClassifier",
]
