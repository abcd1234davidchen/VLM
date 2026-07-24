"""Trainable video classifiers."""

from .gru import GRUVideoClassifier
from .prototype import PrototypeVideoClassifier

__all__ = ["GRUVideoClassifier", "PrototypeVideoClassifier"]
