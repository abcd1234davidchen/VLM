"""Trainable video classifiers."""

from .gru import GRUVideoClassifier
from .prototype import PrototypeVideoClassifier
from .xclip import XCLIPFeatureClassifier
from .xclip_lora import XCLIPMITClassifier, XCLIPMITLoRAClassifier

__all__ = [
    "GRUVideoClassifier",
    "PrototypeVideoClassifier",
    "XCLIPFeatureClassifier",
    "XCLIPMITClassifier",
    "XCLIPMITLoRAClassifier",
]
