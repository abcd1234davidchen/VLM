"""Trainable video classifiers."""

from .gru import GRUVideoClassifier
from .prototype import PrototypeVideoClassifier
from .xclip import XCLIPFeatureClassifier
from .xclip_lora import XCLIPMITClassifier, XCLIPMITLoRAClassifier
from .xclip_vision_lora import XCLIPSpatialLoRAClassifier, XCLIPVisionLoRAClassifier

__all__ = [
    "GRUVideoClassifier",
    "PrototypeVideoClassifier",
    "XCLIPFeatureClassifier",
    "XCLIPMITClassifier",
    "XCLIPMITLoRAClassifier",
    "XCLIPVisionLoRAClassifier",
    "XCLIPSpatialLoRAClassifier",
]
