"""Shared engine and task-specific training pipelines."""

from .engine import resolve_device, set_seed, train_epoch

__all__ = ["resolve_device", "set_seed", "train_epoch"]
