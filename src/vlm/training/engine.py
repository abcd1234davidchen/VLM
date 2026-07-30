"""Common training utilities shared by all classifiers."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class ValidationCheckpoint:
    """Keep an in-memory copy of the model with the best validation score."""

    epoch: int = 0
    score: tuple[float, ...] | None = None
    state_dict: dict[str, torch.Tensor] = field(default_factory=dict)

    def consider(
        self,
        model: nn.Module,
        *,
        epoch: int,
        score: float | tuple[float, ...],
    ) -> bool:
        normalized_score = score if isinstance(score, tuple) else (score,)
        if self.score is not None and normalized_score <= self.score:
            return False
        self.epoch = epoch
        self.score = normalized_score
        self.state_dict = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.state_dict().items()
        }
        return True

    def restore(self, model: nn.Module) -> None:
        if not self.state_dict:
            raise RuntimeError(
                "Training did not produce a validation checkpoint."
            )
        model.load_state_dict(self.state_dict)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(requested_device: str) -> torch.device:
    if requested_device != "auto":
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float,
    epoch: int | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    batches = (
        tqdm(
            loader,
            desc=f"Epoch {epoch} Training",
            leave=False,
        )
        if epoch is not None
        else loader
    )
    for embeddings, lengths, labels in batches:
        embeddings = embeddings.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(embeddings, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss * embeddings.size(0)
        if epoch is not None:
            batches.set_postfix(loss=f"{batch_loss:.4f}")

    return total_loss / len(loader.dataset)


def train_feature_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float,
    epoch: int | None = None,
) -> float:
    """Train a classifier over one fixed feature vector per sample."""
    model.train()
    total_loss = 0.0
    batches = (
        tqdm(
            loader,
            desc=f"Epoch {epoch} Training",
            leave=False,
        )
        if epoch is not None
        else loader
    )
    for features, labels in batches:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss * features.size(0)
        if epoch is not None:
            batches.set_postfix(loss=f"{batch_loss:.4f}")

    return total_loss / len(loader.dataset)
