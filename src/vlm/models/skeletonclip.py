"""SkeletonCLIP backbone and classifier contracts for Phase 1.

The adapters preserve the official NTU input convention and expose pooled
features instead of embedding either upstream repository's training stack.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn


BackboneName = Literal["blockgcn", "hyperformer"]
BLOCKGCN_SOURCE_REVISION = "7b45971a931c5b0ba60f3a7ba11a91e544de54fb"
HYPERFORMER_SOURCE_REVISION = "c35d996bcf51df6172d33b7a32fd278ee1cfd20f"
JOINT_LABELS = (0, 4, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 2, 3, 3, 3, 2, 3, 3, 3, 1, 0, 1, 0, 1)
INWARD_EDGES = (
    (0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5),
    (7, 6), (8, 20), (9, 8), (10, 9), (11, 10), (12, 0), (13, 12),
    (14, 13), (15, 14), (16, 0), (17, 16), (18, 17), (19, 18),
    (21, 7), (22, 7), (23, 11), (24, 11),
)


def ntu_adjacency() -> Tensor:
    adjacency = torch.eye(25)
    for source, target in INWARD_EDGES:
        adjacency[source, target] = 1
        adjacency[target, source] = 1
    degree = adjacency.sum(dim=0).clamp_min(1)
    return adjacency / degree


def validate_skeleton_batch(batch: Tensor) -> None:
    if batch.ndim != 5 or batch.shape[1] != 3 or batch.shape[3:] != (25, 2):
        raise ValueError(
            "Skeleton batches must have shape [N, 3, T, 25, 2]; "
            f"received {list(batch.shape)}"
        )
    if batch.shape[2] < 1:
        raise ValueError("Skeleton batches must contain at least one frame")
    if not torch.isfinite(batch).all():
        raise ValueError("Skeleton batches must contain only finite values")


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, stride: int = 1) -> None:
        super().__init__()
        branch_channels = channels // 4
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, branch_channels, 1),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        branch_channels,
                        branch_channels,
                        (3, 1),
                        stride=(stride, 1),
                        padding=(dilation, 0),
                        dilation=(dilation, 1),
                    ),
                    nn.BatchNorm2d(branch_channels),
                )
                for dilation in (1, 2, 3)
            ]
        )
        self.branches.append(
            nn.Sequential(
                nn.Conv2d(channels, branch_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(branch_channels),
            )
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.cat([branch(inputs) for branch in self.branches], dim=1)


class BlockGraphLayer(nn.Module):
    """Grouped topology-aware spatial projection followed by temporal mixing."""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        groups = 8 if in_channels >= 8 else 1
        self.register_buffer("adjacency", ntu_adjacency())
        self.dynamic_adjacency = nn.Parameter(torch.zeros(groups, 25, 25))
        self.spatial = nn.Conv2d(in_channels, out_channels, 1, groups=groups)
        self.spatial_bn = nn.BatchNorm2d(out_channels)
        self.temporal = TemporalBlock(out_channels, stride=stride)
        if in_channels == out_channels and stride == 1:
            self.residual: nn.Module = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        batch, channels, frames, joints = inputs.shape
        groups = self.dynamic_adjacency.shape[0]
        grouped = inputs.reshape(batch, groups, channels // groups, frames, joints)
        topology = self.adjacency.unsqueeze(0) + self.dynamic_adjacency
        mixed = torch.einsum("ngctv,gvw->ngctw", grouped, topology)
        mixed = mixed.reshape(batch, channels, frames, joints)
        output = self.temporal(self.spatial_bn(self.spatial(mixed)))
        return self.activation(output + self.residual(inputs))


class BlockGCNAdapter(nn.Module):
    output_dim = 256
    source_revision = BLOCKGCN_SOURCE_REVISION

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.BatchNorm1d(2 * 3 * 25)
        self.input_projection = nn.Conv2d(3, 64, 1)
        widths = ((64, 1), (64, 1), (64, 1), (128, 2), (128, 1), (128, 1), (256, 2), (256, 1), (256, 1), (256, 1))
        layers: list[nn.Module] = []
        current = 64
        for width, stride in widths:
            layers.append(BlockGraphLayer(current, width, stride=stride))
            current = width
        self.layers = nn.Sequential(*layers)

    def forward(self, batch: Tensor) -> Tensor:
        validate_skeleton_batch(batch)
        samples, channels, frames, joints, people = batch.shape
        normalized = batch.permute(0, 4, 3, 1, 2).reshape(samples, people * joints * channels, frames)
        normalized = self.input_norm(normalized)
        normalized = normalized.reshape(samples, people, joints, channels, frames)
        normalized = normalized.permute(0, 1, 3, 4, 2).reshape(samples * people, channels, frames, joints)
        features = self.layers(self.input_projection(normalized))
        return features.mean(dim=(2, 3)).reshape(samples, people, -1).mean(dim=1)


class HypergraphLayer(nn.Module):
    def __init__(self, channels: int, heads: int, *, stride: int = 1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.group_projection = nn.Linear(channels, channels)
        self.temporal = nn.Conv2d(channels, channels, (3, 1), stride=(stride, 1), padding=(1, 0))
        self.temporal_bn = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        batch, channels, frames, joints = inputs.shape
        tokens = inputs.permute(0, 2, 3, 1).reshape(batch * frames, joints, channels)
        normalized = self.norm(tokens)
        group_count = max(JOINT_LABELS) + 1
        groups = torch.zeros(batch * frames, group_count, channels, device=inputs.device, dtype=inputs.dtype)
        labels = torch.tensor(JOINT_LABELS, device=inputs.device)
        groups.index_add_(1, labels, normalized)
        counts = torch.bincount(labels, minlength=group_count).to(inputs.dtype).view(1, -1, 1)
        context = self.group_projection(groups / counts)[..., :]
        hyper_context = context.index_select(1, labels)
        attended, _ = self.attention(normalized + hyper_context, normalized, normalized, need_weights=False)
        output = (tokens + attended).reshape(batch, frames, joints, channels).permute(0, 3, 1, 2)
        return self.activation(self.temporal_bn(self.temporal(output)))


class HyperformerAdapter(nn.Module):
    output_dim = 216
    source_revision = HYPERFORMER_SOURCE_REVISION

    def __init__(self, *, heads: int = 9) -> None:
        super().__init__()
        channels = 24 * heads
        self.input_norm = nn.BatchNorm1d(2 * 3 * 25)
        self.input_projection = nn.Conv2d(3, channels, 1)
        self.layers = nn.ModuleList(
            HypergraphLayer(channels, heads, stride=2 if index in (4, 7) else 1)
            for index in range(10)
        )

    def forward(self, batch: Tensor) -> Tensor:
        validate_skeleton_batch(batch)
        samples, channels, frames, joints, people = batch.shape
        normalized = batch.permute(0, 4, 3, 1, 2).reshape(samples, people * joints * channels, frames)
        normalized = self.input_norm(normalized)
        normalized = normalized.reshape(samples, people, joints, channels, frames)
        normalized = normalized.permute(0, 1, 3, 4, 2).reshape(samples * people, channels, frames, joints)
        features = self.input_projection(normalized)
        for layer in self.layers:
            features = layer(features)
        return features.mean(dim=(2, 3)).reshape(samples, people, -1).mean(dim=1)


def build_skeleton_backbone(name: BackboneName) -> nn.Module:
    if name == "blockgcn":
        return BlockGCNAdapter()
    if name == "hyperformer":
        return HyperformerAdapter()
    raise ValueError(f"Unsupported skeleton backbone: {name}")


class SkeletonLinearTeacher(nn.Module):
    def __init__(self, backbone_name: BackboneName, num_classes: int = 59) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = build_skeleton_backbone(backbone_name)
        self.head = nn.Sequential(
            nn.LayerNorm(self.backbone.output_dim),
            nn.Linear(self.backbone.output_dim, num_classes),
        )

    def forward(self, batch: Tensor) -> Tensor:
        return self.head(self.backbone(batch))


@dataclass(frozen=True)
class SkeletonCheckpointIdentity:
    backbone: BackboneName
    ordered_class_ids: tuple[str, ...]
    preprocessing_id: str
    source_revision: str


def save_skeleton_checkpoint(
    path: Path,
    model: SkeletonLinearTeacher,
    identity: SkeletonCheckpointIdentity,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"identity": asdict(identity), "state_dict": model.state_dict()}, path)


def load_skeleton_checkpoint_strict(
    path: Path,
    model: SkeletonLinearTeacher,
    expected_identity: SkeletonCheckpointIdentity,
) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("identity") != asdict(expected_identity):
        raise ValueError("Skeleton checkpoint identity does not match the requested model")
    model.load_state_dict(payload["state_dict"], strict=True)
