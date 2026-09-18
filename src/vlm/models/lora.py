"""Small project-owned LoRA implementation for XCLIP MIT attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Wrap a frozen linear layer with a mergeable low-rank update."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be at least 1")
        self.base = base
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(
            torch.empty(
                rank,
                base.in_features,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        self.lora_b = nn.Parameter(
            torch.zeros(
                base.out_features,
                rank,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        self.base.requires_grad_(False)
        self.merged = False

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_b @ self.lora_a) * self.scaling

    def merge(self) -> None:
        if self.merged:
            return
        with torch.no_grad():
            self.base.weight.add_(self.delta_weight())
        self.merged = True

    def unmerge(self) -> None:
        if not self.merged:
            return
        with torch.no_grad():
            self.base.weight.sub_(self.delta_weight())
        self.merged = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.base(inputs)
        if self.merged:
            return output
        update = nn.functional.linear(self.dropout(inputs), self.lora_a)
        update = nn.functional.linear(update, self.lora_b)
        return output + update * self.scaling


@dataclass(frozen=True)
class TrainableParameterAudit:
    names: list[str]
    trainable_parameters: int
    total_parameters: int

    @property
    def trainable_ratio(self) -> float:
        if self.total_parameters == 0:
            return 0.0
        return self.trainable_parameters / self.total_parameters

    def as_dict(self) -> dict[str, object]:
        return {
            "names": self.names,
            "trainable_parameters": self.trainable_parameters,
            "total_parameters": self.total_parameters,
            "trainable_ratio": self.trainable_ratio,
        }


def inject_mit_attention_lora(
    mit: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
) -> list[str]:
    """Inject LoRA into every MIT self-attention q/v projection."""
    encoder = getattr(mit, "encoder", None)
    layers = getattr(encoder, "layers", None)
    if layers is None or len(layers) == 0:
        raise ValueError("XCLIP MIT encoder layers were not found")
    targets: list[str] = []
    for layer_index, layer in enumerate(layers):
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            raise ValueError(
                f"MIT layer {layer_index} has no self_attn module"
            )
        for projection_name in ("q_proj", "v_proj"):
            projection = getattr(attention, projection_name, None)
            if not isinstance(projection, nn.Linear):
                raise TypeError(
                    f"MIT layer {layer_index} {projection_name} is not "
                    "an nn.Linear"
                )
            setattr(
                attention,
                projection_name,
                LoRALinear(
                    projection,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                ),
            )
            targets.append(
                f"mit.encoder.layers.{layer_index}.self_attn."
                f"{projection_name}"
            )
    expected = len(layers) * 2
    if len(targets) != expected:
        raise RuntimeError(
            f"Injected {len(targets)} MIT LoRA targets; expected {expected}"
        )
    return targets


def inject_vision_attention_lora(
    vision_model: nn.Module,
    *,
    upper_blocks: int,
    rank: int,
    alpha: float,
    dropout: float,
) -> list[str]:
    """Inject q/v LoRA into the final vision Transformer blocks only."""
    encoder = getattr(vision_model, "encoder", None)
    layers = getattr(encoder, "layers", None)
    if layers is None or not 0 <= upper_blocks <= len(layers):
        raise ValueError("upper_blocks must select existing vision encoder layers")
    targets: list[str] = []
    for index in range(len(layers) - upper_blocks, len(layers)):
        attention = getattr(layers[index], "self_attn", None)
        if attention is None:
            raise ValueError(f"Vision layer {index} has no self_attn module")
        for name in ("q_proj", "v_proj"):
            projection = getattr(attention, name, None)
            if not isinstance(projection, nn.Linear):
                raise TypeError(f"Vision layer {index} {name} is not nn.Linear")
            setattr(attention, name, LoRALinear(projection, rank=rank, alpha=alpha, dropout=dropout))
            targets.append(f"vision_model.encoder.layers.{index}.self_attn.{name}")
    return targets


def iter_lora_modules(module: nn.Module):
    for child in module.modules():
        if isinstance(child, LoRALinear):
            yield child


def merge_lora(module: nn.Module) -> None:
    for lora in iter_lora_modules(module):
        lora.merge()


def unmerge_lora(module: nn.Module) -> None:
    for lora in iter_lora_modules(module):
        lora.unmerge()


def lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in module.state_dict().items()
        if name.endswith("lora_a") or name.endswith("lora_b")
    }


def load_lora_state_dict(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> None:
    expected = set(lora_state_dict(module))
    received = set(state_dict)
    if received != expected:
        missing = sorted(expected - received)
        unexpected = sorted(received - expected)
        raise ValueError(
            "LoRA checkpoint keys do not match the injected targets; "
            f"missing={missing}, unexpected={unexpected}"
        )
    parameters = dict(module.named_parameters())
    with torch.no_grad():
        for name, value in state_dict.items():
            parameters[name].copy_(value)


def merged_base_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Return a state dict compatible with the original unwrapped module."""
    if any(not lora.merged for lora in iter_lora_modules(module)):
        raise RuntimeError("All LoRA modules must be merged before export")
    merged: dict[str, torch.Tensor] = {}
    for name, value in module.state_dict().items():
        if name.endswith("lora_a") or name.endswith("lora_b"):
            continue
        merged[name.replace(".base.", ".")] = value.detach().cpu()
    return merged


def audit_trainable_parameters(module: nn.Module) -> TrainableParameterAudit:
    named = list(module.named_parameters())
    return TrainableParameterAudit(
        names=[name for name, parameter in named if parameter.requires_grad],
        trainable_parameters=sum(
            parameter.numel()
            for _, parameter in named
            if parameter.requires_grad
        ),
        total_parameters=sum(parameter.numel() for _, parameter in named),
    )
