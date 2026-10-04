"""Selection and loss contracts for the fixed patch16 skeleton experiment."""

import torch
from torch.nn import functional as F


def temporal_loss(weights, target, valid):
    if weights.shape != target.shape or weights.shape[1:] != (8,) or valid.shape != weights.shape[:1]:
        raise ValueError("Expected eight-cell distributions and sample validity")
    if not torch.isfinite(weights).all() or not torch.isfinite(target).all() or (target < 0).any():
        raise ValueError("Invalid temporal distribution")
    if not torch.allclose(target.sum(-1), valid.float(), atol=1e-5, rtol=0):
        raise ValueError("Valid targets sum to one; invalid targets sum to zero")
    if not valid.any():
        return weights.sum() * 0
    return F.kl_div(weights[valid].float().clamp_min(1e-8).log(), target[valid].float(), reduction="batchmean")


def view_probabilities(logits):
    if set(logits) != {"whole", "center_zoom70"}:
        raise ValueError("Both fixed views are required")
    whole, zoom = logits["whole"], logits["center_zoom70"]
    if whole.shape != zoom.shape or whole.ndim != 2 or whole.shape[1] != 120:
        raise ValueError("Expected aligned raw120 view logits")
    if not torch.isfinite(whole).all() or not torch.isfinite(zoom).all():
        raise ValueError("Nonfinite predictions")
    probabilities = {key: value.float().softmax(-1) for key, value in logits.items()}
    probabilities["ensemble"] = (probabilities["whole"] + probabilities["center_zoom70"]) / 2
    return probabilities


def promising(report, baseline):
    return (report["correct"] - baseline["correct"] >= 26
            and report["raw_macro_f1"] >= baseline["raw_macro_f1"]
            and all(report["groups"][key]["accuracy"] >= baseline["groups"][key]["accuracy"] - .002 - 1e-12
                    for key in ("unpaired", "A060_A120"))
            and report["attention_correct"] >= baseline["attention_correct"])


def early_kill(history, baseline):
    return len(history) == 2 and all(epoch[view]["raw_accuracy"] <= baseline[view]["raw_accuracy"] - .005 + 1e-12
                                     for epoch in history for view in ("whole", "center_zoom70", "ensemble"))


def terminal_status(best, qualified, baseline, killed=False):
    if killed:
        return "no-go"
    if any(qualified.values()):
        return "promising"
    if all(best[view]["raw_accuracy"] < baseline[view]["raw_accuracy"] for view in ("whole", "ensemble")):
        return "no-go"
    return "inconclusive"
