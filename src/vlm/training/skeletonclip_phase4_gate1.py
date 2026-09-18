"""Phase 4 Gate 1: real RGB/joint spatial-gradient and checkpoint smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import av
import numpy as np
import torch
from torch.nn import functional as F
from transformers import XCLIPModel, XCLIPProcessor

from vlm.experiments.video_sampling import uniform_frame_indices
from vlm.models.lora import load_lora_state_dict, lora_state_dict, merge_lora, merged_base_state_dict
from vlm.models.xclip_vision_lora import XCLIPSpatialLoRAClassifier, XCLIPVisionClassifier
from vlm.phase4_alignment import decode_selected
from vlm.reporting import write_json
from vlm.skeleton_teacher_cache import sha256_file
from vlm.spatial_joint_targets import build_spatial_targets, spatial_region_loss
from vlm.training.engine import set_seed


def _make_model(args: argparse.Namespace, device: torch.device) -> XCLIPSpatialLoRAClassifier:
    base = XCLIPModel.from_pretrained(args.model_name, revision=args.model_revision, local_files_only=True)
    return XCLIPSpatialLoRAClassifier(
        base, num_classes=120, head_dropout=0.25, feature_noise_std=0.01,
        upper_blocks=2, lora_rank=16, lora_alpha=32.0, lora_dropout=0.05,
    ).to(device)


def _grad_names(model: XCLIPSpatialLoRAClassifier) -> list[str]:
    return [name for name, parameter in model.named_parameters()
            if parameter.grad is not None and bool(parameter.grad.detach().abs().sum() > 0)]


def run(args: argparse.Namespace) -> dict[str, object]:
    gate0 = json.loads(Path(args.gate0_decision).read_text())
    if gate0.get("status") != "pass" or gate0.get("test_evaluated") is not False:
        raise ValueError("Phase 4 Gate 0 must pass without test evaluation")
    paired_path = Path(args.paired_manifest)
    paired = json.loads(paired_path.read_text())
    row = next((row for row in paired["train"] if row["sample_id"] == args.sample_id), None)
    if row is None:
        raise ValueError("Gate 1 needs an exact paired train sample")
    if row["class_id"] != "A059":
        raise ValueError("Gate 1 smoke is locked to an A059 multi-person train sample")
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 8))
    set_seed(42)
    processor = XCLIPProcessor.from_pretrained(args.model_name, revision=args.model_revision, local_files_only=True)
    video_path = Path(row["rgb_path"])
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        count, width, height = stream.frames, stream.width, stream.height
        if count < 1:
            raise ValueError("Video frame count is unavailable")
        indices = uniform_frame_indices(8, count)
        frames, timing = decode_selected(container, indices)
    pixels = processor(images=list(frames), return_tensors="pt")["pixel_values"].to(device)
    targets = build_spatial_targets(Path(row["skeleton_path"]), indices.tolist(), source_width=width, source_height=height)
    if pixels.shape != (1, 8, 3, 224, 224) or targets.occupancy.shape != (8, 49):
        raise RuntimeError("RGB or patch-target shape differs from the locked contract")
    if not bool(targets.frame_mask.any()):
        raise RuntimeError("Selected sample has no spatial-supervised frames")
    target = targets.occupancy.unsqueeze(0).to(device)
    frame_mask = targets.frame_mask.unsqueeze(0).to(device)
    lock = {
        "schema_version": 1, "gate": "phase4_gate1", "seed": 42,
        "model_name": args.model_name, "model_revision": args.model_revision,
        "sample_id": args.sample_id, "paired_manifest_sha256": sha256_file(paired_path),
        "gate0_decision_sha256": sha256_file(Path(args.gate0_decision)),
        "video_path": str(video_path), "skeleton_path": row["skeleton_path"],
        "frame_indices": indices.tolist(), "decoded_timing": timing,
        "processor_sha256": hashlib.sha256(json.dumps(processor.image_processor.to_dict(), sort_keys=True).encode()).hexdigest(),
        "target": "tracked-state-2 color joints; top two body IDs; 224x224 patch32 occupancy; at least 10 in-crop joints per supervised frame",
        "loss": "per-frame balanced BCE with logits, positive weight clipped to [1,16], masked-frame mean",
        "architecture": "upper two vision q/v LoRA, MIT q/v LoRA, 120-class head, training-only linear patch probe",
        "spatial_probe_lr": 1e-4, "vision_lora_lr": 1e-4,
        "test_evaluated": False,
    }
    output = Path(args.artifacts_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "experiment_lock.json", lock)

    model = _make_model(args, device)
    trainable = model.trainable_parameter_audit().as_dict()
    unexpected = [name for name in trainable["names"] if not (
        name.endswith(("lora_a", "lora_b")) or name.startswith(("classifier.", "spatial_head.")))]
    if unexpected:
        raise RuntimeError(f"Unexpected trainable parameters: {unexpected}")
    if len([name for name in model.target_modules if name.startswith("vision_model.")]) != 4:
        raise RuntimeError("Expected q/v LoRA in the final two vision blocks")
    model.eval()
    with torch.inference_mode():
        initial_eval_logits, initial_eval_tokens = model(pixels, return_patch_tokens=True)
        initial_eval_logits = initial_eval_logits.detach().clone()
        initial_eval_tokens = initial_eval_tokens.detach().clone()
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    logits, patch_logits, patch_tokens = model(pixels, return_spatial=True, return_patch_tokens=True)
    if logits.shape != (1, 120) or patch_logits.shape != (1, 8, 49) or patch_tokens.shape[:3] != (1, 8, 49):
        raise RuntimeError("Unexpected model output or patch-token shape")
    spatial_loss = spatial_region_loss(patch_logits, target, frame_mask)
    if not torch.isfinite(logits).all() or not torch.isfinite(patch_logits).all() or not torch.isfinite(spatial_loss):
        raise RuntimeError("Non-finite forward or spatial loss")
    patch_token_shape = list(patch_tokens.shape)
    patch_logit_shape = list(patch_logits.shape)
    spatial_loss_value = float(spatial_loss.detach())
    spatial_loss.backward()
    spatial_grad = _grad_names(model)
    if not any(name.startswith("vision_model.") for name in spatial_grad):
        raise RuntimeError("Spatial loss did not reach vision LoRA")
    if not any(name.startswith("spatial_head.") for name in spatial_grad):
        raise RuntimeError("Spatial probe has no gradient")
    if any(name.startswith(("mit.", "classifier.")) for name in spatial_grad):
        raise RuntimeError("Spatial-only loss leaked into MIT/classifier")
    frozen_with_grad = [name for name, parameter in model.named_parameters() if not parameter.requires_grad and parameter.grad is not None]
    if frozen_with_grad:
        raise RuntimeError(f"Frozen base received gradients: {frozen_with_grad}")
    del logits, patch_logits, patch_tokens, spatial_loss
    before_vision = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                     if name.startswith("vision_model.") and parameter.requires_grad}
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if n.startswith("vision_model.") and p.requires_grad], "lr": 1e-4},
        {"params": list(model.spatial_head.parameters()), "lr": 1e-4},
    ], weight_decay=0.0)
    optimizer.step()
    changed_vision = [name for name, parameter in model.named_parameters()
                      if name in before_vision and not torch.equal(parameter.detach(), before_vision[name])]
    if not changed_vision:
        raise RuntimeError("Spatial optimizer step did not update vision LoRA")
    model.eval()
    with torch.inference_mode():
        changed_logits, _, changed_tokens = model(pixels, return_spatial=True, return_patch_tokens=True)
    token_delta = float((changed_tokens - initial_eval_tokens).abs().max())
    logit_delta = float((changed_logits - initial_eval_logits).abs().max())
    if token_delta <= 0:
        raise RuntimeError("Spatial step did not alter the vision patch representation")
    if logit_delta <= 0:
        raise RuntimeError("Spatial step did not alter RGB classification logits")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    spatial_step_seconds = time.perf_counter() - started

    model.zero_grad(set_to_none=True)
    model.train()
    ce_logits, _unused_spatial = model(pixels, return_spatial=True)
    ce_loss = F.cross_entropy(ce_logits.float(), torch.tensor([58], device=device), label_smoothing=0.05)
    ce_loss_value = float(ce_loss.detach())
    ce_loss.backward()
    del ce_logits, _unused_spatial, ce_loss
    ce_grad = _grad_names(model)
    if any(name.startswith("spatial_head.") for name in ce_grad):
        raise RuntimeError("CE-only control updated the spatial probe")
    if not all(any(name.startswith(prefix) for name in ce_grad) for prefix in ("vision_model.", "mit.", "classifier.")):
        raise RuntimeError("CE-only gradient path is incomplete")
    masked_zero = spatial_region_loss(torch.zeros_like(target), torch.zeros_like(target), torch.zeros_like(frame_mask))
    if float(masked_zero) != 0:
        raise RuntimeError("Unpaired sample did not mask the spatial loss")

    model.eval()
    with torch.inference_mode():
        reference_logits, reference_spatial = model(pixels, return_spatial=True)
        reference_logits = reference_logits.detach().cpu()
        reference_spatial = reference_spatial.detach().cpu()
    adapter = output / "adapter_smoke.pt"
    torch.save({"schema_version": 1, "lora_state_dict": lora_state_dict(model),
                "classifier_state_dict": {k: v.detach().cpu() for k, v in model.classifier.state_dict().items()},
                "spatial_head_state_dict": {k: v.detach().cpu() for k, v in model.spatial_head.state_dict().items()},
                "target_modules": model.target_modules, "model_revision": args.model_revision}, adapter)
    merge_lora(model)
    with torch.inference_mode():
        merged_logits, merged_spatial = model(pixels, return_spatial=True)
    merge_logits_diff = float((merged_logits.cpu() - reference_logits).abs().max())
    merge_spatial_diff = float((merged_spatial.cpu() - reference_spatial).abs().max())
    if max(merge_logits_diff, merge_spatial_diff) > 1e-5:
        raise RuntimeError("FP32 LoRA merge parity failed")
    merged = output / "merged_rgb_only_smoke.pt"
    rgb_state = {name: value for name, value in merged_base_state_dict(model).items() if not name.startswith("spatial_head.")}
    torch.save({"schema_version": 1, "state_dict": rgb_state, "model_revision": args.model_revision}, merged)
    peak_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    del optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    restored = _make_model(args, device)
    adapter_payload = torch.load(adapter, map_location="cpu", weights_only=True)
    if adapter_payload["target_modules"] != restored.target_modules:
        raise RuntimeError("Adapter targets differ on strict reload")
    load_lora_state_dict(restored, adapter_payload["lora_state_dict"])
    restored.classifier.load_state_dict(adapter_payload["classifier_state_dict"], strict=True)
    restored.spatial_head.load_state_dict(adapter_payload["spatial_head_state_dict"], strict=True)
    restored.eval()
    with torch.inference_mode():
        restored_logits, restored_spatial = restored(pixels, return_spatial=True)
    adapter_logits_diff = float((restored_logits.cpu() - reference_logits).abs().max())
    adapter_spatial_diff = float((restored_spatial.cpu() - reference_spatial).abs().max())
    if max(adapter_logits_diff, adapter_spatial_diff) > 1e-6:
        raise RuntimeError("Strict adapter reload parity failed")
    del restored
    if device.type == "cuda":
        torch.cuda.empty_cache()
    clean = XCLIPVisionClassifier(
        XCLIPModel.from_pretrained(args.model_name, revision=args.model_revision, local_files_only=True),
        num_classes=120, head_dropout=0.25,
    ).to(device)
    clean.load_state_dict(torch.load(merged, map_location="cpu", weights_only=True)["state_dict"], strict=True)
    clean.eval()
    with torch.inference_mode():
        rgb_only_logits = clean(pixels).cpu()
    rgb_only_diff = float((rgb_only_logits - reference_logits).abs().max())
    if rgb_only_diff > 1e-5:
        raise RuntimeError("Merged RGB-only reload parity failed")
    report = {
        "status": "pass", "gate": "phase4_gate1", "device": str(device),
        "sample_id": args.sample_id, "pixel_shape": list(pixels.shape),
        "logit_shape": list(reference_logits.shape), "patch_token_shape": patch_token_shape,
        "patch_logit_shape": patch_logit_shape, "frame_mask": targets.frame_mask.tolist(),
        "joint_valid_count": int(targets.joint_validity.sum()), "tracked_in_crop": targets.tracked_in_crop,
        "body_ids": targets.body_ids, "occupied_patch_counts": targets.occupancy.sum(-1).tolist(),
        "spatial_loss": spatial_loss_value, "ce_loss": ce_loss_value,
        "spatial_gradient_names": spatial_grad, "ce_gradient_names": ce_grad,
        "frozen_gradient_count": len(frozen_with_grad), "changed_vision_parameters": changed_vision,
        "patch_token_max_abs_change_after_spatial_step": token_delta,
        "rgb_logit_max_abs_change_after_spatial_step": logit_delta,
        "spatial_step_seconds": spatial_step_seconds, "peak_memory_allocated_bytes": peak_bytes,
        "merge_logits_max_abs_diff": merge_logits_diff, "merge_spatial_max_abs_diff": merge_spatial_diff,
        "adapter_logits_max_abs_diff": adapter_logits_diff, "adapter_spatial_max_abs_diff": adapter_spatial_diff,
        "rgb_only_reload_max_abs_diff": rgb_only_diff,
        "trainable_parameter_audit": trainable,
        "adapter_sha256": sha256_file(adapter), "rgb_only_merged_sha256": sha256_file(merged),
        "test_evaluated": False,
        "limitation": "One real paired train clip; CUDA performance was not measured when device is CPU.",
    }
    write_json(output / "gate1_report.json", report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="microsoft/xclip-base-patch32")
    parser.add_argument("--model-revision", default="a2e27a78a2b5d802e894b8a1ef14f3a8ce490963")
    parser.add_argument("--paired-manifest", default="artifacts/skeleton_data/paired_manifest.json")
    parser.add_argument("--gate0-decision", default="artifacts/skeletonclip_phase4/gate0/gate0_decision.json")
    parser.add_argument("--sample-id", default="S001C001P001R001A059")
    parser.add_argument("--artifacts-dir", default="artifacts/skeletonclip_phase4/gate1")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)
    report = run(args)
    print(json.dumps({key: report[key] for key in (
        "status", "device", "pixel_shape", "patch_token_shape", "spatial_loss",
        "patch_token_max_abs_change_after_spatial_step", "rgb_only_reload_max_abs_diff")}, indent=2))


if __name__ == "__main__":
    main()
