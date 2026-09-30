"""Motion target audit and decision-coupled spatial mechanism smoke (no test)."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import av
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F
from transformers import XCLIPModel, XCLIPProcessor

from vlm.experiments.video_sampling import uniform_frame_indices
from vlm.models.lora import load_lora_state_dict, lora_state_dict, merge_lora, merged_base_state_dict
from vlm.models.xclip_vision_lora import XCLIPAttentionClassifier, XCLIPAttentionLoRAClassifier
from vlm.phase4_alignment import choose_rows, decode_selected
from vlm.reporting import write_json
from vlm.skeleton_teacher_cache import sha256_file
from vlm.spatial_joint_targets import MOTION_TARGET_VERSION, build_motion_spatial_targets, spatial_attention_loss
from vlm.training.engine import set_seed


REVISION = "a2e27a78a2b5d802e894b8a1ef14f3a8ce490963"
MODEL = "microsoft/xclip-base-patch32"


def _gradients(model: torch.nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters()
            if parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)]


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(8)
    set_seed(42)
    output = args.artifacts_dir
    output.mkdir(parents=True, exist_ok=True)
    paired = json.loads(args.paired_manifest.read_text())
    alignment = json.loads(args.gate0_decision.read_text())
    if (alignment.get("status") != "pass" or alignment.get("test_evaluated") is not False
            or alignment.get("manifest_sha256") != sha256_file(args.paired_manifest)):
        raise ValueError("Alignment decision does not certify this paired manifest")
    split_hash = sha256_file(args.split_manifest)
    checkpoint = torch.load(args.init_adapter, map_location="cpu", weights_only=False)
    identity = checkpoint["identity"]
    if (identity.get("model_revision") != REVISION or identity.get("arm") != "vision"
            or identity.get("split_sha256") != split_hash or identity.get("test_evaluated") is not False):
        raise ValueError("Warm-start revision, split or test provenance changed")
    # Selection fixed before inspecting target quality; always includes interaction.
    train = choose_rows(paired["train"], 3)
    interaction = next(row for row in paired["train"] if row["class_id"] == "A059")
    if interaction not in train:
        train.append(interaction)
    validation = choose_rows(paired["val"], 4)
    rows = train + validation
    lock = {
        "schema_version": 2, "target_version": MOTION_TARGET_VERSION, "seed": 42,
        "model_revision": REVISION, "test_evaluated": False,
        "sample_ids": [row["sample_id"] for row in rows], "head_fit_steps": args.head_fit_steps,
        "head_fit_lr": 0.001, "pooling_gradient_probe_alpha": 0.1,
        "source_sha256": {"paired_manifest": sha256_file(args.paired_manifest),
                          "split": split_hash, "init_adapter": sha256_file(args.init_adapter),
                          "gate0": sha256_file(args.gate0_decision), "runner": sha256_file(Path(__file__)),
                          "targets": sha256_file(Path(__file__).parents[1] / "spatial_joint_targets.py"),
                          "model": sha256_file(Path(__file__).parents[1] / "models/xclip_vision_lora.py")},
        "samples": {row["sample_id"]: {"rgb": sha256_file(Path(row["rgb_path"])),
                                      "skeleton": sha256_file(Path(row["skeleton_path"]))} for row in rows},
        "shuffle": "fixed cyclic shift of selected paired train samples; held-out validation remains correct",
        "head_fit_scope": "frozen RGB patch tokens, head only; mechanism screen, not classification pilot",
    }
    lock_path = output / "experiment_lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise ValueError("Immutable smoke lock differs; use another artifacts directory")
    write_json(lock_path, lock)
    processor = XCLIPProcessor.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    samples, audits = [], []
    for row in rows:
        with av.open(row["rgb_path"]) as container:
            stream = container.streams.video[0]
            count, width, height = stream.frames, stream.width, stream.height
            indices = uniform_frame_indices(8, count)
            frames, _ = decode_selected(container, indices)
        target = build_motion_spatial_targets(Path(row["skeleton_path"]), indices.tolist(),
                                              source_width=width, source_height=height)
        if abs(count - row["skeleton"]["frame_count"]) > 2:
            raise ValueError("RGB/skeleton frame-count alignment changed")
        pixels = processor(images=list(frames), return_tensors="pt")["pixel_values"]
        samples.append((pixels, target))
        distribution = target.distribution
        entropy = -(distribution * distribution.clamp_min(1e-12).log()).sum(-1)
        center = torch.tensor([(r - 3) ** 2 + (c - 3) ** 2 <= 4 for r in range(7) for c in range(7)])
        audits.append({"sample_id": row["sample_id"], "split": row["split"],
                       "frames": indices.tolist(), "valid_frames": int(target.frame_mask.sum()),
                       "body_ids": target.body_ids, "entropy": entropy.tolist(),
                       "center_mass": distribution[:, center].sum(-1).tolist(),
                       "body_part_mass": target.body_part_mass.sum(-1).tolist()})
        crop = processor(images=[frames[3]], return_tensors="np", do_normalize=False, do_rescale=False)["pixel_values"][0, 0]
        image = Image.fromarray(crop.transpose(1, 2, 0).astype("uint8")).convert("RGBA")
        overlay = Image.new("RGBA", image.size)
        draw = ImageDraw.Draw(overlay)
        weights = distribution[3] / distribution[3].max().clamp_min(1e-12)
        for patch, weight in enumerate(weights):
            r, c = divmod(patch, 7)
            draw.rectangle((c * 32, r * 32, c * 32 + 31, r * 32 + 31), fill=(255, 0, 0, int(weight * 160)))
        Image.alpha_composite(image, overlay).convert("RGB").save(output / f"{row['sample_id']}_motion.png")
    write_json(output / "target_audit.json", {"target_version": MOTION_TARGET_VERSION, "samples": audits,
                                             "test_evaluated": False, "scope": "8 deterministic train/validation samples; not full-coverage audit"})
    if args.audit_only:
        return {"status": "audit_complete", "samples": len(samples), "test_evaluated": False}
    if not torch.cuda.is_available() and not args.cpu:
        return {"status": "audit_complete_pending_gpu", "test_evaluated": False}
    device = torch.device("cpu" if args.cpu else "cuda:0")
    base = XCLIPModel.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model = XCLIPAttentionLoRAClassifier(base, num_classes=120, head_dropout=0.25, upper_blocks=2,
                                        lora_rank=16, lora_alpha=32.0, lora_dropout=0.05).to(device).eval()
    source = checkpoint["model"]
    if source["target_modules"] != model.target_modules:
        raise ValueError("Warm-start target modules differ")
    load_lora_state_dict(model, source["lora_state_dict"])
    model.classifier.load_state_dict(source["classifier_state_dict"], strict=True)
    unexpected = [name for name, parameter in model.named_parameters() if parameter.requires_grad
                  and not (name.endswith(("lora_a", "lora_b")) or name.startswith(("classifier.", "spatial_head."))
                           or name == "pooling_alpha")]
    if unexpected:
        raise RuntimeError(f"Unexpected trainable parameters: {unexpected}")
    pixels, target = samples[0]
    pixels = pixels.to(device)
    with torch.no_grad():
        vision = model.vision_model(pixel_values=pixels.flatten(0, 1), return_dict=True)
        projected = model.visual_projection(vision.pooler_output).reshape(1, 8, -1)
        original = model.classifier(model.mit(projected, return_dict=True).pooler_output)
        parity = float((model(pixels) - original).abs().max())
    if parity != 0:
        raise RuntimeError(f"Identity parity failed: {parity}")
    del vision, projected, original
    logits, attention = model(pixels, return_spatial=True)
    distribution = target.distribution.unsqueeze(0).to(device)
    mask = target.frame_mask.unsqueeze(0).to(device)
    spatial = spatial_attention_loss(attention, distribution, mask)
    spatial.backward()
    spatial_grads = _gradients(model)
    if (not any(name.startswith("vision_model.") for name in spatial_grads)
            or not any(name.startswith("spatial_head.") for name in spatial_grads)
            or any(name.startswith(("mit.", "classifier.")) or name == "pooling_alpha" for name in spatial_grads)):
        raise RuntimeError("Spatial gradient routing failed")
    model.zero_grad(set_to_none=True)
    del logits, attention, spatial
    with torch.no_grad():
        model.pooling_alpha.fill_(0.1)
    logits = model(pixels)
    F.cross_entropy(logits, torch.tensor([int(train[0]["class_id"][1:]) - 1], device=device)).backward()
    ce_grads = _gradients(model)
    if not all(any(name.startswith(prefix) for name in ce_grads)
               for prefix in ("spatial_head.", "vision_model.", "mit.", "classifier.")):
        raise RuntimeError("CE attention coupling failed")
    model.zero_grad(set_to_none=True)
    del logits
    zero_logits = torch.randn_like(distribution, requires_grad=True)
    invalid = spatial_attention_loss(zero_logits, torch.zeros_like(distribution), torch.zeros_like(mask))
    invalid.backward()
    if invalid.item() != 0 or zero_logits.grad.abs().sum().item() != 0:
        raise RuntimeError("All-invalid target masking failed")
    tokens = []
    with torch.no_grad():
        for rgb, _ in samples:
            _, patches = model(rgb.to(device), return_patch_tokens=True)
            tokens.append(patches.detach())
    train_tokens = torch.cat(tokens[:len(train)])
    targets = torch.stack([target.distribution for _, target in samples[:len(train)]]).to(device)
    masks = torch.stack([target.frame_mask for _, target in samples[:len(train)]]).to(device)
    val_tokens = torch.cat(tokens[len(train):])
    val_targets = torch.stack([target.distribution for _, target in samples[len(train):]]).to(device)
    val_masks = torch.stack([target.frame_mask for _, target in samples[len(train):]]).to(device)
    head_state = copy.deepcopy(model.spatial_head.state_dict())
    fits = {}
    for arm in ("correct", "shuffled"):
        head = copy.deepcopy(model.spatial_head)
        head.load_state_dict(head_state)
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0)
        arm_target = targets if arm == "correct" else targets.roll(1, 0)
        arm_mask = masks if arm == "correct" else masks.roll(1, 0)
        losses = []
        for _ in range(args.head_fit_steps):
            optimizer.zero_grad(set_to_none=True)
            loss = spatial_attention_loss(head(train_tokens).squeeze(-1), arm_target, arm_mask)
            losses.append(float(loss.detach()))
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            heldout = float(spatial_attention_loss(head(val_tokens).squeeze(-1), val_targets, val_masks))
        fits[arm] = {"train_losses": losses, "heldout_attention_ce": heldout}
    with torch.no_grad():
        reference_logits, reference_attention = model(pixels, return_spatial=True)
    adapter = {"lora": lora_state_dict(model), "classifier": model.classifier.state_dict(),
               "spatial_head": model.spatial_head.state_dict(), "pooling_alpha": model.pooling_alpha.detach().cpu(),
               "target_version": MOTION_TARGET_VERSION, "provenance": lock}
    torch.save(adapter, output / "adapter_smoke.pt")
    load_lora_state_dict(model, adapter["lora"])
    model.classifier.load_state_dict(adapter["classifier"], strict=True)
    model.spatial_head.load_state_dict(adapter["spatial_head"], strict=True)
    merge_lora(model)
    state = merged_base_state_dict(model)
    torch.save({"state_dict": state, "target_version": MOTION_TARGET_VERSION, "provenance": lock}, output / "merged_rgb_only_smoke.pt")
    del model, base, checkpoint
    torch.cuda.empty_cache()
    fresh = XCLIPModel.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    restored = XCLIPAttentionClassifier(fresh, num_classes=120, head_dropout=0.25).to(device).eval()
    restored.load_state_dict(state, strict=True)
    with torch.no_grad():
        reloaded_logits, reloaded_attention = restored(pixels, return_spatial=True)
    reload_diff = float((reloaded_logits - reference_logits).abs().max())
    attention_diff = float((reloaded_attention - reference_attention).abs().max())
    if max(reload_diff, attention_diff) > 1e-4:
        raise RuntimeError("RGB-only merged strict reload parity failed")
    correct_faster = (fits["correct"]["train_losses"][0] - fits["correct"]["train_losses"][-1]
                      > fits["shuffled"]["train_losses"][0] - fits["shuffled"]["train_losses"][-1])
    report = {"status": "pass" if correct_faster else "mechanism_no_go", "test_evaluated": False,
              "identity_max_abs_diff": parity, "rgb_only_reload_max_abs_diff": reload_diff,
              "attention_reload_max_abs_diff": attention_diff, "spatial_gradient_names": spatial_grads,
              "ce_gradient_names": ce_grads, "all_invalid_loss": float(invalid.detach()),
              "fits": fits, "correct_target_faster": correct_faster,
              "limitation": "Head-only fit compares learnability on fixed real tokens; full joint-training smoke and causal evaluation remain required."}
    write_json(output / "mechanism_report.json", report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-manifest", type=Path, default=Path("artifacts/skeleton_data/paired_manifest.json"))
    parser.add_argument("--split-manifest", type=Path, default=Path("artifacts/xclip/selected_split.json"))
    parser.add_argument("--gate0-decision", type=Path, default=Path("artifacts/skeletonclip_phase4/gate0/gate0_decision.json"))
    parser.add_argument("--init-adapter", type=Path, default=Path("artifacts/skeletonclip_phase3/vision_batch16_workers24_uint8_seed42/vision_lora_ce_seed42/best_adapter.pt"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/spatial_attention_v2/smoke"))
    parser.add_argument("--head-fit-steps", type=int, default=40)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)
    if args.head_fit_steps < 2:
        raise ValueError("Head-fit steps must be at least two")
    result = run(args)
    print(json.dumps({key: value for key, value in result.items() if key not in ("fits", "spatial_gradient_names", "ce_gradient_names")}, indent=2))


if __name__ == "__main__":
    main()
