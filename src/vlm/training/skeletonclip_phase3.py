"""Phase 3 Gate 0: reproducible RGB vision-LoRA compatibility audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections.abc import Sequence
from pathlib import Path

import torch
from transformers import XCLIPModel, XCLIPProcessor
from transformers import __version__ as transformers_version

from vlm.models.lora import lora_state_dict, load_lora_state_dict, merge_lora, merged_base_state_dict, unmerge_lora
from vlm.models.xclip_vision_lora import XCLIPVisionClassifier, XCLIPVisionLoRAClassifier
from vlm.reporting import write_json
from vlm.skeleton_teacher_cache import sha256_file
from vlm.training.engine import set_seed
from vlm.xclip_features import FRAME_SAMPLING_STRATEGY, read_uniform_video_frames


def _hash_json(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _make_model(args: argparse.Namespace, device: torch.device) -> XCLIPVisionLoRAClassifier:
    base = XCLIPModel.from_pretrained(args.model_name, revision=args.model_revision, local_files_only=True)
    if int(base.config.vision_config.num_frames) != args.frames:
        raise ValueError("Frame count differs from the pretrained XCLIP contract")
    return XCLIPVisionLoRAClassifier(
        base, num_classes=120, head_dropout=args.head_dropout,
        upper_blocks=args.upper_blocks, lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
    ).to(device)


def run_gate0(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 3 Gate 0 requires a visible CUDA GPU")
    if args.batch_size < 1 or args.frames != 8:
        raise ValueError("Gate 0 requires positive batch size and the locked eight-frame input")
    device = torch.device("cuda:0")
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    processor = XCLIPProcessor.from_pretrained(args.model_name, revision=args.model_revision, local_files_only=True)
    split_manifest = Path(args.split_manifest)
    split_payload = json.loads(split_manifest.read_text(encoding="utf-8"))
    if split_payload.get("frame_sampling_strategy") != FRAME_SAMPLING_STRATEGY or split_payload.get("max_seq_len") != args.frames:
        raise ValueError("Canonical split sampling contract differs from Gate 0")
    rows = split_payload.get("train", [])
    sources = [Path(row["video_path"]) for row in rows[:args.batch_size]]
    if len(sources) != args.batch_size or len(set(sources)) != args.batch_size:
        raise ValueError("Gate 0 needs distinct canonical train videos")
    frames_by_video = [read_uniform_video_frames(source, args.frames) for source in sources]
    pixels = processor(images=[list(frames) for frames in frames_by_video], return_tensors="pt")["pixel_values"].to(device)
    if tuple(pixels.shape) != (args.batch_size, 8, 3, 224, 224):
        raise RuntimeError(f"Unexpected processor shape: {tuple(pixels.shape)}")
    torch.cuda.reset_peak_memory_stats(device)
    model = _make_model(args, device)
    audit = model.trainable_parameter_audit().as_dict()
    if not all(name.endswith(("lora_a", "lora_b")) or name.startswith("classifier.") for name in audit["names"]):
        raise RuntimeError("Trainable parameter whitelist failed")
    if not any(name.startswith("vision_model.") for name in audit["names"]):
        raise RuntimeError("Vision LoRA was not injected")
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    started = time.perf_counter()
    logits = model(pixels)
    loss = torch.nn.functional.cross_entropy(logits.float(), torch.zeros(args.batch_size, dtype=torch.long, device=device))
    loss.backward()
    torch.cuda.synchronize(device)
    step_seconds = time.perf_counter() - started
    vision_grad = any(p.grad is not None and bool(p.grad.abs().sum() > 0) for name, p in model.named_parameters() if name.startswith("vision_model.") and p.requires_grad)
    frozen_grad = [name for name, p in model.named_parameters() if not p.requires_grad and p.grad is not None]
    if not torch.isfinite(logits).all() or not torch.isfinite(loss) or not vision_grad or frozen_grad:
        raise RuntimeError("Forward, loss, vision gradient, or frozen-base audit failed")
    optimizer.step()
    model.eval()
    with torch.inference_mode():
        reference = model(pixels).detach().cpu()
        merge_lora(model)
        merged = model(pixels).detach().cpu()
    merge_error = float((reference - merged).abs().max())
    if merge_error > 1e-5:
        raise RuntimeError(f"LoRA merge parity failed: {merge_error}")
    output = Path(args.artifacts_dir)
    output.mkdir(parents=True, exist_ok=True)
    unmerge_lora(model)
    adapter_path = output / "adapter_smoke.pt"
    torch.save({"lora_state_dict": lora_state_dict(model), "classifier_state_dict": model.classifier.state_dict(), "target_modules": model.target_modules}, adapter_path)
    merge_lora(model)
    merged_path = output / "merged_smoke.pt"
    torch.save({"state_dict": merged_base_state_dict(model), "model_revision": args.model_revision}, merged_path)
    peak_bytes = torch.cuda.max_memory_allocated(device)
    targets = model.target_modules
    del optimizer, model
    torch.cuda.empty_cache()

    adapter_payload = torch.load(adapter_path, map_location="cpu", weights_only=True)
    restored = _make_model(args, device)
    if adapter_payload["target_modules"] != restored.target_modules:
        raise RuntimeError("Adapter target modules differ on reload")
    load_lora_state_dict(restored, adapter_payload["lora_state_dict"])
    restored.classifier.load_state_dict(adapter_payload["classifier_state_dict"], strict=True)
    restored.eval()
    with torch.inference_mode():
        adapter_error = float((restored(pixels).cpu() - reference).abs().max())
    del restored
    torch.cuda.empty_cache()
    merged_payload = torch.load(merged_path, map_location="cpu", weights_only=True)
    clean_base = XCLIPModel.from_pretrained(args.model_name, revision=args.model_revision, local_files_only=True)
    clean_model = XCLIPVisionClassifier(clean_base, num_classes=120, head_dropout=args.head_dropout).to(device)
    clean_model.load_state_dict(merged_payload["state_dict"], strict=True)
    clean_model.eval()
    with torch.inference_mode():
        merged_reload_error = float((clean_model(pixels).cpu() - reference).abs().max())
    if adapter_error > 1e-6 or merged_reload_error > 1e-5:
        raise RuntimeError(f"Checkpoint reload parity failed: adapter={adapter_error}, merged={merged_reload_error}")
    config = {
        "model_name": args.model_name, "model_revision": args.model_revision,
        "frames": args.frames, "sampling": FRAME_SAMPLING_STRATEGY,
        "image_processor_sha256": _hash_json(processor.image_processor.to_dict()),
        "upper_blocks": args.upper_blocks, "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha, "lora_dropout": args.lora_dropout,
        "head_dropout": args.head_dropout, "batch_size": args.batch_size, "seed": args.seed,
        "split_manifest": str(split_manifest), "split_manifest_sha256": sha256_file(split_manifest),
        "sample_videos": [str(source) for source in sources],
        "sample_video_sha256": [sha256_file(source) for source in sources],
    }
    report = {
        "status": "passed", "gate": "phase3_gate0", "config": config,
        "config_sha256": _hash_json(config), "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__, "transformers_version": transformers_version,
        "python_version": platform.python_version(), "processor_shape": list(pixels.shape),
        "logit_shape": list(reference.shape), "trainable_parameter_audit": audit,
        "target_modules": targets, "vision_gradient_nonzero": vision_grad,
        "frozen_base_gradient_count": len(frozen_grad), "step_seconds": step_seconds,
        "model_step_examples_per_second": args.batch_size / step_seconds,
        "peak_memory_allocated_bytes": peak_bytes,
        "merge_max_abs_diff": merge_error, "adapter_reload_max_abs_diff": adapter_error,
        "merged_reload_max_abs_diff": merged_reload_error,
        "adapter_checkpoint": str(adapter_path), "adapter_sha256": sha256_file(adapter_path),
        "merged_checkpoint": str(merged_path), "merged_sha256": sha256_file(merged_path),
        "test_evaluated": False,
    }
    write_json(output / "experiment_lock.json", {"gate": "phase3_gate0", "config": config, "config_sha256": report["config_sha256"], "test_evaluated": False})
    write_json(output / "gate0_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 Gate 0 on one canonical train RGB clip")
    parser.add_argument("--model-name", default="microsoft/xclip-base-patch32")
    parser.add_argument("--model-revision", default="a2e27a78a2b5d802e894b8a1ef14f3a8ce490963")
    parser.add_argument("--split-manifest", default="artifacts/xclip/selected_split.json")
    parser.add_argument("--artifacts-dir", default="artifacts/skeletonclip_phase3/gate0")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--upper-blocks", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--head-dropout", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    print(json.dumps(run_gate0(parser.parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
