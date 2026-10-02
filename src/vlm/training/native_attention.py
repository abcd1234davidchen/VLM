"""Native XCLIP attention audit, bounded GPU smoke and fixed full two-epoch pilot."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import XCLIPModel, XCLIPProcessor

from vlm.models.lora import load_lora_state_dict, lora_state_dict, merge_lora, merged_base_state_dict
from vlm.models.xclip_vision_lora import XCLIPVisionClassifier, XCLIPVisionLoRAClassifier
from vlm.native_attention_targets import TARGET_VERSION, native_pose_distribution, native_attention_loss, native_attention_metrics, native_patch_distribution
from vlm.processed_uint8_cache import ProcessedUInt8Cache
from vlm.reporting import write_json
from vlm.skeleton_teacher_cache import sha256_file
from vlm.training.engine import set_seed
from vlm.training.ski_semantic import MODEL, balanced
from vlm.training.skeletonclip_phase4_gate2 import MODEL_REVISION, Gate2Dataset, _sample_id, _worker_seed
from vlm.training.spatial_attention import classification_report, mask_patches
from vlm.xclip_features import FRAME_SAMPLING_STRATEGY

SOURCES = {
    "runner": Path(__file__),
    "model": Path(__file__).parents[1] / "models/xclip_vision_lora.py",
    "targets": Path(__file__).parents[1] / "native_attention_targets.py",
    "joint_targets": Path(__file__).parents[1] / "spatial_joint_targets.py",
    "alignment": Path(__file__).parents[1] / "phase4_alignment.py",
    "dataset": Path(__file__).with_name("skeletonclip_phase4_gate2.py"),
    "rgb_cache": Path(__file__).parents[1] / "processed_uint8_cache.py",
    "lora": Path(__file__).parents[1] / "models/lora.py",
    "sampling": Path(__file__).with_name("ski_semantic.py"),
    "reporting": Path(__file__).with_name("spatial_attention.py"),
    "dispatcher": Path(__file__).with_name("spatial_dispatch.py"),
}


def decision(current: dict, baseline: dict) -> str:
    gain = current["raw_accuracy"] - baseline["raw_accuracy"]
    if gain < 0:
        return "no_go"
    if (gain >= .002 and current["raw_macro_f1"] >= baseline["raw_macro_f1"]
            and current["groups"]["unpaired"]["accuracy"] >= baseline["groups"]["unpaired"]["accuracy"] - .002
            and current["attention_correct"] >= baseline["attention_correct"]):
        return "promising"
    return "inconclusive"


def summary(logits, labels, ids, paired_ids):
    if not torch.isfinite(logits).all() or len(ids) != len(set(ids)) or logits.shape != (len(ids), 120):
        raise ValueError("Invalid validation logits or duplicate identities")
    report = classification_report(logits, labels, ids, paired_ids)
    attention = (labels >= 41) & (labels <= 47)
    report["correct"] = int(logits.argmax(-1).eq(labels).sum())
    report["attention_correct"] = int(logits.argmax(-1).eq(labels)[attention].sum())
    report["attention_samples"] = int(attention.sum())
    return report


def changes(logits, prior, labels):
    good, old = logits.argmax(-1).eq(labels), prior.argmax(-1).eq(labels)
    return {"fixed": int((good & ~old).sum()), "regressed": int((~good & old).sum()),
            "net": int(good.sum() - old.sum()),
            "per_action": {f"A{i+1:03d}": {"fixed": int((good & ~old & (labels == i)).sum()),
                            "regressed": int((~good & old & (labels == i)).sum())} for i in range(120)}}


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; no pending run lock created")
    set_seed(args.seed)
    torch.set_num_threads(8)
    root = args.artifacts_dir
    if (root / "experiment_lock.json").exists():
        raise ValueError("Use a new directory; immutable runs cannot be overwritten")
    split = json.loads(args.split_manifest.read_text())
    paired_manifest = json.loads(args.paired_manifest.read_text())
    gate = json.loads(args.gate0_decision.read_text())
    if gate["status"] != "pass" or gate["test_evaluated"] is not False or gate["manifest_sha256"] != sha256_file(args.paired_manifest):
        raise ValueError("Alignment gate or paired manifest differs")
    if ((len(split["train"]), len(split["val"])) != (80640, 12960)
            or split["max_seq_len"] != 8 or split["frame_sampling_strategy"] != FRAME_SAMPLING_STRATEGY):
        raise ValueError("Canonical RGB split/sampling differs")
    paired = {}
    for part, count in (("train", 45236), ("val", 4180)):
        rgb_index = {_sample_id(r["video_path"]): r for r in split[part]}
        rows = paired_manifest[part]
        paired[part] = {r["sample_id"]: r for r in rows}
        if len(rgb_index) != len(split[part]) or len(rows) != count or len(paired[part]) != count or not set(paired[part]) <= set(rgb_index):
            raise ValueError("Paired identity coverage differs")
        if any(int(rgb_index[s]["class_id"]) != r["class_index"] + 1 or _sample_id(r["rgb_path"]) != s
               or Path(r["rgb_path"]).resolve() != Path(rgb_index[s]["video_path"]).resolve()
               for s, r in paired[part].items()):
            raise ValueError("Paired identity, label or RGB source differs")
    if set(paired["train"]) & set(paired["val"]):
        raise ValueError("Paired split overlap")
    source = torch.load(args.init_adapter, map_location="cpu", weights_only=False)
    identity = source["identity"]
    if (identity["split_sha256"] != sha256_file(args.split_manifest) or identity["model_revision"] != MODEL_REVISION
            or identity["test_evaluated"] is not False or identity["arm"] != "vision"):
        raise ValueError("Warm-start adapter provenance differs")
    reference = torch.load(args.baseline_predictions, map_location="cpu", weights_only=True)
    index = {s: i for i, s in enumerate(reference["sample_ids"])}
    if len(index) != 12960 or len(reference["sample_ids"]) != 12960 or set(index) != {_sample_id(r["video_path"]) for r in split["val"]}:
        raise ValueError("Baseline validation identities differ")
    val_labels = torch.tensor([int(r["class_id"]) - 1 for r in split["val"]])
    val_indices = torch.tensor([index[_sample_id(r["video_path"])] for r in split["val"]])
    if not torch.equal(reference["labels"][val_indices], val_labels) or not torch.isfinite(reference["logits"]).all():
        raise ValueError("Baseline labels or finite logits differ")
    hashes = {name: sha256_file(path) for name, path in SOURCES.items()}
    for name in ("split_manifest", "paired_manifest", "init_adapter", "baseline_predictions", "gate0_decision"):
        hashes[name] = sha256_file(getattr(args, name))
    prerequisite_hashes = {}
    if args.mode == "pilot":
        for name, path, expected in (("smoke", args.smoke_report, "pass"), ("audit", args.audit_report, "audit_complete")):
            prior = json.loads(path.read_text())
            if prior["status"] != expected or prior["source_sha256"] != hashes or prior["test_evaluated"] is not False:
                raise ValueError(f"Pilot requires {name} from identical sources and inputs")
            prerequisite_hashes[name] = sha256_file(path)
    audit_rows = balanced([r for r in split["val"] if _sample_id(r["video_path"]) in paired["val"]], 472, args.seed)
    counts = torch.bincount(torch.tensor([int(r["class_id"]) - 1 for r in audit_rows]), minlength=59)
    if len(counts) != 59 or not torch.equal(counts, torch.full((59,), 8)):
        raise ValueError("Audit requires exactly eight paired validation samples per class")
    train_rows = [] if args.mode == "audit" else balanced(split["train"], 128, args.seed) if args.mode == "smoke" else split["train"]
    val_rows = audit_rows if args.mode == "audit" else balanced(split["val"], 120, args.seed) if args.mode == "smoke" else split["val"]
    ids = [_sample_id(r["video_path"]) for r in val_rows]
    selected = torch.tensor([index[s] for s in ids])
    labels = reference["labels"][selected]
    baseline_logits = reference["logits"][selected]
    processor = XCLIPProcessor.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=True)
    cache = ProcessedUInt8Cache(args.rgb_cache_dir, processor, model_name=MODEL, model_revision=MODEL_REVISION)
    base = XCLIPModel.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=True, attn_implementation="eager")
    config = base.config.vision_config
    if (config.num_hidden_layers, config.num_attention_heads, config.patch_size, config.image_size, config.num_frames, config.attention_dropout) != (12, 12, 32, 224, 8, 0):
        raise ValueError("Native attention token/layer/head assumptions changed")
    model = XCLIPVisionLoRAClassifier(base, num_classes=120, head_dropout=.25, feature_noise_std=.01,
                                     upper_blocks=2, lora_rank=16, lora_alpha=32, lora_dropout=.05).cuda().eval()
    if model.target_modules != source["model"]["target_modules"]:
        raise ValueError("Warm-start LoRA targets differ")
    load_lora_state_dict(model, source["model"]["lora_state_dict"])
    model.classifier.load_state_dict(source["model"]["classifier_state_dict"], strict=True)
    del base, source
    audit = model.trainable_parameter_audit().as_dict()
    root.mkdir(parents=True, exist_ok=True)
    lock = {"mode": args.mode, "seed": args.seed, "source_sha256": hashes, "prerequisite_sha256": prerequisite_hashes,
            "model_revision": MODEL_REVISION, "attention_backend": "eager", "layers": [10, 11], "heads": [0, 1, 2, 3],
            "tokens": "CLS, 49 patches, cross-frame message; conditional patch KL, unconditioned mass reported",
            "target_version": TARGET_VERSION, "attention_weight": .1, "epochs": 2 if args.mode == "pilot" else 0 if args.mode == "audit" else 1,
            "train_ids": [_sample_id(r["video_path"]) for r in train_rows], "validation_ids": ids,
            "audit_ids": [_sample_id(r["video_path"]) for r in audit_rows], "batch_size": args.batch_size, "workers": args.workers,
            "optimizer": "AdamW", "learning_rate": 2.25e-5, "weight_decay": 5e-4, "gradient_clip": 1., "label_smoothing": .1,
            "selection": "Fixed epoch2 for pilot; epoch1 reported separately; no best-epoch substitution",
            "rgb_only_inference": True, "test_evaluated": False,
            "limitations": "Partial59 skeleton; no matched CE control; selected heads are a fixed heuristic"}
    write_json(root / "experiment_lock.json", lock)
    write_json(root / "trainable_parameters.json", audit)
    snapshots = root / "sources"
    snapshots.mkdir()
    for name, path in SOURCES.items():
        (snapshots / f"{name}.py").write_text(path.read_text())
    (root / "runner_snapshot.py").write_text(Path(__file__).read_text())
    phase = root / args.mode
    phase.mkdir()
    started = time.monotonic()
    def progress(state, **fields):
        record = {"state": state, "elapsed_seconds": time.monotonic() - started, "test_evaluated": False, **fields}
        write_json(phase / "progress.json", record)
        print(json.dumps(record), flush=True)
    def loader(rows, pairing, train=False):
        return DataLoader(Gate2Dataset(rows, pairing, cache, args.target_cache_dir), batch_size=args.batch_size,
                          shuffle=train, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
                          worker_init_fn=_worker_seed, generator=torch.Generator().manual_seed(args.seed))
    def parity_check(rows):
        batch = next(iter(loader(rows, paired["val"])))
        pixels = batch[0].cuda()
        with torch.no_grad():
            direct = model(pixels)
            observed, _ = model(pixels, return_native_attentions=True)
        expected = reference["logits"][[index[_sample_id(r["video_path"])] for r in rows[:len(pixels)]]].cuda()
        errors = {"saved_baseline": float((direct - expected).abs().max()), "attention_output": float((direct - observed).abs().max()),
                  "prediction_disagreements": int(direct.argmax(-1).ne(expected.argmax(-1)).sum())}
        if errors["saved_baseline"] > 1e-3 or errors["attention_output"] > 1e-5 or errors["prediction_disagreements"]:
            raise RuntimeError(f"Warm-start parity failed: {errors}")
        return errors
    parity = parity_check(audit_rows)
    progress("warm_start_verified", errors=parity)
    def evaluate(rows, native=False, output_name="validation", overlays=False):
        model.eval()
        logits_all, labels_all, native_all, targets_all, masks_all = [], [], [], [], []
        evaluated = 0
        for batch_number, (pixels, y, occupancy, mask) in enumerate(loader(rows, paired["val"] if native else {}), 1):
            with torch.inference_mode():
                if native:
                    logits, attention = model(pixels.cuda(), return_native_attentions=True)
                    native_all.append(attention.cpu()); targets_all.append(occupancy); masks_all.append(mask)
                    if overlays and batch_number == 1:
                        save_overlay(pixels[0], attention[0].cpu(), occupancy[0], processor, root / f"{output_name}_overlay.png")
                else:
                    logits = model(pixels.cuda())
            logits_all.append(logits.cpu()); labels_all.append(y)
            evaluated += len(y)
            if batch_number % 25 == 0 or evaluated == len(rows):
                progress(output_name, samples=evaluated, total_samples=len(rows))
        z, y = torch.cat(logits_all), torch.cat(labels_all)
        sample_ids = [_sample_id(r["video_path"]) for r in rows]
        if not torch.equal(y, reference["labels"][[index[s] for s in sample_ids]]):
            raise RuntimeError("Validation labels/order differs")
        payload = {"logits": z, "labels": y, "sample_ids": sample_ids, "test_evaluated": False}
        metrics = None
        if native:
            a, o, m = torch.cat(native_all), torch.cat(targets_all), torch.cat(masks_all)
            payload.update(attention=a, occupancy=o, frame_mask=m)
            metrics = native_attention_metrics(a, o, m)
            if not metrics["valid_frames"]:
                raise RuntimeError("Audit has no valid frames")
            payload["target"] = native_pose_distribution(o, m)
        torch.save(payload, root / f"{output_name}_predictions.pt")
        return payload, metrics
    # The fixed native map audit is small; full validation does not retain maps.
    before, before_metrics = evaluate(audit_rows, True, "baseline_audit", True)
    full_audit_prior = reference["logits"][[index[s] for s in before["sample_ids"]]]
    parity["audit_max_abs_diff"] = float((before["logits"] - full_audit_prior).abs().max())
    parity["audit_prediction_disagreements"] = int(before["logits"].argmax(-1).ne(full_audit_prior.argmax(-1)).sum())
    if parity["audit_max_abs_diff"] > 1e-3 or parity["audit_prediction_disagreements"]:
        raise RuntimeError("Full baseline audit parity failed")
    if args.mode == "audit":
        progress("complete", samples=len(audit_rows))
        return {"status": "audit_complete", "mode": args.mode, "raw_accuracy_gain": 0., "source_sha256": hashes,
                "native_metrics": before_metrics, "baseline_identity_errors": parity, "samples": len(audit_rows),
                "test_evaluated": False, "training_steps": 0}
    # Verify the attention-only backward separately from CE; it must reach Q, not same-layer V.
    pixels, y, occupancy, mask = next(iter(loader(audit_rows, paired["val"])))
    pixels, y, occupancy, mask = [item.cuda() for item in (pixels, y, occupancy, mask)]
    _, a = model(pixels, return_native_attentions=True)
    spatial = native_attention_loss(a, native_pose_distribution(occupancy, mask), mask)
    spatial.backward()
    native_gradients = {n: float(p.grad.norm()) for n, p in model.named_parameters()
                        if n.startswith("vision_model.") and n.endswith("lora_b") and p.grad is not None}
    if not any("q_proj" in n and v > 0 for n, v in native_gradients.items()) or not all(np.isfinite(v) for v in native_gradients.values()):
        raise RuntimeError(f"Attention-to-Q gradient failed: {native_gradients}")
    model.zero_grad(set_to_none=True)
    z, a = model(pixels, return_native_attentions=True)
    F.cross_entropy(z, y, label_smoothing=.1).backward()
    ce_gradients = {family: sum(float(p.grad.norm()) for n, p in model.named_parameters() if n.startswith(prefix) and p.grad is not None)
                    for family, prefix in (("vision", "vision_model."), ("mit", "mit."), ("classifier", "classifier."))}
    if not all(v > 0 and np.isfinite(v) for v in ce_gradients.values()):
        raise RuntimeError(f"CE gradient failed: {ce_gradients}")
    zero = native_attention_loss(a, torch.zeros_like(occupancy), torch.zeros_like(mask))
    if float(zero.detach()) != 0:
        raise RuntimeError("Invalid frame loss must be zero")
    model.zero_grad(set_to_none=True)
    gradients = {"attention_only": native_gradients, "ce": ce_gradients, "invalid_frame_loss": float(zero.detach())}
    write_json(root / "gradients.json", gradients)
    del a, z, zero, spatial
    training = loader(train_rows, paired["train"], True)
    epochs = lock["epochs"]
    steps_per_epoch = 8 if args.mode == "smoke" else len(training)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=2.25e-5, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps_per_epoch * epochs)
    history, steps, seen = [], 0, 0
    step_times = []
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, epochs + 1):
        model.train()
        totals = {"ce": 0., "attention_kl": 0., "samples": 0, "valid_frames": 0}
        for step, (x, y, o, m) in enumerate(training, 1):
            tick = time.monotonic()
            x, y, o, m = [item.cuda(non_blocking=True) for item in (x, y, o, m)]
            optimizer.zero_grad(set_to_none=True)
            z, a = model(x, return_native_attentions=True)
            ce = F.cross_entropy(z.float(), y, label_smoothing=.1)
            kl = native_attention_loss(a, native_pose_distribution(o, m), m)
            loss = ce + .1 * kl
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite native attention training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step(); scheduler.step()
            torch.cuda.synchronize()
            step_times.append(time.monotonic() - tick)
            steps += 1; seen += len(y)
            totals["ce"] += float(ce.detach()) * len(y)
            totals["attention_kl"] += float(kl.detach()) * int(m.sum())
            totals["samples"] += len(y); totals["valid_frames"] += int(m.sum())
            if step % 25 == 0 or step == steps_per_epoch:
                progress("training", epoch=epoch, step=step, total_steps=steps_per_epoch, samples=totals["samples"],
                         valid_frames=totals["valid_frames"], ce=float(ce.detach()), attention_kl=float(kl.detach()),
                         learning_rate=scheduler.get_last_lr()[0], step_seconds=step_times[-1],
                         peak_memory_bytes=torch.cuda.max_memory_allocated())
            if step == steps_per_epoch:
                break
        if totals["samples"] != (32 if args.mode == "smoke" else 80640):
            raise RuntimeError("Training sample coverage differs from fixed budget")
        checkpoint = {"lock": lock, "epoch": epoch, "lora": lora_state_dict(model), "classifier": model.classifier.state_dict(),
                      "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "steps": steps, "history": history,
                      "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all()}
        torch.save(checkpoint, root / f"epoch{epoch}_checkpoint.pt")
        predictions, _ = evaluate(val_rows, False, f"epoch{epoch}_validation")
        report = summary(predictions["logits"], labels, ids, set(paired["val"]))
        history.append({"epoch": epoch, "training": totals, "classification": report})
        write_json(root / "history.json", history)
        progress("epoch_complete", epoch=epoch, raw_accuracy=report["raw_accuracy"], raw_macro_f1=report["raw_macro_f1"])
    after, after_metrics = evaluate(audit_rows, True, "final_audit", True)
    causal = causal_diagnostic(model, split["val"], paired["val"], cache, args.target_cache_dir, progress)
    write_json(root / "masking_diagnostic.json", causal)
    final = summary(predictions["logits"], labels, ids, set(paired["val"]))
    baseline = summary(baseline_logits, labels, ids, set(paired["val"]))
    delta = changes(predictions["logits"], baseline_logits, labels)
    # Verify trained export before merging: no skeleton or attention output in the inference model.
    model.eval()
    with torch.no_grad():
        expected = model(pixels)
    merge_lora(model)
    state = merged_base_state_dict(model)
    restored = XCLIPVisionClassifier(XCLIPModel.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=True),
                                    num_classes=120, head_dropout=.25, feature_noise_std=.01).cuda().eval()
    restored.load_state_dict(state, strict=True)
    with torch.no_grad():
        reload_error = float((restored(pixels) - expected).abs().max())
    if reload_error > 1e-4:
        raise RuntimeError("RGB-only merged reload parity failed")
    torch.save({"state_dict": state, "provenance": lock}, root / "merged_rgb_only.pt")
    status = "pass" if args.mode == "smoke" else decision(final, baseline)
    mechanism = after_metrics["target_kl"] < before_metrics["target_kl"] and after_metrics["joint_region_mass"] > before_metrics["joint_region_mass"]
    progress("complete", status=status, training_steps=steps)
    return {"status": status, "mode": args.mode, "raw_accuracy_gain": final["raw_accuracy"] - baseline["raw_accuracy"],
            "source_sha256": hashes, "prerequisite_sha256": prerequisite_hashes, "classification": final, "baseline": baseline,
            "prediction_changes": delta, "native_metrics": {"baseline": before_metrics, "final": after_metrics, "mechanism_direction_pass": mechanism},
            "baseline_identity_errors": parity, "gradients": gradients, "merged_reload_max_abs_diff": reload_error,
            "training_steps": steps, "train_samples_seen": seen, "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "median_step_seconds": float(np.median(step_times)), "elapsed_seconds": time.monotonic() - started,
            "history": history, "test_evaluated": False,
            "interpretation": "Fixed-budget single-seed adaptation, not control-isolated attention gain or deployment approval"}


def save_overlay(pixels, attention, occupancy, processor, path):
    mean = torch.tensor(processor.image_processor.image_mean).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_processor.image_std).view(1, 3, 1, 1)
    rgb = ((pixels * std + mean).clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).numpy()
    patch = attention[:, :, 1:50].mean(1)
    patch /= patch.sum(-1, keepdim=True).clamp_min(1e-8)
    strip = Image.new("RGB", (224 * 8, 224 * 2))
    for frame in range(8):
        original = Image.fromarray(rgb[frame])
        heat = patch[frame].reshape(7, 7).numpy()
        heat = (heat / max(float(heat.max()), 1e-8) * 255).astype(np.uint8)
        alpha = Image.fromarray(heat).resize((224, 224), Image.Resampling.NEAREST)
        color = Image.new("RGB", (224, 224), "red")
        mixed = Image.composite(color, original, alpha.point(lambda v: int(v * .45)))
        draw = ImageDraw.Draw(mixed)
        for p in occupancy[frame].nonzero().flatten().tolist():
            r, c = divmod(p, 7)
            draw.rectangle((c*32, r*32, c*32+31, r*32+31), outline="lime", width=2)
        strip.paste(original, (frame*224, 0)); strip.paste(mixed, (frame*224, 224))
    strip.save(path)


def causal_diagnostic(model, rows, paired, cache, target_cache, progress):
    model.eval()
    records = []
    generator = torch.Generator().manual_seed(42)
    dataset = Gate2Dataset([r for r in rows if _sample_id(r["video_path"]) in paired], paired, cache, target_cache)
    with torch.inference_mode():
        for row_index in range(len(dataset)):
            pixels, label, _, mask = dataset[row_index]
            if not bool(mask.any()):
                continue
            pixels = pixels.unsqueeze(0).cuda()
            logits, attention = model(pixels, return_native_attentions=True)
            scores = native_patch_distribution(attention).mean(2)
            ranking = scores.argsort(dim=-1, descending=True, stable=True)
            selections = {"highest": ranking[..., :5], "lowest": ranking[..., -5:],
                          "random": torch.rand(scores.shape, generator=generator).topk(5, -1).indices.cuda()}
            record = {"sample_id": _sample_id(dataset.rows[row_index]["video_path"]), "label": label, "valid_frame_mask": mask.tolist()}
            for name, patches in selections.items():
                record[name] = float(logits[0, label] - model(mask_patches(pixels, patches))[0, label])
            records.append(record)
            if len(records) == 32:
                break
    if len(records) != 32:
        raise RuntimeError("Masking diagnostic requires 32 valid paired samples")
    progress("masking_complete", samples=len(records))
    return {"samples": len(records), "patches_per_frame": 5, "replacement": "zero in normalized RGB", "records": records,
            "mean_logit_drop": {name: sum(r[name] for r in records) / len(records) for name in selections},
            "limitation": "Input intervention changes distribution and is not a standalone causal attribution", "test_evaluated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "smoke", "pilot"), required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    defaults = {
        "split_manifest": "artifacts/xclip/selected_split.json", "paired_manifest": "artifacts/skeleton_data/paired_manifest.json",
        "gate0_decision": "artifacts/skeletonclip_phase4/gate0/gate0_decision.json",
        "init_adapter": "artifacts/skeletonclip_phase3/vision_batch16_workers24_uint8_seed42/vision_lora_ce_seed42/best_adapter.pt",
        "baseline_predictions": "artifacts/skeletonclip_phase3/vision_batch16_workers24_uint8_seed42/vision_lora_ce_seed42/best_validation_logits.pt",
        "rgb_cache_dir": "cache/skeletonclip_phase3_rgb_uint8", "target_cache_dir": "cache/skeletonclip_phase4_targets_v1",
        "smoke_report": "artifacts/native_attention_v1/smoke/run_report.json", "audit_report": "artifacts/native_attention_v1/audit/run_report.json"}
    for name, default in defaults.items():
        parser.add_argument("--" + name.replace("_", "-"), type=Path, default=Path(default))
    args = parser.parse_args()
    if args.batch_size != 4 or args.seed != 42 or not 0 <= args.workers <= 24:
        parser.error("v1 fixes batch4/seed42; workers must be 0–24")
    try:
        report = run(args)
    except Exception as error:
        path = args.artifacts_dir / args.mode / "progress.json"
        if (args.artifacts_dir / "experiment_lock.json").exists():
            write_json(path, {"state": "failed", "error": str(error), "test_evaluated": False})
        raise
    write_json(args.artifacts_dir / "run_report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k in ("status", "mode", "raw_accuracy_gain", "native_metrics", "training_steps")}), flush=True)


if __name__ == "__main__":
    main()
