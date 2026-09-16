"""XCLIP MIT LoRA + CE-only training pipeline."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import XCLIPModel
from transformers import __version__ as transformers_version

from vlm.data import (
    build_split,
    load_split_manifest,
    load_split_manifest_metadata,
    save_split_manifest,
    scan_all_classes,
    select_split_classes,
    summarize_split_samples,
)
from vlm.evaluation import evaluate_features
from vlm.models import XCLIPMITLoRAClassifier
from vlm.models.lora import (
    lora_state_dict,
    merge_lora,
    merged_base_state_dict,
    unmerge_lora,
)
from vlm.reporting import save_accuracy_plot, save_checkpoint, write_json
from vlm.skeleton_teacher_cache import (
    DistillationAccumulator,
    DistillationDataset,
    TeacherLogitCache,
    sha256_file,
    validate_class_mapping,
)
from vlm.training.engine import ValidationCheckpoint, resolve_device, set_seed
from vlm.xclip_features import FRAME_SAMPLING_STRATEGY
from vlm.xclip_pre_mit_features import (
    FEATURE_STAGE,
    CachedXCLIPPreMITFeature,
    CachedXCLIPPreMITDataset,
    extract_or_load_xclip_pre_mit_features,
)


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _amp_settings(
    device: torch.device,
    enabled: bool,
) -> tuple[bool, torch.dtype | None]:
    if not enabled or device.type != "cuda":
        return False, None
    if torch.cuda.is_bf16_supported():
        return True, torch.bfloat16
    return True, torch.float16


def _optimizer_groups(
    model: XCLIPMITLoRAClassifier,
    *,
    lora_learning_rate: float,
    head_learning_rate: float,
    weight_decay: float,
) -> list[dict[str, object]]:
    groups: dict[tuple[str, bool], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        family = "head" if name.startswith("classifier.") else "lora"
        no_decay = parameter.ndim == 1 or name.endswith(".bias")
        groups.setdefault((family, no_decay), []).append(parameter)
    result: list[dict[str, object]] = []
    for (family, no_decay), parameters in groups.items():
        result.append(
            {
                "params": parameters,
                "lr": (
                    head_learning_rate
                    if family == "head"
                    else lora_learning_rate
                ),
                "weight_decay": 0.0 if no_decay else weight_decay,
            }
        )
    return result


def train_xclip_lora_epoch(
    model: XCLIPMITLoRAClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    max_grad_norm: float,
    gradient_accumulation_steps: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    epoch: int,
    show_batch_progress: bool,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    scaler_enabled = amp_enabled and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    total_loss = 0.0
    batches = (
        tqdm(loader, desc=f"Epoch {epoch} Training", leave=False)
        if show_batch_progress
        else loader
    )
    for batch_index, (features, labels) in enumerate(batches, start=1):
        features = features.to(device)
        labels = labels.to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            logits = model(features)
            loss = criterion(logits, labels)
            scaled_loss = loss / gradient_accumulation_steps
        scaler.scale(scaled_loss).backward()
        should_step = (
            batch_index % gradient_accumulation_steps == 0
            or batch_index == len(loader)
        )
        if should_step:
            scaler.unscale_(optimizer)
            clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                max_grad_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        batch_loss = loss.item()
        total_loss += batch_loss * features.size(0)
        if show_batch_progress:
            batches.set_postfix(loss=f"{batch_loss:.4f}")
    return total_loss / len(loader.dataset)


def train_xclip_lora_distillation_epoch(
    model: XCLIPMITLoRAClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    class_indices: torch.Tensor,
    temperature: float,
    kd_weight: float,
    label_smoothing: float,
    max_grad_norm: float,
    gradient_accumulation_steps: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
) -> dict[str, float]:
    """Train one epoch with exact sample-normalized CE and paired-normalized KD."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled and amp_dtype == torch.float16
    )
    totals = {"ce": 0.0, "kd": 0.0, "weighted_kd": 0.0, "total": 0.0}
    total_samples = paired_samples = optimizer_steps = 0
    accumulator = DistillationAccumulator()
    for batch_index, (features, labels, _sample_ids, teacher_logits, paired_mask) in enumerate(loader, start=1):
        features, labels = features.to(device), labels.to(device)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            logits = model(features)
        accumulator.add(
            logits,
            labels,
            teacher_logits.to(device),
            paired_mask.to(device),
            class_indices.to(device),
            temperature=temperature,
            label_smoothing=label_smoothing,
        )
        should_step = (
            batch_index % gradient_accumulation_steps == 0
            or batch_index == len(loader)
        )
        if not should_step:
            continue
        ce, kd, loss = accumulator.losses(kd_weight=kd_weight)
        if not all(torch.isfinite(value) for value in (ce, kd, loss)):
            raise RuntimeError("Non-finite CE/KD/total loss in distillation epoch")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if any(
            parameter.grad is not None
            and not torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise RuntimeError("Non-finite gradient in distillation epoch")
        clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            max_grad_norm,
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        totals["ce"] += float(ce.detach()) * accumulator.total_count
        totals["kd"] += float(kd.detach()) * accumulator.paired_count
        totals["weighted_kd"] += (
            float((kd_weight * kd).detach()) * accumulator.paired_count
        )
        totals["total"] += float(loss.detach()) * accumulator.total_count
        total_samples += accumulator.total_count
        paired_samples += accumulator.paired_count
        accumulator = DistillationAccumulator()
    return {
        "ce_loss": totals["ce"] / total_samples,
        "kd_loss": totals["kd"] / paired_samples if paired_samples else 0.0,
        "weighted_kd_loss": totals["weighted_kd"] / paired_samples if paired_samples else 0.0,
        "total_loss": totals["ce"] / total_samples
        + (totals["weighted_kd"] / paired_samples if paired_samples else 0.0),
        "total_samples": float(total_samples),
        "paired_samples": float(paired_samples),
        "paired_fraction": paired_samples / total_samples,
        "optimizer_steps": float(optimizer_steps),
    }


class TrainingFeatureAugmentationDataset(Dataset):
    """Apply stochastic, cache-safe augmentation after loading a feature."""

    def __init__(
        self,
        base: CachedXCLIPPreMITDataset,
        *,
        frame_dropout: float,
        temporal_jitter: int,
        feature_noise_std: float,
    ) -> None:
        if not 0.0 <= frame_dropout < 1.0:
            raise ValueError("frame_dropout must be in [0, 1)")
        if temporal_jitter < 0:
            raise ValueError("temporal_jitter must be non-negative")
        if feature_noise_std < 0:
            raise ValueError("feature_noise_std must be non-negative")
        self.base = base
        self.frame_dropout = frame_dropout
        self.temporal_jitter = temporal_jitter
        self.feature_noise_std = feature_noise_std

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features, label = self.base[index]
        features = features.clone()
        if self.temporal_jitter:
            positions = torch.arange(features.shape[0])
            offsets = torch.randint(
                -self.temporal_jitter,
                self.temporal_jitter + 1,
                (features.shape[0],),
            )
            positions = (positions + offsets).clamp(
                min=0,
                max=features.shape[0] - 1,
            )
            features = features[positions]
        if self.frame_dropout:
            mask = torch.rand(features.shape[0]) < self.frame_dropout
            if bool(mask.all()):
                mask[torch.randint(features.shape[0], ())] = False
            features[mask] = 0
        if self.feature_noise_std:
            features.add_(
                torch.randn_like(features) * self.feature_noise_std
            )
        return features, label


def _scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    if warmup_epochs < 0 or warmup_epochs >= epochs:
        raise ValueError("warmup_epochs must be in [0, epochs)")

    def factor(epoch: int) -> float:
        if warmup_epochs and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _macro_f1_from_predictions(
    predictions: Sequence[dict[str, object]], num_classes: int
) -> float:
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for row in predictions:
        confusion[int(row["true_class_index"]), int(row["predicted_class_index"])] += 1
    scores = []
    for index in range(num_classes):
        true_positive = int(confusion[index, index])
        false_positive = int(confusion[:, index].sum()) - true_positive
        false_negative = int(confusion[index, :].sum()) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return sum(scores) / num_classes


def _evaluate_validation_detailed(
    model: nn.Module,
    loader: DataLoader,
    samples: Sequence[CachedXCLIPPreMITFeature],
    device: torch.device,
    class_ids: Sequence[int],
) -> tuple[dict[str, float], list[dict[str, object]], torch.Tensor, torch.Tensor]:
    model.eval()
    logits_rows, label_rows = [], []
    with torch.inference_mode():
        for features, labels in loader:
            logits_rows.append(model(features.to(device)).float().cpu())
            label_rows.append(labels.long().cpu())
    logits, labels = torch.cat(logits_rows), torch.cat(label_rows)
    probabilities = logits.softmax(dim=1)
    predicted = logits.argmax(dim=1)
    predictions = []
    for sample, truth, prediction in zip(samples, labels.tolist(), predicted.tolist()):
        predictions.append(
            {
                "video": sample.video_path.name,
                "true_class_id": sample.class_id,
                "predicted_class_id": class_ids[prediction],
                "true_class_index": truth,
                "predicted_class_index": prediction,
            }
        )
    confidence = probabilities.max(dim=1).values
    correct = predicted.eq(labels)
    ece = 0.0
    for lower in torch.arange(0, 1, 0.1):
        mask = (confidence > lower) & (confidence <= lower + 0.1)
        if mask.any():
            ece += float(mask.float().mean()) * abs(
                float(correct[mask].float().mean()) - float(confidence[mask].mean())
            )
    metrics = {
        "accuracy": float(correct.float().mean()),
        "macro_f1": _macro_f1_from_predictions(predictions, len(class_ids)),
        "nll": float(nn.functional.cross_entropy(logits, labels)),
        "ece": ece,
    }
    return metrics, predictions, logits, labels


def _append_training_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(message + "\n")
        stream.flush()


def build_cached_datasets(
    cached: dict[str, list[CachedXCLIPPreMITFeature]],
    *,
    preload: bool,
) -> dict[str, CachedXCLIPPreMITDataset]:
    """Build split datasets, optionally loading every tensor into RAM once."""
    return {
        split_name: CachedXCLIPPreMITDataset(samples, preload=preload)
        for split_name, samples in cached.items()
    }


def run_pipeline(
    args: argparse.Namespace,
    *,
    evaluate_test: bool = True,
    cache_read_only: bool = False,
    prepared_cached: dict[str, list[CachedXCLIPPreMITFeature]] | None = None,
    prepared_datasets: dict[str, CachedXCLIPPreMITDataset] | None = None,
    print_results: bool = True,
    print_run_details: bool = True,
) -> dict[str, object]:
    if args.teacher_cache and evaluate_test:
        raise ValueError(
            "Distillation runs are under the Phase 2 test embargo; "
            "pass --no-evaluate-test until the locked Gate 5 test-once command."
        )
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if args.warmup_epochs < 0 or args.warmup_epochs >= args.epochs:
        raise ValueError("warmup_epochs must be in [0, epochs)")
    if (
        args.early_stopping_patience is not None
        and args.early_stopping_patience < 1
    ):
        raise ValueError("early_stopping_patience must be at least 1")
    if args.train_eval_frequency < 1:
        raise ValueError("train_eval_frequency must be at least 1")
    if args.epoch_log_frequency < 1:
        raise ValueError("epoch_log_frequency must be at least 1")
    set_seed(args.seed)
    videos_dir = Path(args.videos_dir)
    cache_dir = Path(args.cache_dir)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = Path(args.split_manifest) if args.split_manifest else None
    loaded_split = (
        load_split_manifest(source_manifest)
        if source_manifest is not None
        else None
    )
    source_metadata = (
        load_split_manifest_metadata(source_manifest)
        if source_manifest is not None
        else {}
    )
    if loaded_split is not None and (
        args.max_videos_per_class is not None
        or args.videos_per_split is not None
    ):
        raise ValueError(
            "A loaded --split-manifest already fixes sample membership; "
            "do not combine it with a video cap."
        )
    discovered_class_ids = (
        sorted(
            {
                sample.class_id
                for samples in loaded_split.values()
                for sample in samples
            }
        )
        if loaded_split is not None
        else scan_all_classes(videos_dir)
    )
    class_ids = list(args.class_ids) if args.class_ids else discovered_class_ids
    if not class_ids:
        raise ValueError(
            f"No supported class-labelled videos found in {videos_dir}"
        )
    if loaded_split is not None and not evaluate_test:
        split = {
            name: [sample for sample in samples if sample.class_id in class_ids]
            for name, samples in loaded_split.items()
        }
        missing = {
            name: sorted(set(class_ids) - {sample.class_id for sample in split[name]})
            for name in ("train", "val")
        }
        if any(missing.values()):
            raise ValueError(
                "Validation-only split does not contain every requested class "
                f"in train and val: {missing}"
            )
    else:
        split = (
            select_split_classes(loaded_split, class_ids)
            if loaded_split is not None
            else build_split(
            videos_dir=videos_dir,
            class_ids=class_ids,
            videos_per_split=args.videos_per_split,
            seed=args.seed,
            max_videos_per_class=args.max_videos_per_class,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            )
        )
    split_summary = {
        name: summarize_split_samples(samples)
        for name, samples in split.items()
    }
    if print_run_details:
        print(json.dumps({"split_summary": split_summary}, indent=2))
    save_split_manifest(
        split,
        artifacts_dir / "selected_split.json",
        seed=int(source_metadata.get("seed", args.seed)),
        sample_fps=None,
        max_seq_len=args.frames,
        split_strategy=str(
            source_metadata.get("split_strategy", "participant_disjoint")
        ),
        frame_sampling_strategy=FRAME_SAMPLING_STRATEGY,
        source_manifest=source_manifest,
    )
    device = resolve_device(args.device)
    active_split = (
        split
        if evaluate_test
        else {name: split[name] for name in ("train", "val")}
    )
    cached = prepared_cached
    if cached is None:
        cached = extract_or_load_xclip_pre_mit_features(
            active_split,
            cache_dir=cache_dir,
            model_name=args.model_name,
            num_frames=args.frames,
            device=device,
            force_extract=args.force_extract,
            feature_batch_size=args.feature_batch_size,
            decode_workers=args.decode_workers,
            read_only=cache_read_only,
            show_progress=args.cache_progress,
        )
    else:
        cached = {
            name: cached[name]
            for name in active_split
        }
    expected_counts = {
        name: len(samples) for name, samples in active_split.items()
    }
    actual_counts = {name: len(samples) for name, samples in cached.items()}
    if actual_counts != expected_counts:
        raise ValueError(
            "Prepared XCLIP cache split counts do not match the manifest"
        )
    extraction_summary = {
        "device": str(device),
        "train_samples": len(cached["train"]),
        "val_samples": len(cached["val"]),
        "test_samples": len(cached.get("test", [])),
        "model_name": args.model_name,
        "frame_sampling_strategy": FRAME_SAMPLING_STRATEGY,
        "feature_stage": FEATURE_STAGE,
        "frames": args.frames,
        "feature_batch_size": args.feature_batch_size,
        "decode_workers": args.decode_workers,
        "cache_dir": str(cache_dir),
        "split_summary": split_summary,
    }
    if args.extract_only:
        if print_results:
            print(json.dumps(extraction_summary, indent=2))
        return {**extraction_summary, "_cached_features": cached}

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.loader_workers,
        "pin_memory": args.pin_memory,
        "persistent_workers": (
            args.persistent_workers and args.loader_workers > 0
        ),
    }
    datasets = prepared_datasets or build_cached_datasets(
        cached,
        preload=args.preload_cache,
    )
    canonical_train_dataset = datasets["train"]
    train_dataset = TrainingFeatureAugmentationDataset(
        canonical_train_dataset,
        frame_dropout=args.frame_dropout,
        temporal_jitter=args.temporal_jitter,
        feature_noise_std=args.feature_noise_std,
    )
    teacher_provenance = None
    if args.teacher_cache:
        validation = json.loads(
            Path(args.teacher_cache_validation).read_text(encoding="utf-8")
        )
        if validation.get("status") != "passed":
            raise ValueError("Teacher cache validation report is not passed")
        if validation.get("cache_path") != str(args.teacher_cache) or validation.get(
            "cache_sha256"
        ) != sha256_file(Path(args.teacher_cache)):
            raise ValueError("Teacher cache path/hash does not match its validation report")
        teacher_cache = TeacherLogitCache(
            Path(args.teacher_cache),
            expected_identity=validation["cache_identity"],
        )
        train_dataset = DistillationDataset(train_dataset, cached["train"], teacher_cache)
        mapping_payload = json.loads(
            Path(args.class_mapping).read_text(encoding="utf-8")
        )
        class_indices = validate_class_mapping(mapping_payload)
        teacher_provenance = {
            "cache_identity": validation["cache_identity"],
            "class_mapping_sha256": mapping_payload["mapping_sha256"],
        }
    val_dataset = datasets["val"]
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    train_eval_loader = DataLoader(
        canonical_train_dataset,
        shuffle=False,
        **loader_options,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    test_loader = (
        DataLoader(
            datasets["test"],
            shuffle=False,
            **loader_options,
        )
        if evaluate_test
        else None
    )

    xclip = XCLIPModel.from_pretrained(args.model_name)
    full_base_model_parameters = sum(
        parameter.numel() for parameter in xclip.parameters()
    )
    native_frames = int(xclip.config.vision_config.num_frames)
    if args.frames != native_frames:
        raise ValueError(
            f"{args.model_name} expects {native_frames} frames, but "
            f"--frames is {args.frames}."
        )
    model = XCLIPMITLoRAClassifier(
        xclip,
        num_classes=len(class_ids),
        head_dropout=args.dropout,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    ).to(device)
    del xclip
    audit = model.trainable_parameter_audit()
    audit_payload = audit.as_dict()
    audit_payload["full_base_model_parameters"] = full_base_model_parameters
    audit_payload["ratio_to_full_base_model"] = (
        audit.trainable_parameters / full_base_model_parameters
    )
    if print_run_details:
        print(json.dumps({"trainable_parameter_audit": audit_payload}, indent=2))
    optimizer = torch.optim.AdamW(
        _optimizer_groups(
            model,
            lora_learning_rate=args.lora_learning_rate,
            head_learning_rate=args.head_learning_rate,
            weight_decay=args.weight_decay,
        )
    )
    scheduler = _scheduler(
        optimizer,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    amp_enabled, amp_dtype = _amp_settings(device, args.amp)
    effective_batch_size = (
        args.batch_size * args.gradient_accumulation_steps
    )

    history: list[dict[str, float | None]] = []
    training_started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best_checkpoint = ValidationCheckpoint()
    epochs_without_improvement = 0
    start_epoch = 1
    last_checkpoint_path = artifacts_dir / "last_checkpoint.pt"
    training_identity = {
        "schema_version": 1,
        "arguments": {
            key: value
            for key, value in vars(args).items()
            if key not in {"resume_training", "batch_progress", "cache_progress", "epoch_log_frequency"}
        },
        "class_ids": class_ids,
        "split_manifest_sha256": (
            sha256_file(source_manifest) if source_manifest is not None else None
        ),
        "teacher_provenance": teacher_provenance,
    }
    if args.resume_training and last_checkpoint_path.exists():
        resumed = torch.load(last_checkpoint_path, map_location=device, weights_only=False)
        if resumed.get("identity") != training_identity:
            raise ValueError("Resume checkpoint identity differs from this training run")
        model.load_state_dict(resumed["model_state_dict"], strict=True)
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        scheduler.load_state_dict(resumed["scheduler_state_dict"])
        history = resumed["history"]
        best_checkpoint = ValidationCheckpoint(
            epoch=int(resumed["best_epoch"]),
            score=tuple(resumed["best_score"]),
            state_dict=resumed["best_state_dict"],
        )
        epochs_without_improvement = int(resumed["epochs_without_improvement"])
        torch.set_rng_state(resumed["torch_rng_state"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in resumed["cuda_rng_states"]]
            )
        start_epoch = int(resumed["epoch"]) + 1
        _append_training_log(
            artifacts_dir / "train.log",
            f"resume epoch={start_epoch} completed_epochs={len(history)}",
        )
    elif args.resume_training:
        _append_training_log(
            artifacts_dir / "train.log", "resume requested; no checkpoint found; starting epoch=1"
        )
    for epoch in range(start_epoch, args.epochs + 1):
        distillation_metrics = None
        if args.teacher_cache:
            distillation_metrics = train_xclip_lora_distillation_epoch(
                model, train_loader, optimizer, device, class_indices=class_indices,
                temperature=args.kd_temperature, kd_weight=args.kd_weight,
                label_smoothing=args.label_smoothing, max_grad_norm=args.max_grad_norm,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                amp_enabled=amp_enabled, amp_dtype=amp_dtype,
            )
            loss = distillation_metrics["total_loss"]
        else:
            loss = train_xclip_lora_epoch(
                model, train_loader, optimizer, criterion, device,
                max_grad_norm=args.max_grad_norm, gradient_accumulation_steps=args.gradient_accumulation_steps,
                amp_enabled=amp_enabled, amp_dtype=amp_dtype, epoch=epoch,
                show_batch_progress=args.batch_progress,
            )
        val_accuracy, _ = evaluate_features(
            model,
            val_loader,
            cached["val"],
            device,
            class_ids,
        )
        scheduler.step()
        improved = best_checkpoint.consider(
            model,
            epoch=epoch,
            score=val_accuracy,
        )
        epochs_without_improvement = (
            0 if improved else epochs_without_improvement + 1
        )
        should_stop = (
            args.early_stopping_patience is not None
            and epochs_without_improvement >= args.early_stopping_patience
        )
        should_evaluate_train = (
            epoch == 1
            or epoch % args.train_eval_frequency == 0
            or epoch == args.epochs
            or should_stop
        )
        train_accuracy: float | None = None
        if should_evaluate_train:
            train_accuracy, _ = evaluate_features(
                model,
                train_eval_loader,
                cached["train"],
                device,
                class_ids,
            )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
                **(distillation_metrics or {}),
            }
        )
        progress = {
            "state": "running",
            "identity": training_identity,
            "completed_epoch": epoch,
            "total_epochs": args.epochs,
            "best_epoch": best_checkpoint.epoch,
            "best_validation_accuracy": (
                best_checkpoint.score[0] if best_checkpoint.score else None
            ),
            "latest": history[-1],
            "test_evaluated": False,
        }
        write_json(artifacts_dir / "progress.json", progress)
        log_line = (
            f"epoch={epoch:02d} train_loss={loss:.6f} "
            f"train_accuracy={train_accuracy if train_accuracy is not None else 'n/a'} "
            f"val_accuracy={val_accuracy:.6f} "
            f"ce={distillation_metrics['ce_loss'] if distillation_metrics else loss:.6f} "
            f"kd={distillation_metrics['kd_loss'] if distillation_metrics else 0.0:.6f} "
            f"paired={int(distillation_metrics['paired_samples']) if distillation_metrics else 0}"
        )
        _append_training_log(artifacts_dir / "train.log", log_line)
        checkpoint_payload = {
            "identity": training_identity,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "best_epoch": best_checkpoint.epoch,
            "best_score": best_checkpoint.score,
            "best_state_dict": best_checkpoint.state_dict,
            "epochs_without_improvement": epochs_without_improvement,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": (
                torch.cuda.get_rng_state_all() if device.type == "cuda" else []
            ),
        }
        temporary_checkpoint = last_checkpoint_path.with_suffix(".pt.tmp")
        torch.save(checkpoint_payload, temporary_checkpoint)
        temporary_checkpoint.replace(last_checkpoint_path)
        train_accuracy_text = (
            f"{train_accuracy:.4f}"
            if train_accuracy is not None
            else "n/a"
        )
        should_log_epoch = (
            epoch == 1
            or epoch % args.epoch_log_frequency == 0
            or epoch == args.epochs
            or should_stop
        )
        if should_log_epoch:
            print(
                f"epoch={epoch:02d} train_loss={loss:.4f} "
                f"train_accuracy={train_accuracy_text} "
                f"val_accuracy={val_accuracy:.4f}"
            )
        if should_stop:
            print(
                "early_stopping "
                f"epoch={epoch} patience={args.early_stopping_patience}"
            )
            break

    best_checkpoint.restore(model)
    validation_metrics, validation_predictions, validation_logits, validation_labels = (
        _evaluate_validation_detailed(
            model, val_loader, cached["val"], device, class_ids
        )
    )
    validation_accuracy = validation_metrics["accuracy"]
    final_accuracy: float | None = None
    predictions: list[dict[str, object]] | None = None
    if evaluate_test:
        assert test_loader is not None
        final_accuracy, predictions = evaluate_features(
            model,
            test_loader,
            cached["test"],
            device,
            class_ids,
        )
    config = vars(args).copy()
    adapter_path = artifacts_dir / "xclip_lora_classifier.pt"
    merged_path = artifacts_dir / "xclip_lora_merged.pt"
    save_checkpoint(
        adapter_path,
        {
            "base_model_id": args.model_name,
            "lora_state_dict": lora_state_dict(model.mit),
            "classifier_state_dict": model.classifier.state_dict(),
            "class_ids": class_ids,
            "config": config,
            "target_modules": model.target_modules,
            "best_epoch": best_checkpoint.epoch,
            "validation_accuracy": validation_accuracy,
            "teacher_provenance": teacher_provenance,
        },
    )
    parity_features, _ = next(iter(val_loader))
    parity_features = parity_features.to(device)
    model.eval()
    # Check the algebraic merge in FP32. Under CUDA autocast the unmerged path
    # performs two low-precision linear operations while the merged path performs
    # one, so BF16/FP16 rounding alone can exceed a meaningful parity tolerance.
    # Keeping both forwards outside the same autocast context also prevents its
    # cast-weight cache from spanning the in-place base-weight update.
    with torch.inference_mode():
        unmerged_logits = model(parity_features)
        merge_lora(model.mit)
        merged_logits = model(parity_features)
    parity_error = float(
        (unmerged_logits - merged_logits).abs().max().cpu().item()
    )
    parity_tolerance = 1e-5
    if parity_error > parity_tolerance:
        unmerge_lora(model.mit)
        raise RuntimeError(
            "LoRA merge parity failed: max absolute error "
            f"{parity_error:.8f} > {parity_tolerance:.8f}"
        )
    save_checkpoint(
        merged_path,
        {
            "base_model_id": args.model_name,
            "mit_state_dict": merged_base_state_dict(model.mit),
            "classifier_state_dict": model.classifier.state_dict(),
            "class_ids": class_ids,
            "config": config,
            "best_epoch": best_checkpoint.epoch,
            "validation_accuracy": validation_accuracy,
            "teacher_provenance": teacher_provenance,
            "merge_parity_max_absolute_error": parity_error,
        },
    )
    unmerge_lora(model.mit)

    run_config = {
        "arguments": config,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "git_revision": _git_revision(),
        "device": str(device),
        "amp_dtype": str(amp_dtype).removeprefix("torch.") if amp_dtype else None,
        "effective_batch_size": effective_batch_size,
        "target_modules": model.target_modules,
        "test_evaluated": evaluate_test,
    }
    results = {
        **extraction_summary,
        "history": history,
        "best_epoch": best_checkpoint.epoch,
        "validation_accuracy": validation_accuracy,
        "validation_macro_f1": validation_metrics["macro_f1"],
        "validation_nll": validation_metrics["nll"],
        "validation_ece": validation_metrics["ece"],
        "validation_predictions": validation_predictions,
        "test_evaluated": evaluate_test,
        "trainable_parameter_audit": audit_payload,
        "adapter_checkpoint_path": str(adapter_path),
        "merged_checkpoint_path": str(merged_path),
        "merge_parity_max_absolute_error": parity_error,
        "amp_dtype": run_config["amp_dtype"],
        "micro_batch_size": args.batch_size,
        "effective_batch_size": effective_batch_size,
        "training_elapsed_seconds": time.perf_counter() - training_started,
        "peak_memory_allocated_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0
        ),
    }
    if evaluate_test:
        results["final_test_accuracy"] = final_accuracy
        results["predictions"] = predictions
    write_json(artifacts_dir / "run_config.json", run_config)
    write_json(artifacts_dir / "evaluation_results.json", results)
    write_json(
        artifacts_dir / "validation_predictions.json",
        validation_predictions,
    )
    torch.save(
        {
            "sample_ids": [sample.video_path.stem.removesuffix("_rgb") for sample in cached["val"]],
            "labels": validation_labels,
            "logits": validation_logits,
        },
        artifacts_dir / "validation_logits.pt",
    )
    write_json(
        artifacts_dir / "progress.json",
        {
            "state": "completed",
            "completed_epoch": len(history),
            "total_epochs": args.epochs,
            "best_epoch": best_checkpoint.epoch,
            "best_validation_accuracy": validation_accuracy,
            "test_evaluated": evaluate_test,
        },
    )
    save_accuracy_plot(
        history,
        artifacts_dir / "accuracy_graph.png",
        train_key="train_accuracy",
        evaluation_key="val_accuracy",
        evaluation_label="Validation Accuracy",
        title="XCLIP MIT LoRA + CE Training and Validation Accuracy",
    )
    if print_results:
        print(json.dumps(results, indent=2))
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train MIT LoRA over cached pre-MIT XCLIP features."
    )
    parser.add_argument("--videos-dir", default="dataset/subset264")
    parser.add_argument(
        "--config",
        default=None,
        help="JSON configuration produced by xclip-lora-tune.",
    )
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts/xclip_lora_ce",
    )
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument(
        "--model-name",
        default="microsoft/xclip-base-patch32",
    )
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--feature-batch-size", type=int, default=1)
    parser.add_argument("--decode-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument(
        "--preload-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--train-eval-frequency", type=int, default=1)
    parser.add_argument("--epoch-log-frequency", type=int, default=1)
    parser.add_argument(
        "--batch-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cache-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--lora-learning-rate", type=float, default=1e-4)
    parser.add_argument("--head-learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--frame-dropout", type=float, default=0.0)
    parser.add_argument("--temporal-jitter", type=int, default=0)
    parser.add_argument("--feature-noise-std", type=float, default=0.0)
    parser.add_argument("--teacher-cache", default=None)
    parser.add_argument(
        "--teacher-cache-validation",
        default="artifacts/skeletonclip_phase2/cache_validation.json",
    )
    parser.add_argument(
        "--class-mapping",
        default="artifacts/skeletonclip_phase2/class_mapping.json",
    )
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--kd-weight", type=float, default=0.0)
    parser.add_argument(
        "--evaluate-test",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument(
        "--resume-training",
        dest="resume_training",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--class-ids", type=int, nargs="+", default=None)
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument("--max-videos-per-class", type=int, default=None)
    limits.add_argument("--videos-per-split", type=int, default=None)
    parser.add_argument("--test-fraction", type=float, default=1 / 6)
    parser.add_argument("--val-fraction", type=float, default=1 / 6)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    arguments = list(argv) if argv is not None else None
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config")
    selected, _ = config_parser.parse_known_args(arguments)
    parser = build_parser()
    if selected.config:
        config_path = Path(selected.config)
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("XCLIP LoRA config must be a JSON object")
        valid = {action.dest for action in parser._actions}
        unknown = sorted(set(payload) - valid)
        if unknown:
            raise ValueError(
                "Unknown XCLIP LoRA config keys: " + ", ".join(unknown)
            )
        parser.set_defaults(**payload)
    return parser.parse_args(arguments)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_pipeline(args, evaluate_test=args.evaluate_test)


if __name__ == "__main__":
    main()
