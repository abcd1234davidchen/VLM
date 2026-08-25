"""XCLIP MIT LoRA + CE-only training pipeline."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
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
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    scaler_enabled = amp_enabled and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    total_loss = 0.0
    batches = tqdm(loader, desc=f"Epoch {epoch} Training", leave=False)
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
        batches.set_postfix(loss=f"{batch_loss:.4f}")
    return total_loss / len(loader.dataset)


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
) -> dict[str, object]:
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
    cached = prepared_cached
    if cached is None:
        cached = extract_or_load_xclip_pre_mit_features(
            split,
            cache_dir=cache_dir,
            model_name=args.model_name,
            num_frames=args.frames,
            device=device,
            force_extract=args.force_extract,
            feature_batch_size=args.feature_batch_size,
            decode_workers=args.decode_workers,
            read_only=cache_read_only,
        )
    expected_counts = {name: len(samples) for name, samples in split.items()}
    actual_counts = {name: len(samples) for name, samples in cached.items()}
    if actual_counts != expected_counts:
        raise ValueError(
            "Prepared XCLIP cache split counts do not match the manifest"
        )
    extraction_summary = {
        "device": str(device),
        "train_samples": len(cached["train"]),
        "val_samples": len(cached["val"]),
        "test_samples": len(cached["test"]),
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
    best_checkpoint = ValidationCheckpoint()
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        loss = train_xclip_lora_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            max_grad_norm=args.max_grad_norm,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            epoch=epoch,
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
            }
        )
        train_accuracy_text = (
            f"{train_accuracy:.4f}"
            if train_accuracy is not None
            else "n/a"
        )
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
    validation_accuracy, _ = evaluate_features(
        model,
        val_loader,
        cached["val"],
        device,
        class_ids,
    )
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
        "test_evaluated": evaluate_test,
        "trainable_parameter_audit": audit_payload,
        "adapter_checkpoint_path": str(adapter_path),
        "merged_checkpoint_path": str(merged_path),
        "merge_parity_max_absolute_error": parity_error,
        "amp_dtype": run_config["amp_dtype"],
        "micro_batch_size": args.batch_size,
        "effective_batch_size": effective_batch_size,
    }
    if evaluate_test:
        results["final_test_accuracy"] = final_accuracy
        results["predictions"] = predictions
    write_json(artifacts_dir / "run_config.json", run_config)
    write_json(artifacts_dir / "evaluation_results.json", results)
    save_accuracy_plot(
        history,
        artifacts_dir / "accuracy_graph.png",
        train_key="train_accuracy",
        evaluation_key="val_accuracy",
        evaluation_label="Validation Accuracy",
        title="XCLIP MIT LoRA + CE Training and Validation Accuracy",
    )
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
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
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
    run_pipeline(parse_args(argv))


if __name__ == "__main__":
    main()
