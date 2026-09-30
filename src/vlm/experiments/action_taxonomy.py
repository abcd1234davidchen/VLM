"""Audit the reduced action taxonomy against saved 120-class logits."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor

from vlm.action_taxonomy import (
    ActionTaxonomy,
    aggregate_source_logits,
    build_reduced_action_taxonomy,
    remap_source_labels,
)
from vlm.reporting import write_json
from vlm.skeleton_teacher_cache import sha256_file


def evaluate_taxonomy(
    source_logits: Tensor,
    source_labels: Tensor,
    taxonomy: ActionTaxonomy,
) -> tuple[dict[str, object], Tensor, Tensor]:
    output_logits = aggregate_source_logits(source_logits.float(), taxonomy)
    output_labels = remap_source_labels(source_labels.long(), taxonomy)
    predictions = output_logits.argmax(dim=1)
    output_ids = torch.tensor(taxonomy.output_class_ids)
    predicted_ids = output_ids[predictions]
    true_ids = output_ids[output_labels]
    irrelevant_ids = torch.tensor(taxonomy.irrelevant_output_ids)
    true_irrelevant = torch.isin(true_ids, irrelevant_ids)
    predicted_irrelevant = torch.isin(predicted_ids, irrelevant_ids)
    exact = predictions.eq(output_labels)
    public_correct = torch.where(
        true_irrelevant,
        predicted_irrelevant,
        exact,
    )
    relevant = ~true_irrelevant
    report: dict[str, object] = {
        "schema_version": 1,
        "samples": int(source_labels.numel()),
        "source_classes": len(taxonomy.source_class_ids),
        "output_classes": len(taxonomy.output_class_ids),
        "named_outputs": len(taxonomy.output_class_ids) - len(taxonomy.irrelevant_output_ids),
        "internal_irrelevant_outputs": len(taxonomy.irrelevant_output_ids),
        "exact_106_class_accuracy": float(exact.float().mean()),
        "public_accuracy": float(public_correct.float().mean()),
        "relevant_named_accuracy": float(exact[relevant].float().mean()),
        "irrelevant_public_recall": float(predicted_irrelevant[true_irrelevant].float().mean()),
        "irrelevant_internal_group_accuracy": float(exact[true_irrelevant].float().mean()),
        "relevant_false_irrelevant_rate": float(predicted_irrelevant[relevant].float().mean()),
        "relevant_samples": int(relevant.sum()),
        "irrelevant_samples": int(true_irrelevant.sum()),
    }
    per_group = {}
    for output_id in taxonomy.irrelevant_output_ids:
        mask = true_ids.eq(output_id)
        per_group[str(output_id)] = {
            "name": taxonomy.output_names[output_id],
            "samples": int(mask.sum()),
            "exact_group_accuracy": float(predicted_ids[mask].eq(output_id).float().mean()),
            "public_irrelevant_recall": float(predicted_irrelevant[mask].float().mean()),
        }
    report["per_irrelevant_group"] = per_group
    return report, predicted_ids, true_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the 103-named + 3-internal-irrelevant taxonomy using saved logits."
    )
    parser.add_argument(
        "--logits",
        type=Path,
        default=Path(
            "artifacts/skeletonclip_phase3/vision_batch16_workers24_uint8_seed42/"
            "vision_lora_ce_seed42/best_validation_logits.pt"
        ),
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts/action_taxonomy_audit"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = torch.load(args.logits, map_location="cpu", weights_only=True)
    logits, labels = payload.get("logits"), payload.get("labels")
    if not isinstance(logits, Tensor) or not isinstance(labels, Tensor):
        raise ValueError("Logit payload must contain tensor logits and labels")
    taxonomy = build_reduced_action_taxonomy()
    report, predicted_ids, true_ids = evaluate_taxonomy(logits, labels, taxonomy)
    report.update(
        {
            "status": "offline_audit_complete",
            "logits_path": str(args.logits),
            "logits_sha256": sha256_file(args.logits),
            "test_evaluated": False,
            "limitation": (
                "Probability-mass aggregation of a frozen 120-class model is an offline "
                "feasibility audit, not a trained reduced-taxonomy classifier."
            ),
            "taxonomy": taxonomy.as_dict(),
        }
    )
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.artifacts_dir / "audit_report.json", report)
    torch.save(
        {
            "sample_ids": payload.get("sample_ids"),
            "true_output_ids": true_ids,
            "predicted_output_ids": predicted_ids,
        },
        args.artifacts_dir / "audit_predictions.pt",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
