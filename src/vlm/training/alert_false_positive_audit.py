"""CPU-only audit of which source actions trigger attention-event false alarms.

Reads saved validation ``view_predictions.pt`` files, by default the locked final
configuration (crop-aware checkpoint A on center_zoom70 plus person-crop-trained
checkpoint B on person crops), reuses the shared calibration/evaluation halves, and
breaks false positives down by true NTU action so bending and face-touch look-alikes
become visible. It can also export the evaluation-half clips the final view flags,
which is the input distribution of the second-stage VLM verifier. Validation only;
thresholds are exploratory and the test split is never read.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from vlm.reporting import write_json
from vlm.training.rgb_input_audit import ATTENTION_IDS, action_names, binary_metrics, calibration_mask, recall_threshold

TARGETS = {"attention_group": ATTENTION_IDS, "falling": (43,), "nausea_vomiting": (48,)}
RECALL_TARGETS = (0.95, 0.98)
FINAL_VIEW = "A.center_zoom70+B.person_crop"
DEFAULT_SOURCES = ["A=artifacts/person_crop_stage0/evaluation/view_predictions.pt",
                   "B=artifacts/patch16_person_crop/seed42/final_view_predictions.pt"]


def load_sources(specs: list[str]) -> tuple[dict[str, torch.Tensor], torch.Tensor, list[str]]:
    """Load ``NAME=path`` prediction files into ``NAME.view`` logits after checking they align."""
    logits, labels, ids = {}, None, None
    for spec in specs:
        name, _, path = spec.partition("=")
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("test_evaluated") is not False:
            raise ValueError(f"Refusing {path}: predictions must be validation-only")
        if labels is None:
            labels, ids = payload["labels"], list(payload["sample_ids"])
        elif list(payload["sample_ids"]) != ids or not payload["labels"].equal(labels):
            raise ValueError(f"{path} does not align with the first source")
        logits |= {f"{name}.{view}": value.float() for view, value in payload["logits"].items()}
    return logits, labels, ids


def view_probabilities(logits: dict[str, torch.Tensor], views: list[str]) -> dict[str, torch.Tensor]:
    """A view is one saved ``NAME.view`` or a ``+``-joined equal-weight softmax mean of several."""
    probabilities = {}
    for view in views:
        parts = view.split("+")
        missing = [part for part in parts if part not in logits]
        if missing:
            raise ValueError(f"Views {missing} not available; saved views are {sorted(logits)}")
        probabilities[view] = sum(logits[part].softmax(-1) for part in parts) / len(parts)
    return probabilities


def false_positive_sources(prediction, truth, labels, names, top_k):
    sources = labels[prediction & ~truth]
    counts = torch.bincount(sources, minlength=120)
    values, indices = counts.topk(top_k)
    return [{"class_id": int(i) + 1, "name": names[int(i) + 1], "false_positives": int(v),
             "attention_event": int(i) + 1 in ATTENTION_IDS}
            for v, i in zip(values.tolist(), indices.tolist()) if v > 0]


def event_score(probs, class_ids):
    return probs[:, torch.tensor([cls - 1 for cls in class_ids])].sum(-1)


def audit(probabilities, labels, ids, names, top_k):
    cal = calibration_mask(labels, ids)
    evaluation = ~cal
    report = {}
    for view, probs in probabilities.items():
        report[view] = {}
        for target, class_ids in TARGETS.items():
            score, truth = event_score(probs, class_ids), torch.isin(labels + 1, torch.tensor(class_ids))
            operating_points = []
            for recall in RECALL_TARGETS:
                threshold = recall_threshold(score[cal], truth[cal], recall)
                prediction = score[evaluation] >= threshold
                operating_points.append({
                    "calibration_recall_target": recall,
                    **binary_metrics(score[evaluation], truth[evaluation], threshold),
                    "false_positive_sources": false_positive_sources(
                        prediction, truth[evaluation], labels[evaluation], names, top_k)})
            report[view][target] = operating_points
    return report


def flagged_clips(probs, labels, ids, recall):
    """Evaluation-half clips whose attention-group score passes the calibration-half recall threshold."""
    cal = calibration_mask(labels, ids)
    score, truth = event_score(probs, ATTENTION_IDS), torch.isin(labels + 1, torch.tensor(ATTENTION_IDS))
    threshold = recall_threshold(score[cal], truth[cal], recall)
    evaluation = ~cal
    clips = [{"id": ids[i], "class_id": int(labels[i]) + 1, "event_score": float(score[i]),
              "stage1_prediction": int(probs[i].argmax()) + 1,
              "group": "true_alert" if truth[i] else "false_alert"}
             for i in (evaluation & (score >= threshold)).nonzero().flatten().tolist()]
    return {"view": FINAL_VIEW, "target": "attention_group", "calibration_recall_target": recall,
            "threshold": threshold, "evaluation_positives": int((evaluation & truth).sum()),
            "evaluation_samples": int(evaluation.sum()), "clips": clips}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES, help="NAME=path validation prediction files")
    parser.add_argument("--views", nargs="+", default=[FINAL_VIEW, "A.person_crop", "B.person_crop"],
                        help="NAME.view or '+'-joined softmax means, e.g. A.whole+A.center_zoom70")
    parser.add_argument("--action-names", type=Path, default=Path("docs/xclip-vision-lora-validation-action-accuracy.md"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/alert_false_positive_audit/report.json"))
    parser.add_argument("--flagged-output", type=Path, default=None,
                        help="Also write the final view's evaluation-half flagged clips for the VLM verifier")
    parser.add_argument("--flagged-recall", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=15)
    args = parser.parse_args()
    logits, labels, ids = load_sources(args.sources)
    probabilities = view_probabilities(logits, args.views)
    report = {"sources": args.sources, "split": "validation calibration/evaluation halves", "test_evaluated": False,
              "scope": "acted NTU clips; false-positive sources are look-alike actions, not real-world false alarms per hour",
              "views": audit(probabilities, labels, ids, action_names(args.action_names), args.top_k)}
    write_json(args.output, report)
    for view, targets in report["views"].items():
        for target, points in targets.items():
            for point in points:
                top = ", ".join(f"A{row['class_id']:03d}x{row['false_positives']}" for row in point["false_positive_sources"][:5])
                print(f"{view:<30} {target:<16} R@{point['calibration_recall_target']:.2f} th={point['threshold']:.4f} "
                      f"recall={point['recall']:.3f} precision={point['precision']:.3f} fp={point['fp']} top: {top}")
    print(f"Wrote {args.output}")
    if args.flagged_output:
        final = view_probabilities(logits, [FINAL_VIEW])[FINAL_VIEW]
        flagged = flagged_clips(final, labels, ids, args.flagged_recall)
        write_json(args.flagged_output, flagged)
        groups = [clip["group"] for clip in flagged["clips"]]
        print(f"Wrote {len(groups)} flagged clips ({groups.count('true_alert')} true, "
              f"{groups.count('false_alert')} false) to {args.flagged_output}")


if __name__ == "__main__":
    main()
