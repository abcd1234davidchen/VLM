import json

from vlm.training.skeletonclip import _write_bakeoff_report


def _result(backbone: str, seed: int, macro_f1: float, speed: float) -> dict[str, object]:
    return {
        "status": "complete",
        "backbone": backbone,
        "seed": seed,
        "epochs": 20,
        "best_epoch": 20,
        "best_macro_f1": macro_f1,
        "best_accuracy": macro_f1,
        "examples_per_second": speed,
        "peak_memory_allocated_bytes": 100,
        "history": [{"epoch": 20, "validation": {"macro_f1": macro_f1, "accuracy": macro_f1, "nll": 1.0, "ece": 0.1}}],
    }


def test_bakeoff_report_aggregates_separate_runs_and_eliminates_dominated_backbone(tmp_path) -> None:
    values = {
        ("blockgcn", 42): (0.86, 150.0),
        ("blockgcn", 43): (0.85, 152.0),
        ("hyperformer", 42): (0.77, 70.0),
        ("hyperformer", 43): (0.69, 72.0),
    }
    for (backbone, seed), (score, speed) in values.items():
        run_dir = tmp_path / f"{backbone}_seed{seed}"
        run_dir.mkdir()
        (run_dir / "result.json").write_text(json.dumps(_result(backbone, seed, score, speed)))

    report = _write_bakeoff_report(artifacts_dir=tmp_path, manifest=tmp_path / "manifest.json")

    assert report["status"] == "complete"
    assert len(report["results"]) == 4
    assert report["aggregates"]["blockgcn"]["macro_f1"]["mean"] == 0.855
    assert report["finalists"] == ["blockgcn"]
    assert json.loads((tmp_path / "summary.json").read_text()) == report
    assert json.loads((tmp_path / "bakeoff_report.json").read_text()) == report
