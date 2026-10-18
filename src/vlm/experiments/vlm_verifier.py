"""Zero-shot local VLM bake-off for the second-stage alert verifier.

The verifier only sees clips the stage-1 classifier already flagged, so ``select`` samples
from the evaluation-half flagged export of ``vlm.training.alert_false_positive_audit``.
Each answer becomes a tier: ``confirmed`` (an alert option), ``dismissed`` (a normal
activity) or ``uncertain`` (the explicit cannot-tell option or an unparseable answer).
Uncertain is passed on as an alert, so only ``dismissed`` can lower an alert's priority.

Self-contained on purpose: it runs in an isolated mlx-vlm environment whose
transformers version differs from the training venv, so it must not import torch
or the rest of the ``vlm`` package.

    uv run --no-project --python 3.13 --with mlx-vlm==0.7.8 --with opencv-python \
        python src/vlm/experiments/vlm_verifier.py select --flagged artifacts/vlm_verifier/stage1_flagged.json \
        --split artifacts/xclip/selected_split.json
    uv run --no-project --python 3.13 --with mlx-vlm==0.7.8 --with opencv-python \
        python src/vlm/experiments/vlm_verifier.py run --model mlx-community/Qwen3.5-9B-MLX-4bit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2

EVENTS = {42: "staggering", 43: "falling", 44: "headache", 45: "chest_pain", 46: "back_pain", 47: "neck_pain",
          48: "nausea_vomiting"}

# Letter -> (option text, alert event id or None for a normal activity). "U" is the uncertain tier.
CHOICES = {
    "A": ("fell down or is lying on the floor after a fall", 43),
    "B": ("vomiting or about to vomit (retching, hand over mouth while bending forward)", 48),
    "C": ("staggering or losing balance", 42),
    "D": ("holding their head in pain (headache)", 44),
    "E": ("holding their chest or stomach in pain", 45),
    "F": ("holding their back in pain", 46),
    "G": ("holding their neck in pain", 47),
    "H": ("picking up or putting down an object, or putting on or taking off shoes", None),
    "I": ("sitting down, standing up, squatting, jumping, or hopping", None),
    "J": ("grooming or touching the face or hair normally (combing hair, wiping face, yawning, blowing nose)", None),
    "K": ("exercising, stretching, or bending on purpose", None),
    "L": ("another normal everyday activity", None),
    "U": ("cannot tell from these frames", None),
}

PROMPT = (
    "You are checking a home-safety camera. The images are {n} frames sampled in time order from one short clip "
    "of a person. An automatic detector raised a possible health alert for this clip. Judge only what is visible; "
    "people often bend, crouch, or touch their face or body during normal activity.\n"
    "Which option best describes what the person is doing?\n"
    + "\n".join(f"{key}. {text}" for key, (text, _) in CHOICES.items())
    + "\nAnswer with JSON only, no other text:\n"
    '{{"choice": "<letter>", "person_on_floor_at_end": true|false, "confidence": <0.0-1.0>, '
    '"evidence": "<one short sentence>"}}'
)


def _stable_rank(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"verifier{seed}:{sample_id}".encode()).hexdigest()


def tier(choice: str | None) -> str:
    if choice is None or choice == "U":
        return "uncertain"
    return "confirmed" if CHOICES[choice][1] is not None else "dismissed"


def select_clips(flagged_path: Path, split_path: Path, seed: int, true_per_event: int) -> list[dict[str, object]]:
    """Sample flagged true alerts per event and keep every flagged false alert; validation only."""
    flagged = json.loads(flagged_path.read_text())
    paths = {Path(record["video_path"]).stem.removesuffix("_rgb"): record["video_path"]
             for record in json.loads(split_path.read_text())["val"]}
    by_event: dict[int, list[dict]] = defaultdict(list)
    chosen = []
    for clip in flagged["clips"]:
        (by_event[clip["class_id"]] if clip["group"] == "true_alert" else chosen).append(clip)
    for event in sorted(by_event):
        candidates = sorted(by_event[event], key=lambda clip: _stable_rank(seed, clip["id"]))
        if len(candidates) < true_per_event:
            raise ValueError(f"A{event:03d} has only {len(candidates)} flagged true alerts")
        chosen.extend(candidates[:true_per_event])
    return [clip | {"server_path": paths[clip["id"]]} for clip in sorted(chosen, key=lambda clip: clip["id"])]


def extract_frames(video: Path, out_dir: Path, num_frames: int, max_side: int) -> list[Path]:
    """Uniformly sample whole, uncropped frames and save them as JPEGs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = [out_dir / f"{index:02d}.jpg" for index in range(num_frames)]
    if all(frame.exists() for frame in frames):
        return frames
    capture = cv2.VideoCapture(str(video))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        raise ValueError(f"Cannot read frames from {video}")
    indices = [round(i * (total - 1) / max(num_frames - 1, 1)) for i in range(num_frames)]
    for index, frame_path in zip(indices, frames):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = capture.read()
        if not ok:
            raise ValueError(f"Failed to decode frame {index} of {video}")
        scale = max_side / max(image.shape[:2])
        if scale < 1:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(frame_path), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    capture.release()
    return frames


def parse_answer(text: str) -> dict[str, object] | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        return None
    try:
        answer = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    choice = str(answer.get("choice", "")).strip().upper()[:1]
    return answer | {"choice": choice} if choice in CHOICES else None


def summarize(results: list[dict[str, object]]) -> dict[str, object]:
    def share(rows, predicate):
        return sum(bool(predicate(row)) for row in rows) / len(rows) if rows else None

    true_alerts = [row for row in results if row["group"] == "true_alert"]
    false_alerts = [row for row in results if row["group"] == "false_alert"]
    per_event = {}
    for event, name in EVENTS.items():
        rows = [row for row in true_alerts if row["class_id"] == event]
        per_event[name] = {"clips": len(rows), "dismissed": sum(row["tier"] == "dismissed" for row in rows),
                           "exact_event": sum(row["choice"] is not None and CHOICES[row["choice"]][1] == event
                                              for row in rows)}
    false_sources = defaultdict(Counter)
    for row in false_alerts:
        false_sources[f"A{row['class_id']:03d}"][row["tier"]] += 1
    latencies = sorted(row["latency_s"] for row in results)
    return {
        "clips": len(results), "true_alerts": len(true_alerts), "false_alerts": len(false_alerts),
        "valid_json_rate": share(results, lambda row: row["answer"] is not None),
        "true_alert_retained_rate": share(true_alerts, lambda row: row["tier"] != "dismissed"),
        "true_alert_confirmed_rate": share(true_alerts, lambda row: row["tier"] == "confirmed"),
        "false_alert_dismissed_rate": share(false_alerts, lambda row: row["tier"] == "dismissed"),
        "false_alert_confirmed_rate": share(false_alerts, lambda row: row["tier"] == "confirmed"),
        "uncertain_rate": share(results, lambda row: row["tier"] == "uncertain"),
        "per_event": per_event,
        "false_alert_tiers_by_source": {source: dict(counts) for source, counts in sorted(false_sources.items())},
        "latency_s_median": latencies[len(latencies) // 2],
        "latency_s_p90": latencies[int(0.9 * (len(latencies) - 1))],
        "peak_memory_gb_max": max(row["peak_memory_gb"] for row in results),
    }


def run(args: argparse.Namespace) -> None:
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    clips = json.loads(args.manifest.read_text())
    if args.limit:
        clips = clips[: args.limit]
    out_dir = args.artifacts_dir / args.model.split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / ("smoke_rows.jsonl" if args.limit else "rows.jsonl")
    done = {}
    if rows_path.exists():
        done = {row["id"]: row for row in map(json.loads, rows_path.read_text().splitlines())}
    model, processor = load(args.model)
    prompt_text = PROMPT.format(n=args.num_frames)
    prompt = apply_chat_template(processor, model.config, prompt_text, num_images=args.num_frames,
                                 enable_thinking=False)
    results = []
    with rows_path.open("a") as sink:
        for number, clip in enumerate(clips, 1):
            if clip["id"] in done:
                results.append(done[clip["id"]])
                continue
            frames = extract_frames(args.clips_dir / Path(clip["server_path"]).name,
                                    args.frames_dir / f"{args.num_frames}x{args.max_side}" / clip["id"],
                                    args.num_frames, args.max_side)
            start = time.perf_counter()
            output = generate(model, processor, prompt, image=[str(frame) for frame in frames], max_tokens=160,
                              temperature=0.0, enable_thinking=False, verbose=False)
            latency = time.perf_counter() - start
            answer = parse_answer(output.text)
            choice = answer["choice"] if answer else None
            row = clip | {"choice": choice, "tier": tier(choice), "answer": answer, "raw": output.text,
                          "latency_s": round(latency, 2), "prompt_tokens": output.prompt_tokens,
                          "peak_memory_gb": round(output.peak_memory, 2)}
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            results.append(row)
            print(f"[{number}/{len(clips)}] {clip['id']} {clip['group']:<11} -> {choice} {row['tier']:<9} "
                  f"{latency:.1f}s", flush=True)
    report = {"model": args.model, "num_frames": args.num_frames, "max_side": args.max_side, "smoke": bool(args.limit),
              "prompt": prompt_text, "summary": summarize(results)}
    (out_dir / ("smoke_report.json" if args.limit else "report.json")).write_text(
        json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report["summary"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select", help="Sample a validation-only clip manifest from the stage-1 flagged export")
    select.add_argument("--flagged", type=Path, default=Path("artifacts/vlm_verifier/stage1_flagged.json"))
    select.add_argument("--split", type=Path, required=True)
    select.add_argument("--seed", type=int, default=42)
    select.add_argument("--true-per-event", type=int, default=8)
    select.add_argument("--output", type=Path, default=Path("artifacts/vlm_verifier/manifest.json"))
    bake = sub.add_parser("run", help="Ask one local VLM to verify every manifest clip")
    bake.add_argument("--model", required=True)
    bake.add_argument("--manifest", type=Path, default=Path("artifacts/vlm_verifier/manifest.json"))
    bake.add_argument("--clips-dir", type=Path, default=Path("cache/vlm_verifier_clips"))
    bake.add_argument("--frames-dir", type=Path, default=Path("cache/vlm_verifier_frames"))
    bake.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/vlm_verifier"))
    bake.add_argument("--num-frames", type=int, default=8)
    bake.add_argument("--max-side", type=int, default=768)
    bake.add_argument("--limit", type=int, default=0, help="Only run the first N clips (smoke)")
    args = parser.parse_args()
    if args.command == "select":
        clips = select_clips(args.flagged, args.split, args.seed, args.true_per_event)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(clips, indent=2))
        groups = Counter(clip["group"] for clip in clips)
        print(f"Wrote {len(clips)} clips ({dict(groups)}) to {args.output}")
    else:
        run(args)


if __name__ == "__main__":
    main()
