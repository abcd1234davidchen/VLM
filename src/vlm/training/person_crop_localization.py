"""RGB-only person crop localization for the person-aware crop experiments.

Runs MediaPipe PoseLandmarker on the classifier's uniform 8 frames of every canonical
train or validation video and derives one clip-static square crop per sample. Labels and skeletons
are never read. Contract: docs/person-aware-crop-stage0-plan.md section 4.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import av
import numpy as np

from vlm.experiments.video_sampling import read_video_pyav, uniform_frame_indices
from vlm.reporting import write_json
from vlm.skeleton_teacher_cache import sha256_file

CONFIDENCE = 0.5
NUM_POSES = 2
MIN_FRAME_LANDMARKS = 8
MIN_VALID_FRAMES = 4
SIDE_SCALE, MIN_SIDE_FRACTION = 1.4, 0.5

_DETECTOR = None
_MP = None


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, payload)
    temporary.replace(path)


def center_zoom70_box(width: int, height: int) -> list[int]:
    """[left, top, right, bottom] of rgb_input_audit.ViewDataset center_zoom70."""
    side = round(min(height, width) * .7)
    top, left = (height - side) // 2, (width - side) // 2
    return [left, top, left + side, top + side]


def person_square(frames: list[list[list[list[float]]]], width: int, height: int) -> tuple[list[int], list[float] | None, str | None]:
    """Clip-static square from per-frame, per-person usable landmarks; center_zoom70 fallback."""
    valid = [[point for person in persons for point in person] for persons in frames]
    valid = [points for points in valid if len(points) >= MIN_FRAME_LANDMARKS]
    if len(valid) < MIN_VALID_FRAMES:
        return center_zoom70_box(width, height), None, f"{len(valid)} valid frames (< {MIN_VALID_FRAMES})"
    points = np.asarray([point for points in valid for point in points], dtype=np.float64)
    low, high = points.min(0), points.max(0)
    short = min(width, height)
    side = min(max(round(SIDE_SCALE * float((high - low).max())), round(MIN_SIDE_FRACTION * short)), short)
    center = (low + high) / 2
    left = min(max(math.floor(center[0] - side / 2), 0), width - side)
    top = min(max(math.floor(center[1] - side / 2), 0), height - side)
    return [left, top, left + side, top + side], [*low.tolist(), *high.tolist()], None


def init_detector(model_path: str) -> None:
    global _DETECTOR, _MP
    import cv2
    import mediapipe as mp
    cv2.setNumThreads(1)
    _MP = mp
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_poses=NUM_POSES,
        min_pose_detection_confidence=CONFIDENCE, min_pose_presence_confidence=CONFIDENCE,
        min_tracking_confidence=CONFIDENCE)
    _DETECTOR = mp.tasks.vision.PoseLandmarker.create_from_options(options)


def localize(task: tuple[str, str]) -> dict[str, object]:
    identifier, video_path = task
    with av.open(video_path) as container:
        indices = uniform_frame_indices(8, container.streams.video[0].frames)
        frames = read_video_pyav(container, indices)
    height, width = frames.shape[1:3]
    poses = []
    for frame in frames:
        result = _DETECTOR.detect(_MP.Image(image_format=_MP.ImageFormat.SRGB, data=np.ascontiguousarray(frame)))
        persons = []
        for landmarks in result.pose_landmarks:
            persons.append([[round(p.x * width, 1), round(p.y * height, 1)] for p in landmarks
                            if min(p.visibility, p.presence) >= CONFIDENCE and math.isfinite(p.x) and math.isfinite(p.y)
                            and 0 <= p.x < 1 and 0 <= p.y < 1])
        poses.append(persons)
    crop, box, fallback = person_square(poses, width, height)
    return {"sample_id": identifier, "video_path": video_path, "rgb_sha256": sha256_file(Path(video_path)),
            "width": int(width), "height": int(height), "frame_indices": indices.tolist(), "poses": poses,
            "landmark_box": box, "crop": crop, "fallback": fallback}


def run(args: argparse.Namespace) -> dict[str, object]:
    import mediapipe as mp
    split = json.loads(args.split.read_text(encoding="utf-8"))
    rows = split[args.subset][:args.max_samples]
    tasks = [(Path(row["video_path"]).stem.removesuffix("_rgb"), str(row["video_path"])) for row in rows]
    if len({identifier for identifier, _ in tasks}) != len(tasks):
        raise ValueError(f"Duplicate {args.subset} sample IDs")
    root = args.artifacts_dir
    lock = {
        "experiment": "person_crop_stage0_localization", "split": args.subset, "samples": len(tasks),
        "sample_ids_sha256": hashlib.sha256("\n".join(i for i, _ in tasks).encode()).hexdigest(),
        "source_sha256": {"split": sha256_file(args.split), "pose_model": sha256_file(args.pose_model),
                          "runner": sha256_file(Path(__file__)),
                          "crop_definition": hashlib.sha256(inspect.getsource(person_square).encode()).hexdigest()},
        "mediapipe_version": mp.__version__, "running_mode": "IMAGE", "num_poses": NUM_POSES,
        "detection_confidence": CONFIDENCE, "presence_confidence": CONFIDENCE, "landmark_visibility_presence": CONFIDENCE,
        "min_frame_landmarks": MIN_FRAME_LANDMARKS, "min_valid_frames": MIN_VALID_FRAMES,
        "side_scale": SIDE_SCALE, "min_side_fraction": MIN_SIDE_FRACTION,
        "fallback": "center_zoom70 box", "inputs": "RGB frames only; labels and skeletons not read", "test_evaluated": False,
    }
    lock_path, partial_path = root / "experiment_lock.json", root / "partial_crop_manifest.jsonl"
    done: dict[str, dict[str, object]] = {}
    if lock_path.exists():
        if json.loads(lock_path.read_text(encoding="utf-8")) != lock:
            raise ValueError("Existing lock differs; use a fresh artifacts directory")
        if partial_path.exists():
            for line in partial_path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    break  # torn final line from an interruption
                done[record["sample_id"]] = record
    else:
        _atomic_json(lock_path, lock)
    pending = [task for task in tasks if task[0] not in done]
    started = time.monotonic()
    with partial_path.open("a", encoding="utf-8") as partial, ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=init_detector, initargs=(str(args.pose_model),)) as executor:
        for count, record in enumerate(executor.map(localize, pending, chunksize=4), 1):
            done[record["sample_id"]] = record
            partial.write(json.dumps(record) + "\n")
            if count % 64 == 0 or count == len(pending):
                partial.flush()
                progress = {"state": "running", "samples": len(done), "total": len(tasks),
                            "seconds": round(time.monotonic() - started, 1)}
                _atomic_json(root / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    ordered = [done[identifier] for identifier, _ in tasks]
    manifest = root / "crop_manifest.json"
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"lock_sha256": sha256_file(lock_path), "samples": ordered}), encoding="utf-8")
    temporary.replace(manifest)
    report = {"state": "complete", "samples": len(ordered), "fallbacks": sum(r["fallback"] is not None for r in ordered),
              "seconds": round(time.monotonic() - started, 1), "test_evaluated": False}
    _atomic_json(root / "progress.json", report)
    print(json.dumps(report), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=Path("artifacts/xclip/selected_split.json"))
    parser.add_argument("--pose-model", type=Path, default=Path("artifacts/hand_roi_models/pose_landmarker_lite.task"))
    parser.add_argument("--subset", choices=("train", "val", "test"), default="val")
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
