"""Phase 4 Gate 0 audit of raw NTU color joints against Phase 3 RGB frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw
from transformers import XCLIPProcessor

from vlm.data.skeleton import _parse_body_metadata, _parse_count, _parse_floats, sha256_file
from vlm.experiments.video_sampling import uniform_frame_indices


REVISION = "a2e27a78a2b5d802e894b8a1ef14f3a8ce490963"
MODEL = "microsoft/xclip-base-patch32"


def color_joints(path: Path, wanted: set[int]) -> tuple[int, dict[int, list[dict[str, object]]]]:
    """Read color-image joint coordinates without using root-centered XYZ tensors."""
    selected = {}
    with path.open(encoding="utf-8") as stream:
        total = _parse_count(stream, path=path, context="frame count")
        for frame in range(total):
            bodies = []
            for body in range(_parse_count(stream, path=path, context=f"frame {frame} body count")):
                body_id, metadata = _parse_body_metadata(stream, path=path, context=f"frame {frame} body {body} metadata")
                count = _parse_count(stream, path=path, context=f"frame {frame} body {body} joint count")
                if count != 25:
                    raise ValueError(f"{path}: expected 25 joints, got {count}")
                joints = []
                for joint in range(count):
                    fields = _parse_floats(stream, path=path, context=f"frame {frame} body {body} joint {joint}", minimum_fields=12)
                    if frame in wanted:
                        joints.append({"joint": joint, "x": fields[5], "y": fields[6], "xyz": fields[:3], "state": int(fields[11])})
                if frame in wanted:
                    bodies.append({"body_id": body_id, "body_state": int(metadata[8]), "joints": joints})
            if frame in wanted:
                selected[frame] = bodies
    return total, selected


def transform_point(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    """Match VideoMAEImageProcessor shortest-edge resize and center crop."""
    scale = 224 / min(width, height)
    resized_width = int(width * scale)
    resized_height = int(height * scale)
    return x * resized_width / width - (resized_width - 224) // 2, y * resized_height / height - (resized_height - 224) // 2


def choose_rows(rows: list[dict], limit: int) -> list[dict]:
    ordered = sorted(rows, key=lambda row: hashlib.sha256(row["sample_id"].encode()).hexdigest())
    chosen, seen_classes, seen_setups, seen_cameras = [], set(), set(), set()
    while ordered and len(chosen) < limit:
        best = max(range(len(ordered)), key=lambda i: (
            4 * (ordered[i]["class_id"] not in seen_classes)
            + 2 * (int(ordered[i]["sample_id"][1:4]) not in seen_setups)
            + (int(ordered[i]["sample_id"][5:8]) not in seen_cameras), -i
        ))
        row = ordered.pop(best)
        chosen.append(row)
        seen_classes.add(row["class_id"])
        seen_setups.add(int(row["sample_id"][1:4]))
        seen_cameras.add(int(row["sample_id"][5:8]))
    return chosen


def decode_selected(container: av.container.InputContainer, indices: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    wanted = set(map(int, indices))
    found = {}
    container.seek(0)
    for index, frame in enumerate(container.decode(video=0)):
        if index > max(wanted):
            break
        if index in wanted:
            found[index] = (frame.to_ndarray(format="rgb24"), frame.pts,
                            float(frame.time) if frame.time is not None else None)
    if set(found) != wanted:
        raise ValueError(f"Missing decoded frames: {sorted(wanted - set(found))}")
    return np.stack([found[int(i)][0] for i in indices]), [
        {"index": int(i), "pts": found[int(i)][1], "decoded_seconds": found[int(i)][2]} for i in indices
    ]


def audit_row(row: dict, processor: XCLIPProcessor, output: Path, lock: dict) -> dict:
    video_path, skeleton_path = Path(row["rgb_path"]), Path(row["skeleton_path"])
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        video_count = stream.frames
        if video_count <= 0:
            raise ValueError(f"Unknown video frame count: {video_path}")
        indices = uniform_frame_indices(8, video_count)
        frames, timing = decode_selected(container, indices)
        fps = float(stream.average_rate) if stream.average_rate else None
        time_base = float(stream.time_base) if stream.time_base else None
    skeleton_count, selected = color_joints(skeleton_path, set(map(int, indices)))
    thresholds = lock["automated_thresholds"]
    summary = {"sample_id": row["sample_id"], "split": row["split"], "class_id": row["class_id"],
               "setup": int(row["sample_id"][1:4]), "camera": int(row["sample_id"][5:8]),
               "video_path": str(video_path), "skeleton_path": str(skeleton_path),
               "video_frames": video_count, "skeleton_frames": skeleton_count, "frame_count_difference": video_count - skeleton_count,
               "fps": fps, "time_base": time_base, "frame_indices": indices.tolist(), "frames": []}
    usable = 0
    for position, index in enumerate(indices):
        frame = frames[position]
        height, width = frame.shape[:2]
        processed = processor(images=[frame], return_tensors="np", do_normalize=False, do_rescale=False)["pixel_values"][0, 0]
        image = Image.fromarray(np.clip(processed.transpose(1, 2, 0), 0, 255).astype(np.uint8))
        clean = output / "clean" / row["split"] / row["sample_id"] / f"frame_{int(index):04d}.png"
        clean.parent.mkdir(parents=True, exist_ok=True)
        image.save(clean)
        draw = ImageDraw.Draw(image)
        bodies = selected.get(int(index), [])
        body_records = []
        tracked_in_crop = 0
        for body in bodies:
            joints = []
            for joint in body["joints"]:
                x, y = transform_point(joint["x"], joint["y"], width, height)
                inside = 0 <= x < 224 and 0 <= y < 224
                tracked = joint["state"] == 2
                if tracked and inside:
                    tracked_in_crop += 1
                    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="red")
                joints.append({**joint, "input_x": x, "input_y": y, "inside_crop": inside})
            body_records.append({"body_id": body["body_id"], "body_state": body["body_state"], "joints": joints})
        frame_usable = tracked_in_crop >= thresholds["min_in_crop_tracked_joints_per_frame"]
        usable += frame_usable
        overlay = output / "overlays" / row["split"] / row["sample_id"] / f"frame_{int(index):04d}.png"
        overlay.parent.mkdir(parents=True, exist_ok=True)
        image.save(overlay)
        summary["frames"].append({"index": int(index), "nominal_video_seconds": int(index) / fps if fps else None,
                                  "video_pts": timing[position]["pts"], "decoded_video_seconds": timing[position]["decoded_seconds"],
                                  "source_width": width, "source_height": height, "body_count": len(bodies),
                                  "tracked_in_crop": tracked_in_crop, "usable": frame_usable,
                                  "clean": str(clean), "overlay": str(overlay), "bodies": body_records})
    summary["usable_frame_fraction"] = usable / len(indices)
    summary["automated_usable"] = (abs(video_count - skeleton_count) <= thresholds["max_frame_count_difference"]
                                    and summary["usable_frame_fraction"] >= thresholds["min_usable_frame_fraction_per_sample"])
    return summary


def coverage_row(row: dict) -> dict:
    """Check all paired samples using container metadata and raw skeleton joints."""
    video_path, skeleton_path = Path(row["rgb_path"]), Path(row["skeleton_path"])
    result = {"sample_id": row["sample_id"], "split": row["split"], "class_id": row["class_id"],
              "setup": int(row["sample_id"][1:4]), "camera": int(row["sample_id"][5:8])}
    try:
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            count, width, height = stream.frames, stream.width, stream.height
        if count < 1 or width < 1 or height < 1:
            raise ValueError("Missing video frame count or dimensions")
        indices = uniform_frame_indices(8, count)
        skeleton_count, selected = color_joints(skeleton_path, set(map(int, indices)))
        per_frame = []
        for index in indices:
            bodies = selected.get(int(index), [])
            tracked = 0
            for body in bodies:
                for joint in body["joints"]:
                    x, y = transform_point(joint["x"], joint["y"], width, height)
                    tracked += joint["state"] == 2 and 0 <= x < 224 and 0 <= y < 224
            per_frame.append({"index": int(index), "body_count": len(bodies), "tracked_in_crop": tracked})
        result.update({"video_frames": count, "skeleton_frames": skeleton_count,
                       "frame_count_difference": count - skeleton_count, "frames": per_frame})
    except (OSError, ValueError, av.error.FFmpegError) as error:
        result["error"] = str(error)
    return result


def run_full_coverage(manifest: dict, output: Path, lock: dict, workers: int, *, lock_sha256: str, manifest_sha256: str) -> None:
    rows = manifest["train"] + manifest["val"]
    thresholds = lock["automated_thresholds"]
    output.mkdir(parents=True, exist_ok=True)
    counts = {key: {} for key in ("split", "class_id", "setup", "camera")}
    failures = Counter()
    total_usable = 0
    with ProcessPoolExecutor(max_workers=workers) as pool, (output / "full_coverage.jsonl").open("w") as stream:
        for position, result in enumerate(pool.map(coverage_row, rows, chunksize=16), 1):
            frames = result.get("frames", [])
            result["usable_frame_fraction"] = sum(f["tracked_in_crop"] >= thresholds["min_in_crop_tracked_joints_per_frame"] for f in frames) / 8
            result["automated_usable"] = ("error" not in result
                                          and abs(result["frame_count_difference"]) <= thresholds["max_frame_count_difference"]
                                          and result["usable_frame_fraction"] >= thresholds["min_usable_frame_fraction_per_sample"])
            total_usable += result["automated_usable"]
            if not result["automated_usable"]:
                reason = "error" if "error" in result else "frame_count" if abs(result["frame_count_difference"]) > thresholds["max_frame_count_difference"] else "joint_coverage"
                failures[reason] += 1
            for key, groups in counts.items():
                group = groups.setdefault(str(result[key]), {"paired": 0, "automated_usable": 0})
                group["paired"] += 1
                group["automated_usable"] += result["automated_usable"]
            stream.write(json.dumps(result) + "\n")
            if position % 1000 == 0:
                stream.flush()
                print(f"coverage {position}/{len(rows)}", flush=True)
    report = {"schema_version": 1, "status": "pending_manual_alignment_review", "test_evaluated": False,
              "lock_sha256": lock_sha256, "manifest_sha256": manifest_sha256,
              "paired": len(rows), "automated_usable": total_usable, "automated_usable_fraction": total_usable / len(rows),
              "threshold_met": total_usable / len(rows) >= thresholds["min_usable_sample_fraction"],
              "failure_reasons": dict(failures), "groups": counts,
              "note": "Metadata/frame-index screening only; image alignment still requires manual review."}
    (output / "full_coverage_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("paired", "automated_usable", "automated_usable_fraction", "failure_reasons")}, indent=2))


def temporal_panels(output: Path, annotations_path: Path, processor: XCLIPProcessor) -> None:
    records = {record["sample_id"]: record for filename in ("samples.jsonl", "targeted_a059.jsonl")
               for record in map(json.loads, (output / filename).open())}
    annotations = json.loads(annotations_path.read_text())["annotations"]
    panel_dir = output / "temporal_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    for item in annotations:
        record = records[item["sample_id"]]
        sampled = next(frame for frame in record["frames"] if frame["index"] == item["frame_index"])
        center = item["frame_index"]
        offsets = [-3, 0, 3]
        indices = np.array([center + delta for delta in offsets], dtype=np.int64)
        with av.open(record["video_path"]) as container:
            rgb, timings = decode_selected(container, indices)
        panel = Image.new("RGB", (224 * len(offsets), 252))
        for position, delta in enumerate(offsets):
            processed = processor(images=[rgb[position]], return_tensors="np", do_normalize=False, do_rescale=False)["pixel_values"][0, 0]
            image = Image.fromarray(np.clip(processed.transpose(1, 2, 0), 0, 255).astype(np.uint8))
            draw = ImageDraw.Draw(image)
            for body in sampled["bodies"]:
                for joint in body["joints"]:
                    if joint["state"] != 2 or not joint["inside_crop"]:
                        continue
                    x, y = joint["input_x"], joint["input_y"]
                    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="red")
            panel.paste(image, (224 * position, 28))
        title = ImageDraw.Draw(panel)
        for position, delta in enumerate(offsets):
            title.text((224 * position + 4, 5), f"RGB {center + delta}, skeleton {center} ({delta:+d})", fill="white")
        panel.save(panel_dir / f"{item['sample_id']}.png")


def evaluate_manual_alignment(output: Path, annotations_path: Path, lock: dict) -> dict:
    records = {record["sample_id"]: record for filename in ("samples.jsonl", "targeted_a059.jsonl")
               for record in map(json.loads, (output / filename).open())}
    annotations = json.loads(annotations_path.read_text())["annotations"]
    errors = []
    shifted = {shift: [] for shift in range(-3, 4)}
    moving = {shift: [] for shift in range(-3, 4)}
    detail = []
    for item in annotations:
        record = records[item["sample_id"]]
        center = item["frame_index"]
        frame = next(frame for frame in record["frames"] if frame["index"] == center)
        width, height = frame["source_width"], frame["source_height"]
        _, joint_frames = color_joints(Path(record["skeleton_path"]), set(range(center - 3, center + 4)))
        for body_id, joint_index, annotated_x, annotated_y in item["landmarks"]:
            central_body = next((body for body in joint_frames[center] if body["body_id"] == body_id), None)
            if central_body is None or central_body["joints"][joint_index]["state"] != 2:
                raise ValueError(f"Invalid central tracked landmark: {item['sample_id']} body={body_id} joint={joint_index}")
            central_joint = central_body["joints"][joint_index]
            central_xy = transform_point(central_joint["x"], central_joint["y"], width, height)
            central_error = math.dist(central_xy, (annotated_x, annotated_y))
            errors.append(central_error)
            detail.append({"sample_id": item["sample_id"], "frame_index": center, "body_id": body_id,
                           "joint_index": joint_index, "annotated_xy": [annotated_x, annotated_y],
                           "projected_xy": list(central_xy), "error_px": central_error})
            observed = []
            for shift in range(-3, 4):
                body = next((body for body in joint_frames[center + shift] if body["body_id"] == body_id), None)
                if body is None or body["joints"][joint_index]["state"] != 2:
                    observed = []
                    break
                joint = body["joints"][joint_index]
                observed.append(transform_point(joint["x"], joint["y"], width, height))
            if not observed:
                continue
            point = (annotated_x, annotated_y)
            offset_errors = [math.dist(position, point) for position in observed]
            informative = max(math.dist(observed[0], observed[3]), math.dist(observed[6], observed[3])) >= 8
            detail[-1]["shift_errors_px"] = {str(shift): error for shift, error in zip(range(-3, 4), offset_errors)}
            for shift, error in zip(range(-3, 4), offset_errors):
                shifted[shift].append(error)
                if informative:
                    moving[shift].append(error)
            detail[-1]["motion_informative"] = informative
    median = statistics.median(errors)
    moving_medians = {str(shift): statistics.median(values) for shift, values in moving.items()} if moving[0] else {}
    center_wins = bool(moving_medians) and all(moving_medians["0"] < value for shift, value in moving_medians.items() if shift != "0")
    report = {"schema_version": 1, "annotation_sha256": sha256_file(annotations_path),
              "annotation_method": "non-blinded visual estimates on clean 224x224 images",
              "sample_count": len(annotations), "joint_count": len(errors), "median_error_px": median,
              "max_error_px": max(errors), "spatial_threshold_px": lock["manual_thresholds"]["max_median_visible_joint_error_px"],
              "spatial_threshold_met": median <= lock["manual_thresholds"]["max_median_visible_joint_error_px"],
              "shifted_median_error_px": {str(shift): statistics.median(values) for shift, values in shifted.items()},
              "motion_informative_joint_count": len(moving[0]), "motion_informative_shifted_median_error_px": moving_medians,
              "zero_offset_best_on_moving_joints": center_wins, "details": detail,
              "limitation": "Sparse approximate non-blinded manual labels; shift comparison uses skeleton t+k against image t, not independent RGB timestamps."}
    (output / "manual_alignment_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def finalize_gate0(output: Path, lock_path: Path, manifest_path: Path, landmarks_path: Path,
                   visual_review_path: Path) -> dict:
    lock = json.loads(lock_path.read_text())
    sample = json.loads((output / "report.json").read_text())
    coverage = json.loads((output / "full_coverage_report.json").read_text())
    manual = json.loads((output / "manual_alignment_report.json").read_text())
    review = json.loads(visual_review_path.read_text())
    if sample["lock_sha256"] != sha256_file(lock_path) or coverage["lock_sha256"] != sha256_file(lock_path):
        raise ValueError("Gate 0 lock hash mismatch")
    if sample["manifest_sha256"] != sha256_file(manifest_path) or coverage["manifest_sha256"] != sha256_file(manifest_path):
        raise ValueError("Gate 0 manifest hash mismatch")
    if manual["annotation_sha256"] != sha256_file(landmarks_path):
        raise ValueError("Manual annotation hash mismatch")
    if review["source_hashes"]["samples_jsonl_sha256"] != sha256_file(output / "samples.jsonl"):
        raise ValueError("Visual review sample hash mismatch")
    if review["source_hashes"]["targeted_a059_jsonl_sha256"] != sha256_file(output / "targeted_a059.jsonl"):
        raise ValueError("Visual review A059 hash mismatch")
    sample_rows = [json.loads(line) for line in (output / "samples.jsonl").read_text().splitlines()]
    if [row["sample_id"] for row in sample_rows] != review["stratified_sample_ids"]:
        raise ValueError("Visual review did not cover the sampled rows")
    if len(set(row["setup"] for row in sample_rows)) != 17 or len(set(row["camera"] for row in sample_rows)) != 3:
        raise ValueError("Stratified audit does not cover every setup and camera")
    if coverage["paired"] != sum(group["paired"] for group in coverage["groups"]["split"].values()):
        raise ValueError("Full coverage split accounting mismatch")
    criteria = {
        "sample_coverage": sample["threshold_met_on_sample"],
        "full_coverage": coverage["automated_usable_fraction"] >= lock["automated_thresholds"]["min_usable_sample_fraction"],
        "spatial_error": manual["spatial_threshold_met"],
        "no_systematic_offset": (manual["zero_offset_best_on_moving_joints"]
                                 and not review["findings"]["systematic_signed_offset_observed"]),
        "visual_review": review["findings"]["middle_frame_overlays_on_visible_people"] == len(sample_rows),
        "test_embargo": not sample["test_evaluated"] and not coverage["test_evaluated"],
    }
    result = {"schema_version": 1, "phase": 4, "gate": 0,
              "status": "pass" if all(criteria.values()) else "fail",
              "criteria": criteria, "lock_sha256": sha256_file(lock_path),
              "manifest_sha256": sha256_file(manifest_path), "landmarks_sha256": sha256_file(landmarks_path),
              "visual_review_sha256": sha256_file(visual_review_path),
              "sampled": sample["sampled"], "full_paired": coverage["paired"],
              "full_usable": coverage["automated_usable"],
              "full_usable_fraction": coverage["automated_usable_fraction"],
              "manual_joint_count": manual["joint_count"], "manual_median_error_px": manual["median_error_px"],
              "motion_informative_joint_count": manual["motion_informative_joint_count"],
              "test_evaluated": False,
              "scope": "NTU60 A001-A059 paired train/validation; small stratified visual review, not an exhaustive image-level alignment proof",
              "remaining_risk": "Manual coordinates are approximate, non-blinded, and sparse; A059 has lower usable coverage."}
    (output / "gate0_decision.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-manifest", type=Path, default=Path("artifacts/skeleton_data/paired_manifest.json"))
    parser.add_argument("--lock", type=Path, default=Path("docs/skeletonclip-phase4-gate0-lock.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/skeletonclip_phase4/gate0"))
    parser.add_argument("--full-coverage", action="store_true", help="Screen all paired train/validation samples without writing overlays")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--temporal-panels", action="store_true")
    parser.add_argument("--evaluate-manual", action="store_true")
    parser.add_argument("--landmarks", type=Path, default=Path("docs/skeletonclip-phase4-gate0-landmarks.json"))
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--visual-review", type=Path, default=Path("docs/skeletonclip-phase4-gate0-visual-review.json"))
    args = parser.parse_args(argv)
    lock = json.loads(args.lock.read_text())
    if args.finalize:
        result = finalize_gate0(args.output, args.lock, args.paired_manifest, args.landmarks, args.visual_review)
        print(json.dumps({key: result[key] for key in ("status", "criteria", "full_usable_fraction", "manual_median_error_px")}, indent=2))
        return
    manifest = json.loads(args.paired_manifest.read_text())
    if args.full_coverage:
        run_full_coverage(manifest, args.output, lock, args.workers,
                          lock_sha256=sha256_file(args.lock), manifest_sha256=sha256_file(args.paired_manifest))
        return
    if args.evaluate_manual:
        report = evaluate_manual_alignment(args.output, args.landmarks, lock)
        print(json.dumps({key: report[key] for key in ("sample_count", "joint_count", "median_error_px", "spatial_threshold_met", "motion_informative_joint_count", "zero_offset_best_on_moving_joints")}, indent=2))
        return
    processor = XCLIPProcessor.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    if args.temporal_panels:
        temporal_panels(args.output, args.landmarks, processor)
        return
    selected = choose_rows(manifest["train"], 24) + choose_rows(manifest["val"], 12)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for row in selected:
        try:
            record = audit_row(row, processor, args.output, lock)
        except (OSError, ValueError, av.error.FFmpegError) as error:
            record = {"sample_id": row["sample_id"], "split": row["split"], "class_id": row["class_id"],
                      "setup": int(row["sample_id"][1:4]), "camera": int(row["sample_id"][5:8]),
                      "automated_usable": False, "error": str(error)}
        records.append(record)
        print(row["sample_id"], record["automated_usable"], flush=True)
    with (args.output / "samples.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    by_group = {}
    for key in ("split", "class_id", "setup", "camera"):
        by_group[key] = {str(value): {"sampled": sum(record.get(key) == value for record in records),
                                      "automated_usable": sum(record.get(key) == value and record["automated_usable"] for record in records)}
                         for value in sorted({record.get(key) for record in records})}
    usable = sum(record["automated_usable"] for record in records)
    report = {"schema_version": 1, "status": "pending_manual_review_and_full_coverage", "test_evaluated": False,
              "lock_sha256": sha256_file(args.lock), "manifest_sha256": sha256_file(args.paired_manifest),
              "model": MODEL, "revision": REVISION, "image_processor": processor.image_processor.to_dict(),
              "sampled": len(records), "automated_usable": usable, "automated_usable_fraction": usable / len(records),
              "threshold_met_on_sample": usable / len(records) >= lock["automated_thresholds"]["min_usable_sample_fraction"],
              "groups": by_group, "errors": dict(Counter(record.get("error", "") for record in records if "error" in record))}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "sampled", "automated_usable", "automated_usable_fraction", "threshold_met_on_sample")}, indent=2))


if __name__ == "__main__":
    main()
