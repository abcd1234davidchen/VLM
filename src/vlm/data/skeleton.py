"""NTU RGB+D skeleton parsing, archive extraction, and RGB pairing audit."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, TextIO

import torch
from torch import Tensor


SKELETON_ID_PATTERN = re.compile(
    r"S(?P<setup>\d{3})C(?P<camera>\d{3})P(?P<participant>\d{3})"
    r"R(?P<repetition>\d{3})A(?P<action>\d{3})",
    re.IGNORECASE,
)
SKELETON_FILE_PATTERN = re.compile(
    rf"{SKELETON_ID_PATTERN.pattern}\.skeleton",
    re.IGNORECASE,
)
RGB_FILE_PATTERN = re.compile(
    rf"{SKELETON_ID_PATTERN.pattern}_rgb\.(?:avi|mp4)",
    re.IGNORECASE,
)
PREPROCESSING_ID = "ntu-rgbd-parser-v1"
SKELETON_TENSOR_PREPROCESSING_ID = (
    "ntu-joint-v1_top2-tracked-bodies_root-center_nearest-whole-clip"
)
PHASE1_ACTION_IDS = tuple(range(1, 60))
PHASE1_EXCLUDED_ACTION_IDS = (60,)


@dataclass(frozen=True, order=True)
class SkeletonIdentity:
    setup: int
    camera: int
    participant: int
    repetition: int
    action: int

    @property
    def sample_id(self) -> str:
        return (
            f"S{self.setup:03d}C{self.camera:03d}P{self.participant:03d}"
            f"R{self.repetition:03d}A{self.action:03d}"
        )


@dataclass(frozen=True)
class SkeletonMetadata:
    frame_count: int
    body_observations: int
    joint_observations: int
    tracked_body_ids: tuple[int, ...]


class SkeletonFormatError(ValueError):
    """Raised when an NTU skeleton file violates the parser contract."""


def parse_skeleton_identity(value: str | Path) -> SkeletonIdentity:
    """Parse a complete NTU sample identity from a skeleton or RGB name."""
    name = Path(value).name
    match = SKELETON_FILE_PATTERN.fullmatch(name)
    if match is None:
        match = RGB_FILE_PATTERN.fullmatch(name)
    if match is None:
        match = SKELETON_ID_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Unsupported NTU sample filename: {value}")
    return SkeletonIdentity(**{key: int(item) for key, item in match.groupdict().items()})


def _read_line(stream: TextIO, *, path: Path, context: str) -> str:
    line = stream.readline()
    if line == "":
        raise SkeletonFormatError(f"{path}: unexpected EOF while reading {context}")
    return line.strip()


def _parse_count(stream: TextIO, *, path: Path, context: str) -> int:
    raw = _read_line(stream, path=path, context=context)
    try:
        value = int(raw)
    except ValueError as error:
        raise SkeletonFormatError(f"{path}: invalid {context}: {raw!r}") from error
    if value < 0:
        raise SkeletonFormatError(f"{path}: negative {context}: {value}")
    return value


def _parse_floats(
    stream: TextIO,
    *,
    path: Path,
    context: str,
    minimum_fields: int,
) -> list[float]:
    raw = _read_line(stream, path=path, context=context)
    fields = raw.split()
    if len(fields) != minimum_fields:
        raise SkeletonFormatError(
            f"{path}: {context} has {len(fields)} fields; expected {minimum_fields}"
        )
    try:
        values = [float(field) for field in fields]
    except ValueError as error:
        raise SkeletonFormatError(f"{path}: non-numeric value in {context}") from error
    if not all(math.isfinite(value) for value in values):
        raise SkeletonFormatError(f"{path}: non-finite value in {context}")
    return values


def _parse_body_metadata(
    stream: TextIO,
    *,
    path: Path,
    context: str,
) -> tuple[int, list[float]]:
    raw = _read_line(stream, path=path, context=context)
    fields = raw.split()
    if len(fields) != 10:
        raise SkeletonFormatError(
            f"{path}: {context} has {len(fields)} fields; expected 10"
        )
    try:
        body_id = int(fields[0])
        values = [float(field) for field in fields[1:]]
    except ValueError as error:
        raise SkeletonFormatError(f"{path}: non-numeric value in {context}") from error
    if body_id < 0 or not all(math.isfinite(value) for value in values):
        raise SkeletonFormatError(f"{path}: invalid or non-finite value in {context}")
    if values[8] not in (0.0, 1.0, 2.0):
        raise SkeletonFormatError(
            f"{path}: invalid body tracking state {values[8]}"
        )
    return body_id, values


def parse_skeleton_file(path: Path) -> SkeletonMetadata:
    """Fully validate one official NTU ``.skeleton`` text file."""
    identity = parse_skeleton_identity(path)
    tracked_ids: set[int] = set()
    body_observations = 0
    joint_observations = 0
    with path.open("r", encoding="utf-8") as stream:
        frame_count = _parse_count(stream, path=path, context="frame count")
        if frame_count == 0:
            raise SkeletonFormatError(f"{path}: empty skeleton sequence")
        for frame_index in range(frame_count):
            body_count = _parse_count(
                stream, path=path, context=f"frame {frame_index} body count"
            )
            frame_body_ids: set[int] = set()
            for body_index in range(body_count):
                body_id, _body = _parse_body_metadata(
                    stream,
                    path=path,
                    context=f"frame {frame_index} body {body_index} metadata",
                )
                if body_id in frame_body_ids:
                    raise SkeletonFormatError(
                        f"{path}: duplicate body ID {body_id} in frame {frame_index}"
                    )
                frame_body_ids.add(body_id)
                tracked_ids.add(body_id)
                joint_count = _parse_count(
                    stream,
                    path=path,
                    context=f"frame {frame_index} body {body_index} joint count",
                )
                if joint_count != 25:
                    raise SkeletonFormatError(
                        f"{path}: frame {frame_index} body {body_index} has "
                        f"{joint_count} joints; expected 25"
                    )
                for joint_index in range(joint_count):
                    joint = _parse_floats(
                        stream,
                        path=path,
                        context=(
                            f"frame {frame_index} body {body_index} joint {joint_index}"
                        ),
                        minimum_fields=12,
                    )
                    tracking_state = joint[11]
                    if tracking_state not in (0.0, 1.0, 2.0):
                        raise SkeletonFormatError(
                            f"{path}: invalid joint tracking state {tracking_state}"
                        )
                body_observations += 1
                joint_observations += joint_count
        if stream.read().strip():
            raise SkeletonFormatError(f"{path}: unexpected trailing content")
    if body_observations == 0:
        raise SkeletonFormatError(f"{path}: sequence contains no tracked bodies")
    if not 1 <= identity.action <= 60 or not 1 <= identity.setup <= 17:
        raise SkeletonFormatError(f"{path}: identity is outside NTU RGB+D 60")
    return SkeletonMetadata(
        frame_count=frame_count,
        body_observations=body_observations,
        joint_observations=joint_observations,
        tracked_body_ids=tuple(sorted(tracked_ids)),
    )


def load_skeleton_tensor(path: Path, *, frames: int = 64) -> Tensor:
    """Load deterministic joint coordinates as ``[3, T, 25, 2]``.

    Body slots are identity-consistent and ranked by tracked observations.
    Nearest-frame whole-clip resampling never interpolates padded coordinates.
    Each present body is centered on its spine-base joint independently.
    """
    if frames < 1:
        raise ValueError("frames must be at least 1")
    parse_skeleton_identity(path)
    sequence: list[dict[int, Tensor]] = []
    tracked_observations: Counter[int] = Counter()
    with path.open("r", encoding="utf-8") as stream:
        frame_count = _parse_count(stream, path=path, context="frame count")
        if frame_count == 0:
            raise SkeletonFormatError(f"{path}: empty skeleton sequence")
        for frame_index in range(frame_count):
            body_count = _parse_count(
                stream, path=path, context=f"frame {frame_index} body count"
            )
            bodies: dict[int, Tensor] = {}
            for body_index in range(body_count):
                body_id, _metadata = _parse_body_metadata(
                    stream,
                    path=path,
                    context=f"frame {frame_index} body {body_index} metadata",
                )
                if body_id in bodies:
                    raise SkeletonFormatError(
                        f"{path}: duplicate body ID {body_id} in frame {frame_index}"
                    )
                joint_count = _parse_count(
                    stream,
                    path=path,
                    context=f"frame {frame_index} body {body_index} joint count",
                )
                if joint_count != 25:
                    raise SkeletonFormatError(
                        f"{path}: frame {frame_index} body {body_index} has "
                        f"{joint_count} joints; expected 25"
                    )
                coordinate_rows: list[list[float]] = []
                tracked = 0
                for joint_index in range(25):
                    joint = _parse_floats(
                        stream,
                        path=path,
                        context=f"frame {frame_index} body {body_index} joint {joint_index}",
                        minimum_fields=12,
                    )
                    if joint[11] not in (0.0, 1.0, 2.0):
                        raise SkeletonFormatError(
                            f"{path}: invalid joint tracking state {joint[11]}"
                        )
                    coordinate_rows.append(joint[:3])
                    tracked += int(joint[11] > 0)
                bodies[body_id] = torch.tensor(
                    coordinate_rows, dtype=torch.float32
                ).transpose(0, 1)
                tracked_observations[body_id] += tracked
            sequence.append(bodies)
        if stream.read().strip():
            raise SkeletonFormatError(f"{path}: unexpected trailing content")
    if not tracked_observations:
        raise SkeletonFormatError(f"{path}: sequence contains no tracked bodies")
    selected_ids = sorted(
        tracked_observations,
        key=lambda body_id: (-tracked_observations[body_id], body_id),
    )[:2]
    tensor = torch.zeros(3, frame_count, 25, 2, dtype=torch.float32)
    for frame_index, bodies in enumerate(sequence):
        for person_index, body_id in enumerate(selected_ids):
            coordinates = bodies.get(body_id)
            if coordinates is None:
                continue
            tensor[:, frame_index, :, person_index] = (
                coordinates - coordinates[:, :1]
            )
    indices = torch.linspace(0, frame_count - 1, frames).round().long()
    result = tensor.index_select(1, indices)
    if not torch.isfinite(result).all():
        raise SkeletonFormatError(f"{path}: non-finite preprocessed tensor")
    return result


def load_missing_skeleton_ids(path: Path) -> set[str]:
    """Load sample IDs from the official list, ignoring its prose header."""
    identities: set[str] = set()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = SKELETON_ID_PATTERN.search(line)
        if match is not None:
            identities.add(parse_skeleton_identity(match.group(0)).sample_id)
    if not identities:
        raise ValueError(f"No NTU sample identities found in {path}")
    return identities


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract_skeleton_archive(archive_path: Path, output_dir: Path) -> list[Path]:
    """Extract only flat/wrapped NTU skeleton members after validating ZIP metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(archive_path) as archive:
        members: list[tuple[zipfile.ZipInfo, str]] = []
        seen: set[str] = set()
        for info in archive.infolist():
            member = PurePosixPath(info.filename)
            if info.is_dir():
                continue
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"Unsafe archive member: {info.filename}")
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"Symlink archive member is forbidden: {info.filename}")
            filename = member.name
            if SKELETON_FILE_PATTERN.fullmatch(filename) is None:
                raise ValueError(f"Unexpected archive member: {info.filename}")
            normalized = filename.upper()
            if normalized in seen:
                raise ValueError(f"Duplicate skeleton archive member: {filename}")
            seen.add(normalized)
            members.append((info, filename))
        if not members:
            raise ValueError(f"Archive contains no skeleton files: {archive_path}")
        with tempfile.TemporaryDirectory(dir=output_dir, prefix=".extract-") as temporary:
            staging = Path(temporary)
            for info, filename in members:
                staged_path = staging / filename
                with archive.open(info) as source, staged_path.open("wb") as target:
                    shutil.copyfileobj(source, target)
            bad_member = archive.testzip()
            if bad_member is not None:
                raise zipfile.BadZipFile(f"CRC failure for {bad_member}")
            targets = [output_dir / filename for _info, filename in members]
            existing = next((path for path in targets if path.exists()), None)
            if existing is not None:
                raise FileExistsError(f"Refusing to overwrite {existing}")
            for _info, filename in members:
                target_path = output_dir / filename
                os.replace(staging / filename, target_path)
                extracted.append(target_path)
    return extracted


def _canonical_json_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def audit_skeleton_data(
    *,
    skeleton_dir: Path,
    rgb_manifest_path: Path,
    missing_skeletons_path: Path,
    artifacts_dir: Path,
    archive_paths: Iterable[Path] = (),
    expected_file_count: int = 56_880,
) -> dict[str, object]:
    """Validate NTU60 skeletons and pair only canonical train/validation samples."""
    extracted_count = 0
    archive_sources = []
    for archive_path in archive_paths:
        archive_sources.append({"path": str(archive_path), "sha256": sha256_file(archive_path)})
        extracted_count += len(safe_extract_skeleton_archive(archive_path, skeleton_dir))

    missing_ids = load_missing_skeleton_ids(missing_skeletons_path)
    skeleton_paths = sorted(skeleton_dir.glob("*.skeleton"))
    if len(skeleton_paths) != expected_file_count:
        raise ValueError(
            f"Expected {expected_file_count} skeleton files, found {len(skeleton_paths)} in {skeleton_dir}"
        )

    valid: dict[str, tuple[Path, SkeletonIdentity, SkeletonMetadata]] = {}
    excluded: list[dict[str, object]] = []
    skeleton_exclusions: Counter[str] = Counter()
    duplicate_ids: list[str] = []
    for path in skeleton_paths:
        try:
            identity = parse_skeleton_identity(path)
        except ValueError as error:
            excluded.append({"skeleton_path": str(path), "reason": "invalid_filename", "detail": str(error)})
            skeleton_exclusions["invalid_filename"] += 1
            continue
        sample_id = identity.sample_id
        if sample_id in valid:
            duplicate_ids.append(sample_id)
            excluded.append({"sample_id": sample_id, "skeleton_path": str(path), "reason": "duplicate_identity"})
            skeleton_exclusions["duplicate_identity"] += 1
            continue
        if sample_id in missing_ids:
            excluded.append({"sample_id": sample_id, "skeleton_path": str(path), "reason": "official_missing_or_incomplete"})
            skeleton_exclusions["official_missing_or_incomplete"] += 1
            continue
        try:
            metadata = parse_skeleton_file(path)
        except (OSError, UnicodeError, SkeletonFormatError) as error:
            excluded.append({"sample_id": sample_id, "skeleton_path": str(path), "reason": "parser_validation", "detail": str(error)})
            skeleton_exclusions["parser_validation"] += 1
            continue
        valid[sample_id] = (path, identity, metadata)

    rgb_payload = json.loads(rgb_manifest_path.read_text(encoding="utf-8"))
    paired: dict[str, list[dict[str, object]]] = {"train": [], "val": []}
    rgb_seen: dict[str, str] = {}
    coverage: dict[str, dict[str, object]] = {}
    label_mismatches: list[dict[str, object]] = []
    for split_name in ("train", "val"):
        records = rgb_payload.get(split_name)
        if not isinstance(records, list):
            raise ValueError(f"RGB manifest has no {split_name!r} list")
        reasons: Counter[str] = Counter()
        class_counts: Counter[str] = Counter()
        eligible_samples = 0
        for record in records:
            rgb_path = Path(str(record["video_path"]))
            identity = parse_skeleton_identity(rgb_path)
            sample_id = identity.sample_id
            if sample_id in rgb_seen:
                raise ValueError(
                    f"Duplicate RGB identity {sample_id} in {rgb_seen[sample_id]} and {split_name}"
                )
            rgb_seen[sample_id] = split_name
            declared_action = int(record["class_id"])
            if declared_action != identity.action:
                mismatch = {"sample_id": sample_id, "split": split_name, "declared_action": declared_action, "filename_action": identity.action}
                label_mismatches.append(mismatch)
                reasons["label_mismatch"] += 1
                excluded.append({**mismatch, "rgb_path": str(rgb_path), "reason": "label_mismatch"})
                continue
            if identity.setup > 17 or identity.action not in PHASE1_ACTION_IDS:
                reasons["outside_phase1_scope"] += 1
                continue
            eligible_samples += 1
            skeleton = valid.get(sample_id)
            if skeleton is None:
                reason = "official_missing_or_incomplete" if sample_id in missing_ids else "no_valid_skeleton"
                reasons[reason] += 1
                excluded.append({"sample_id": sample_id, "split": split_name, "rgb_path": str(rgb_path), "reason": reason})
                continue
            skeleton_path, skeleton_identity, metadata = skeleton
            if skeleton_identity.action != declared_action:
                raise AssertionError("Exact identity pairing produced an action mismatch")
            paired[split_name].append(
                {
                    "sample_id": sample_id,
                    "split": split_name,
                    "class_id": f"A{declared_action:03d}",
                    "class_index": PHASE1_ACTION_IDS.index(declared_action),
                    "rgb_path": str(rgb_path),
                    "skeleton_path": str(skeleton_path),
                    "skeleton": asdict(metadata),
                }
            )
            class_counts[f"A{declared_action:03d}"] += 1
        total = len(records)
        coverage[split_name] = {
            "rgb_samples": total,
            "eligible_rgb_samples": eligible_samples,
            "paired_samples": len(paired[split_name]),
            "paired_fraction": len(paired[split_name]) / total if total else 0.0,
            "paired_fraction_of_eligible": (
                len(paired[split_name]) / eligible_samples
                if eligible_samples
                else 0.0
            ),
            "excluded_by_reason": dict(sorted(reasons.items())),
            "paired_by_class": dict(sorted(class_counts.items())),
        }

    participant_sets = {
        split_name: {parse_skeleton_identity(row["sample_id"]).participant for row in rows}
        for split_name, rows in paired.items()
    }
    leakage = sorted(participant_sets["train"] & participant_sets["val"])
    if leakage:
        raise ValueError(f"Participant leakage between paired train and val: {leakage}")

    paired_body = {
        "schema_version": 1,
        "preprocessing_id": PREPROCESSING_ID,
        "ordered_class_ids": [f"A{action:03d}" for action in PHASE1_ACTION_IDS],
        "excluded_action_ids": [
            f"A{action:03d}" for action in PHASE1_EXCLUDED_ACTION_IDS
        ],
        "class_exclusion_reason": (
            "A060 discarded by user decision after Gate 0 data review because "
            "its usable skeleton coverage was severely degraded by non-finite values."
        ),
        "source_rgb_manifest": str(rgb_manifest_path),
        "train": paired["train"],
        "val": paired["val"],
    }
    manifest_hash = _canonical_json_hash(paired_body)
    paired_payload = {**paired_body, "manifest_sha256": manifest_hash}
    source_hashes = {
        "rgb_manifest": {"path": str(rgb_manifest_path), "sha256": sha256_file(rgb_manifest_path)},
        "missing_skeletons": {"path": str(missing_skeletons_path), "sha256": sha256_file(missing_skeletons_path)},
        "archives": archive_sources,
        "skeleton_inventory_sha256": _canonical_json_hash([path.name for path in skeleton_paths]),
    }
    coverage_payload = {
        "schema_version": 1,
        "preprocessing_id": PREPROCESSING_ID,
        "expected_skeleton_files": expected_file_count,
        "discovered_skeleton_files": len(skeleton_paths),
        "valid_skeleton_files": len(valid),
        "skeleton_excluded_by_reason": dict(sorted(skeleton_exclusions.items())),
        "official_exclusion_count": len(missing_ids),
        "ordered_class_ids": [f"A{action:03d}" for action in PHASE1_ACTION_IDS],
        "excluded_action_ids": [
            f"A{action:03d}" for action in PHASE1_EXCLUDED_ACTION_IDS
        ],
        "extracted_files": extracted_count,
        "duplicate_skeleton_identities": sorted(set(duplicate_ids)),
        "label_mismatches": label_mismatches,
        "participant_leakage": leakage,
        "splits": coverage,
        "paired_manifest_sha256": manifest_hash,
        "source_hashes": source_hashes,
    }
    _write_json(artifacts_dir / "paired_manifest.json", paired_payload)
    _write_json(artifacts_dir / "excluded_samples.json", {"schema_version": 1, "samples": excluded})
    _write_json(artifacts_dir / "coverage.json", coverage_payload)
    return coverage_payload
