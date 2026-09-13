"""Command-line entry point for the SkeletonCLIP Gate 0 data audit."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .data.skeleton import audit_skeleton_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit and pair NTU RGB+D 60 skeleton data.")
    parser.add_argument("--skeleton-dir", type=Path, required=True)
    parser.add_argument("--rgb-manifest", type=Path, required=True)
    parser.add_argument("--missing-skeletons", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/skeleton_data"))
    parser.add_argument("--archive", type=Path, action="append", default=[])
    parser.add_argument("--expected-file-count", type=int, default=56_880)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = audit_skeleton_data(
        skeleton_dir=args.skeleton_dir,
        rgb_manifest_path=args.rgb_manifest,
        missing_skeletons_path=args.missing_skeletons,
        artifacts_dir=args.artifacts_dir,
        archive_paths=args.archive,
        expected_file_count=args.expected_file_count,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
