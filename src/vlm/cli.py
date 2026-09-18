"""Unified command-line dispatcher for all VLM workflows."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence


TRAIN_COMMANDS = {
    "multiclass": "vlm.training.multiclass",
    "prototype": "vlm.training.prototype",
    "binary": "vlm.training.binary",
    "xclip": "vlm.training.xclip",
    "xclip-lora": "vlm.training.xclip_lora",
    "xclip-lora-tune": "vlm.training.xclip_lora_tune",
    "skeleton-gate2": "vlm.training.skeletonclip",
    "skeleton-gate3": "vlm.training.skeletonclip_gate3",
    "skeleton-phase2": "vlm.training.skeletonclip_phase2",
    "skeleton-phase2-gate3": "vlm.training.skeletonclip_phase2_gate3",
    "skeleton-phase2-gate4": "vlm.training.skeletonclip_phase2_gate4",
    "skeleton-phase3": "vlm.training.skeletonclip_phase3",
    "skeleton-phase3-gate1": "vlm.training.skeletonclip_phase3_gate1",
    "skeleton-phase4-gate1": "vlm.training.skeletonclip_phase4_gate1",
    "skeleton-phase4-gate2": "vlm.training.skeletonclip_phase4_gate2",
}
EMBEDDING_COMMANDS = {
    name: module
    for name, module in TRAIN_COMMANDS.items()
    if name != "xclip-lora-tune"
}
EXPERIMENT_COMMANDS = {
    "clip": "vlm.experiments.clip_video",
    "xclip": "vlm.experiments.xclip",
    "xclip-vision": "vlm.experiments.xclip_vision",
    "skeleton-gate1": "vlm.experiments.skeleton_gate1",
}
DATA_COMMANDS = {
    "skeleton-audit": "vlm.skeleton_data",
    "skeleton-phase4-gate0": "vlm.phase4_alignment",
}


def _root_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vlm",
        description="Train and experiment with CLIP/XCLIP video models.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("train", "embeddings", "experiment", "data"),
        help="Workflow family to run.",
    )
    return parser


def _print_group_help(group: str, choices: Sequence[str]) -> None:
    print(f"usage: vlm {group} <command> [options]\n")
    print("commands:")
    for choice in choices:
        print(f"  {choice}")
    print(
        f"\nRun 'vlm {group} <command> --help' for command options."
    )


def _dispatch_named(
    group: str,
    argv: list[str],
    commands: dict[str, str],
) -> None:
    if not argv or argv[0] in {"-h", "--help"}:
        _print_group_help(group, tuple(commands))
        return
    name, *remaining = argv
    if name not in commands:
        choices = ", ".join(commands)
        raise SystemExit(
            f"Unknown {group} command {name!r}; choose from: {choices}"
        )
    _invoke(
        commands[name],
        remaining,
        prog=f"vlm {group} {name}",
    )


def _invoke(
    module_name: str,
    argv: list[str],
    *,
    prog: str,
) -> None:
    module = importlib.import_module(module_name)
    original_prog = sys.argv[0]
    try:
        sys.argv[0] = prog
        module.main(argv)
    finally:
        sys.argv[0] = original_prog


def _dispatch_embeddings(argv: list[str]) -> None:
    pipeline_choices = ",".join(EMBEDDING_COMMANDS)
    if not argv or argv[0] in {"-h", "--help"}:
        print(
            "usage: vlm embeddings extract --pipeline "
            f"{{{pipeline_choices}}} [options]"
        )
        return
    action, *remaining = argv
    if action != "extract":
        raise SystemExit(
            f"Unknown embeddings command {action!r}; choose: extract"
        )
    if not remaining or (
        any(argument in {"-h", "--help"} for argument in remaining)
        and "--pipeline" not in remaining
    ):
        print(
            "usage: vlm embeddings extract --pipeline "
            f"{{{pipeline_choices}}} [options]\n\n"
            "Select a pipeline, then append --help to see its options."
        )
        return

    selector = argparse.ArgumentParser(
        prog="vlm embeddings extract",
        add_help=False,
    )
    selector.add_argument(
        "--pipeline",
        required=True,
        choices=tuple(EMBEDDING_COMMANDS),
    )
    selected, pipeline_args = selector.parse_known_args(remaining)
    _invoke(
        EMBEDDING_COMMANDS[selected.pipeline],
        [*pipeline_args, "--extract-only"],
        prog="vlm embeddings extract",
    )


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        _root_parser().print_help()
        return

    group, *remaining = arguments
    if group == "train":
        _dispatch_named("train", remaining, TRAIN_COMMANDS)
        return
    if group == "embeddings":
        _dispatch_embeddings(remaining)
        return
    if group == "experiment":
        _dispatch_named(
            "experiment",
            remaining,
            EXPERIMENT_COMMANDS,
        )
        return
    if group == "data":
        _dispatch_named("data", remaining, DATA_COMMANDS)
        return
    _root_parser().error(f"unknown command: {group}")


def entrypoint() -> None:
    main()


if __name__ == "__main__":
    main()
