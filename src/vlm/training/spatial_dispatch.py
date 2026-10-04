"""Dispatch a spatial job or attach persistent progress logging to a live job."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

from vlm.reporting import write_json


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat.rsplit(")", 1)[1].split()[0] != "Z"
    except (ProcessLookupError, FileNotFoundError):
        return False


def watch(root: Path, pid: int) -> None:
    previous = {}
    with (root / "train.log").open("a", encoding="utf-8", buffering=1) as stream:
        def emit(event: dict) -> None:
            stream.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), **event}) + "\n")

        emit({"event": "logger_attached", "training_pid": pid})
        while True:
            for path in sorted(root.rglob("progress.json")):
                try:
                    progress = json.loads(path.read_text())
                except (FileNotFoundError, json.JSONDecodeError):
                    continue
                if previous.get(str(path)) != progress:
                    emit({"event": "progress", "arm": str(path.parent.relative_to(root)), **progress})
                    previous[str(path)] = progress
            for path in sorted(root.rglob("selection.json")):
                try:
                    selection = json.loads(path.read_text())
                except (FileNotFoundError, json.JSONDecodeError):
                    continue
                summary = {group: {view: ({key: metrics.get(key) for key in
                                          ("epoch", "raw_accuracy", "raw_macro_f1", "correct", "attention_correct")}
                                         if metrics else None)
                                   for view, metrics in values.items()}
                           for group, values in selection.items()}
                if previous.get(str(path)) != summary:
                    emit({"event": "checkpoint_selection", "arm": str(path.parent.relative_to(root)), **summary})
                    previous[str(path)] = summary
            report_path = root / "run_report.json"
            if report_path.exists():
                try:
                    report = json.loads(report_path.read_text())
                except (FileNotFoundError, json.JSONDecodeError):
                    # The trainer may still be writing its final JSON report.
                    time.sleep(1)
                    continue
                emit({"event": "completed", "status": report["status"],
                      "raw_accuracy_gain": report["raw_accuracy_gain"]})
                write_json(root / "dispatch_status.json", {"state": "complete", "training_pid": pid, "report": str(report_path)})
                return
            if not process_alive(pid):
                emit({"event": "process_ended_without_report", "training_pid": pid})
                write_json(root / "dispatch_status.json", {"state": "failed", "training_pid": pid,
                                                           "reason": "Training process ended without final run report; inspect console log."})
                return
            time.sleep(10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--attach-pid", type=int)
    parser.add_argument("--backend", choices=("tmux", "process"), default="tmux")
    parser.add_argument("--tmux-session")
    parser.add_argument("--watch", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = args.artifacts_dir.resolve()
    if args.watch:
        watch(root, args.attach_pid)
        return
    root.mkdir(parents=True, exist_ok=True)
    if (root / "dispatch.json").exists():
        raise ValueError("Existing dispatch metadata must not be overwritten")
    session = None
    if args.attach_pid:
        if not process_alive(args.attach_pid):
            raise ValueError("Cannot attach to a terminal or missing training process")
        command = Path(f"/proc/{args.attach_pid}/cmdline").read_bytes().split(b"\0")
        command = [part.decode() for part in command if part]
        if "vlm.training.spatial_attention" not in command:
            raise ValueError("PID is not a spatial attention training process")
        pid = args.attach_pid
        method = "attached_existing_exec_session"
    else:
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            raise ValueError("Provide a command or --attach-pid")
        if args.backend == "tmux":
            if shutil.which("tmux") is None:
                raise RuntimeError("tmux is unavailable; install it or explicitly use --backend process")
            session = args.tmux_session or "spatial-" + re.sub(r"[^a-zA-Z0-9_-]", "-", root.name)
            if not re.fullmatch(r"[a-zA-Z0-9_-]+", session):
                raise ValueError("Unsupported tmux session name")
            existing = subprocess.run(["tmux", "has-session", "-t", session], capture_output=True)
            if existing.returncode == 0:
                raise ValueError(f"tmux session already exists: {session}")
            shell_command = "exec " + shlex.join(command) + " >> " + shlex.quote(str(root / "console.log")) + " 2>&1"
            subprocess.run(["tmux", "new-session", "-d", "-s", session, "-c", str(Path.cwd()), shell_command], check=True)
            pane = subprocess.run(["tmux", "list-panes", "-t", session, "-F", "#{pane_pid}"],
                                  check=True, capture_output=True, text=True)
            pid, method = int(pane.stdout.strip()), "tmux"
        else:
            with (root / "console.log").open("ab", buffering=0) as console:
                job = subprocess.Popen(command, stdout=console, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
            pid, method = job.pid, "detached_process"
    with (root / "logger.log").open("ab", buffering=0) as console:
        logger = subprocess.Popen([sys.executable, "-m", "vlm.training.spatial_dispatch", "--watch",
                                   "--attach-pid", str(pid), "--artifacts-dir", str(root)],
                                  stdout=console, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                  start_new_session=True)
    record = {"training_pid": pid, "logger_pid": logger.pid, "method": method,
              "tmux_session": session,
              "command": command, "created_at": datetime.now(timezone.utc).isoformat(),
              "log": str(root / "train.log"), "test_evaluated": False}
    write_json(root / "dispatch.json", record)
    write_json(root / "dispatch_status.json", {"state": "running", "training_pid": pid, "logger_pid": logger.pid})
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
