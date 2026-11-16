"""Live alert demo server: browser frames -> sliding-window stage 1 -> alert -> VLM check -> notify or hold.

The browser streams JPEG frames from the webcam or a local video file. Every inference step takes
8 uniform frames from the last ``window`` seconds and runs the locked stage-1 configuration. An
alert fires when the falling or nausea/vomiting probability reaches its own threshold for ``persist``
consecutive steps (one by default: the verifier filters false alerts, and falling scores often cross the
threshold on single steps).

The VLM then checks the alert window together with 4 frames from the ``window`` seconds before it and 4
from the ``window`` seconds after it, so it sees what led up to the alert and how it ended (a fall usually
crosses the threshold before the person is down); the check waits for those later frames. Any answer
other than a normal activity notifies at once. A normal-activity answer puts the alert on hold instead
of deleting it: each recheck sends the 8 frames of the ``window`` seconds that start ``recheck`` seconds
of video after the frames the previous check ended on, so the frames checked depend only on when the
alert fired, not on how fast the VLM answers. Any recheck that is not a normal activity notifies, and
only ``rechecks`` consecutive normal-activity answers clear the alert, which stays in the log. A missing
answer also notifies, so a wrong normal-activity answer costs a delay, never a missed alert. If there is no
new video to recheck (the video file ended or the source stopped), the alert ends as ``ended`` without
notifying and stays in the log.

With an ntfy topic (``ntfy_topic`` in ``demo_local.json``, or ``--ntfy-topic``), every notified alert is
also pushed to the phones subscribed to that topic, with a link to an alert page served from this computer
through a Cloudflare Quick Tunnel (see notify.py). The page loops the frames the verifier saw, marks the
alert as seen by the phone that opens it (named through the registration push sent at startup), has a
demo-only emergency button that places no call, and shows the person's details from ``demo_local.json``.
That file holds personal settings and is ignored by git.

    uv run python -m vlm.demo.server
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from collections import deque
from collections.abc import Iterable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from vlm.demo.notify import AlertShare, Notifier
from vlm.demo.stage1 import ALERT_EVENTS, DEFAULT_THRESHOLDS
from vlm.experiments.vlm_verifier import PROMPT

STATIC = Path(__file__).with_name("static")
VERIFIER_SCRIPT = Path(__file__).parents[1] / "experiments" / "vlm_verifier.py"
VERIFIER_ENV = ["uv", "run", "--offline", "--no-project", "--python", "3.13", "--with", "mlx-vlm==0.7.8",
                "--with", "opencv-python", "python", str(VERIFIER_SCRIPT), "serve", "--model"]
NUM_FRAMES = 8
HISTORY = 600
STEPS = 40
THUMB_SIDE = 224  # XCLIP input resolution: what the classifier sees after its processor
VERIFIER_TIMEOUT_S = 90.0  # notify when a VLM answer takes longer than this
FRESH_WINDOW_S = 5.0  # a recheck still waiting for its frames ends the alert when no frame arrived this recently
BEFORE_FRAMES, AFTER_FRAMES = 4, 4  # first check only, around the 8 alert frames
AFTER_SLACK_S = 2.0  # send the first check this long after its later frames were due, with what has arrived
# Settings the page can change, with their types and the command-line options that set them at startup.
SETTING_KINDS = {"window_s": float, "persist": int, "cooldown_s": float, "verifier": bool, "recheck_s": float, "rechecks": int}
SETTING_ARGS = {"window_s": "window", "persist": "persist", "cooldown_s": "cooldown", "recheck_s": "recheck", "rechecks": "rechecks"}
THRESHOLD_ARGS = {"A043": "fall_threshold", "A048": "vomit_threshold"}


def parse_settings(changes: dict) -> dict:
    """The known settings in ``changes``, converted to their types; unknown keys are dropped."""
    settings = {key: SETTING_KINDS[key](value) for key, value in changes.items() if key in SETTING_KINDS}
    thresholds = {code: float(value) for code, value in changes.get("thresholds", {}).items() if code in THRESHOLD_ARGS}
    return settings | ({"thresholds": thresholds} if thresholds else {})


def settings_from_args(args: argparse.Namespace) -> dict:
    return {"thresholds": {code: getattr(args, dest) for code, dest in THRESHOLD_ARGS.items()},
            "verifier": args.verifier_model != "none"} | {key: getattr(args, dest) for key, dest in SETTING_ARGS.items()}


class Verifier:
    """One long-lived mlx-vlm worker process; requests are answered in order."""

    def __init__(self, model: str, on_result, log_path: Path) -> None:
        self.model, self.on_result = model, on_result
        self.status, self.pending = "starting", 0
        self.lock = threading.Lock()
        # The worker has its own isolated environment; do not let it inherit the demo's virtualenv.
        env = {key: value for key, value in os.environ.items() if key not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME")}
        self.log = log_path.open("a")
        self.process = subprocess.Popen(VERIFIER_ENV + [model], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, text=True, bufsize=1, env=env | {"HF_HUB_OFFLINE": "1"})
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self.process.stdout:
            if not line.startswith("@@ "):
                continue
            payload = json.loads(line[3:])
            if payload.get("ready"):
                self.status = "ready"
                continue
            with self.lock:
                self.pending -= 1
            self.on_result(payload)
        self.status = f"exited ({self.process.wait()}), see {self.log.name}"

    def submit(self, alert_id: str, frames: list[str]) -> None:
        with self.lock:
            self.pending += 1
        self.process.stdin.write(json.dumps({"id": alert_id, "frames": frames}) + "\n")
        self.process.stdin.flush()

    def close(self) -> None:
        self.process.terminate()


class Monitor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.config = settings_from_args(args)
        # The verifier switch has no command-line option, so a saved user default applies directly.
        self.config["verifier"] &= args.user_defaults.get("verifier", True)
        self.lock = threading.Lock()
        self.frames: deque[tuple[float, np.ndarray]] = deque()
        self.history: deque[dict] = deque(maxlen=HISTORY)
        self.alerts: list[dict] = []
        self.latest: dict | None = None
        self.latest_frames: np.ndarray | None = None
        self.steps: deque[dict] = deque(maxlen=STEPS)
        self.images: dict[str, bytes] = {}
        self.step = 0
        self.session, self.last_frame_t, self.new_frames = 0, None, 0
        self.received: deque[float] = deque(maxlen=60)
        self.streak, self.last_alert_t = 0, -1e9
        self.stage1, self.stage1_status = None, "loading"
        self.alerts_dir = args.alerts_dir
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        self.verifier = Verifier(args.verifier_model, self._verified, self.alerts_dir / "verifier.log") if args.verifier_model != "none" else None
        # Per open alert: its phase, the monotonic deadline of an answer, and the recheck windows sampled so far.
        self.open: dict[str, dict] = {}
        self.notifier, self.test_alert = None, None
        self.registrations: dict[str, str] = {}  # phone device id -> registered name, for this tunnel's lifetime
        topic = getattr(args, "ntfy_topic", None) or self._local().get("ntfy_topic")
        if topic:
            share = AlertShare(args.share_port, args.share_hours * 3600, self.alerts_dir / "tunnel.log", self,
                               tunnel=not args.no_tunnel)
            self.notifier = Notifier(topic, args.ntfy_server, share)
        threading.Thread(target=self._load_and_run, daemon=True).start()
        threading.Thread(target=self._watch_open_alerts, daemon=True).start()

    # ------------------------------------------------------------------ input
    def add_frame(self, t: float, jpeg: bytes, session: int) -> None:
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Not a JPEG frame")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with self.lock:
            if session != self.session or (self.last_frame_t is not None and t < self.last_frame_t):
                self._reset_locked(session)  # new source, or a video file looped / seeked back
            self.frames.append((t, rgb))
            while self.frames and self.frames[0][0] < t - max(10.0, 3 * self.config["window_s"]):
                self.frames.popleft()
            self.last_frame_t, self.new_frames = t, self.new_frames + 1
            self.received.append(time.monotonic())

    def _reset_locked(self, session: int) -> None:
        # A new source starts fresh: the previous source's alert must not hold this one back through the cooldown.
        self.session, self.last_frame_t, self.new_frames, self.streak = session, None, 0, 0
        self.last_alert_t = -1e9
        self.frames.clear()
        self.history.clear()
        self.steps.clear()
        self.latest, self.latest_frames = None, None

    def reset(self, session: int) -> None:
        with self.lock:
            self._reset_locked(session)

    def window(self) -> tuple[np.ndarray, list[float]] | None:
        with self.lock:
            if self.new_frames == 0 or not self.frames:
                return None
            end = self.frames[-1][0]
            selected = [(t, f) for t, f in self.frames if t >= end - self.config["window_s"]]
            span = selected[-1][0] - selected[0][0] if selected else 0
            if len(selected) < NUM_FRAMES or span < min(1.5, 0.6 * self.config["window_s"]):
                return None
            self.new_frames = 0
        indices = np.linspace(0, len(selected) - 1, NUM_FRAMES).round().astype(int)
        return np.stack([selected[i][1] for i in indices]), [selected[i][0] for i in indices]

    def _sample_locked(self, start: float, end: float, count: int) -> list[np.ndarray]:
        """Up to ``count`` uniform frames with ``start < t < end``; fewer near the start or end of a source."""
        selected = [f for t, f in self.frames if start < t < end]
        if len(selected) <= count:
            return selected
        return [selected[i] for i in np.linspace(0, len(selected) - 1, count).round().astype(int)]

    # ------------------------------------------------------------------ inference
    def _load_and_run(self) -> None:
        from vlm.demo.stage1 import Stage1
        try:
            self.stage1 = Stage1(self.args.crop_aware, self.args.person_trained, self.args.pose_model, self.args.device)
            self.stage1_status = f"ready ({self.stage1.device})"
        except Exception as error:
            self.stage1_status = f"failed: {error!r}"
            raise
        while True:
            step_start = time.monotonic()
            window = self.window()
            if window is not None:
                frames, times = window
                session = self.session
                result = self.stage1.predict(frames)
                result |= {"t": times[-1], "window": [times[0], times[-1]], "frame_times": times, "wall": time.time()}
                result["inputs"] = self._encode_inputs(frames, result.pop("views"))
                with self.lock:
                    if session == self.session:
                        self._update_locked(result, frames)
            time.sleep(max(0.0, self.args.min_interval - (time.monotonic() - step_start)))

    def _encode_inputs(self, frames: np.ndarray, views: dict[str, np.ndarray]) -> dict[str, list[str]]:
        """JPEG thumbnails of the sampled window frames and of both classifier inputs, kept for the latest step."""
        self.step += 1
        images, urls = {}, {}
        sources = {"window": frames, **views}
        for view, stack in sources.items():
            urls[view] = []
            for index, frame in enumerate(stack):
                if view == "window":
                    size = (THUMB_SIDE * frame.shape[1] // frame.shape[0], THUMB_SIDE)
                else:
                    size = (THUMB_SIDE, THUMB_SIDE)
                image = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), size, interpolation=cv2.INTER_AREA)
                key = f"{self.step}/{view}/{index}.jpg"
                images[key] = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
                urls[view].append(f"/api/input/{key}")
        self.images = images  # swap atomically; older steps are dropped
        return urls

    def _update_locked(self, result: dict, frames: np.ndarray) -> None:
        config = self.config
        # An event is above when its probability reaches its own threshold; a ratio of at least 1 means above.
        ratios = {code: p / config["thresholds"][code] for code, p in result["events"].items()}
        above = max(ratios.values()) >= 1
        result["ratios"] = ratios
        self.streak = self.streak + 1 if above else 0
        result["above"], result["streak"], result["step"] = above, self.streak, self.step
        self.latest, self.latest_frames = result, frames
        self.history.append({"t": result["t"], "events": result["events"], "top": result["top"][0]["class_id"]})
        self.steps.append({"step": self.step, "t": result["t"], "wall": result["wall"], "events": result["events"],
                           "above": above, "top": result["top"][0]["name"], "fallback": result["fallback"] is not None,
                           "ms": result["timing_ms"]["pose"] + result["timing_ms"]["classifier"],
                           "a_events": result["models"]["A_center_zoom70"]["events"],
                           "b_events": result["models"]["B_person_crop"]["events"]})
        # While an alert is still being checked or on hold, its rechecks already watch this person.
        if self.streak >= config["persist"] and time.monotonic() - self.last_alert_t >= config["cooldown_s"] and not self.open:
            self.last_alert_t = time.monotonic()
            self._raise_alert_locked(result, frames)

    def verify_now(self) -> str:
        """Send the latest window to the verifier without an alert (manual check)."""
        with self.lock:
            if self.latest is None:
                raise ValueError("No inference window yet")
            if self.verifier is None:
                raise ValueError("Verifier is disabled")
            return self._raise_alert_locked(self.latest, self.latest_frames, manual=True)

    def _save_frames(self, folder: Path, frames: Iterable[np.ndarray]) -> tuple[list[str], list[str]]:
        """Write the window frames the VLM reads; returns file paths and their page URLs."""
        folder.mkdir(parents=True, exist_ok=True)
        paths = []
        for index, frame in enumerate(frames):
            image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            scale = self.args.verifier_max_side / max(image.shape[:2])
            if scale < 1:
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            path = folder / f"{index:02d}.jpg"
            cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            paths.append(str(path.resolve()))
        urls = ["/alerts/" + str((folder / f"{index:02d}.jpg").relative_to(self.alerts_dir)) for index in range(len(paths))]
        return paths, urls

    def _raise_alert_locked(self, result: dict, frames: np.ndarray, manual: bool = False) -> str:
        alert_id = time.strftime("%Y%m%d-%H%M%S") + f"-{len(self.alerts):03d}"
        start = result["window"][0]
        context = self._sample_locked(start - self.config["window_s"], start, BEFORE_FRAMES)
        paths, urls = self._save_frames(self.alerts_dir / alert_id, [*context, *frames])
        thresholds = self.config["thresholds"]
        event = max(result["events"], key=lambda code: result["events"][code] / thresholds[code])
        alert = {"id": alert_id, "wall": result["wall"], "t": result["t"], "event": event,
                 "event_name": ALERT_EVENTS[event], "event_probability": result["events"][event],
                 "threshold": thresholds[event], "events": result["events"], "top": result["top"],
                 "priority": "manual" if manual else "high",
                 "kind": "manual" if manual else "alert", "step": result["step"],
                 "model_events": {name: value["events"] for name, value in result["models"].items()},
                 "frames": urls, "context_frames": len(context), "after_frames": 0, "verifier": {"status": "pending"}, "checks": [],
                 "status": "manual" if manual else "checking", "reason": None, "decided_wall": None, "next_check_wall": None}
        self.alerts.append(alert)
        if self.verifier and manual:
            self.verifier.submit(alert_id, paths)
        elif self.verifier and self.config["verifier"]:
            # Wait for the frames after the alert window; _watch_open_alerts sends the check.
            self.open[alert_id] = {"collecting": True, "waiting": False, "frames": [*context, *frames], "end": result["t"],
                                   "session": self.session, "deadline": time.monotonic() + self.config["window_s"] + AFTER_SLACK_S,
                                   "rechecks": []}
        else:
            alert["verifier"] = {"status": "disabled"}
            if not manual:
                self._decide_locked(alert, "notified", "Verifier is off")
        self._save_alert(alert)
        return alert_id

    def _save_alert(self, alert: dict) -> None:
        (self.alerts_dir / alert["id"] / "alert.json").write_text(json.dumps(alert, indent=2))

    def _decide_locked(self, alert: dict, status: str, reason: str) -> None:
        alert["status"], alert["reason"], alert["decided_wall"], alert["next_check_wall"] = status, reason, time.time(), None
        self.open.pop(alert["id"], None)
        if status == "notified" and self.notifier:
            self._push_locked(alert)

    def _push_locked(self, alert: dict) -> None:
        """Send the phone notification; its alert page loops the frames of the check that decided it."""
        check = alert["checks"][-1] if alert["checks"] else alert
        verifier = check["verifier"]
        evidence = (verifier.get("answer") or {}).get("evidence", "") if verifier.get("status") == "done" else ""
        title = f"Possible {alert['event_name']} detected"
        message = f"{evidence or 'The camera may have seen ' + alert['event_name'] + '.'} Please check on them now."
        paths = [self.alerts_dir / url.removeprefix("/alerts/") for url in check["frames"]]
        alert |= {"phone": {"status": "sending"}, "phone_text": {"title": title, "message": message}, "responses": []}

        def done(record: dict) -> None:
            with self.lock:
                alert["phone"] = record
                self._save_alert(alert)
        self.notifier.send(alert["id"], title + "!!!", lambda: [path.read_bytes() for path in paths], done)

    def send_test_notification(self) -> None:
        with self.lock:
            if self.notifier is None:
                raise ValueError("Phone notifications are off; set ntfy_topic in demo_local.json")
            frames = [] if self.latest_frames is None else [
                cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
                for f in self.latest_frames]
            text = {"title": "Test alert",
                    "message": "Phone notifications work." + (" The video shows the current camera view." if frames else "")}
            self.test_alert = {"id": "test", "wall": time.time(), "event_name": "test", "phone": {"status": "sending"},
                               "phone_text": text, "responses": []}
            test = self.test_alert

        def done(record: dict) -> None:
            with self.lock:
                test["phone"] = record
        self.notifier.send("test", "Test from the elder safety demo", lambda: frames, done)

    # ------------------------------------------------------------------ alert page (behind the tunnel)
    def _local(self) -> dict:
        """Personal settings; read on every alert page request, so edits show without a restart."""
        try:
            return json.loads(self.args.local.read_text(encoding="utf-8"))
        except (AttributeError, OSError, ValueError):
            return {}

    def _shared_alert_locked(self, alert_id: str) -> dict | None:
        if alert_id == "test":
            return self.test_alert
        return next((a for a in self.alerts if a["id"] == alert_id and "phone_text" in a), None)

    def page_state(self, alert_id: str, device: str) -> dict | None:
        person = self._local().get("person") or {}
        with self.lock:
            alert = self._shared_alert_locked(alert_id)
            if alert is None:
                return None
            responses = [{"name": r["name"], "action": r["action"], "wall": r["wall"], "you": bool(device) and r["device"] == device}
                         for r in alert["responses"]]
            return {"test": alert_id == "test", "wall": alert["wall"], "event_name": alert["event_name"], **alert["phone_text"],
                    "responses": responses, "person": person}

    def record_response(self, alert_id: str, device: str, action: str) -> None:
        with self.lock:
            alert = self._shared_alert_locked(alert_id)
            if alert is None:
                return
            # One "seen" per phone: reopening the page keeps the first time.
            if action == "seen" and any(r["action"] == "seen" and r["device"] == device for r in alert["responses"]):
                return
            if len(alert["responses"]) >= 50:
                return
            name = self.registrations.get(device, "Someone")
            alert["responses"].append({"device": device, "name": name, "action": action, "wall": time.time()})
            if alert_id != "test":
                self._save_alert(alert)

    def registration_state(self, device: str) -> dict:
        names = [str(r.get("name")) for r in self._local().get("relatives") or [] if r.get("name")]
        with self.lock:
            return {"names": names, "registered": self.registrations.get(device)}

    def register(self, device: str, name: str) -> None:
        """Name this phone; responses it already made on open alerts take the name too."""
        with self.lock:
            self.registrations[device] = name
            for alert in [*self.alerts, *([self.test_alert] if self.test_alert else [])]:
                for response in alert.get("responses", []):
                    if response["device"] == device:
                        response["name"] = name

    def _recheck_end(self, entry: dict, number: int) -> float:
        """Video time that recheck ``number`` ends on; its window starts ``recheck`` seconds after the check before it ended."""
        return entry["end"] + self.config["window_s"] + number * (self.config["recheck_s"] + self.config["window_s"])

    def _schedule_recheck_locked(self, alert: dict) -> None:
        entry = self.open[alert["id"]]
        entry["waiting"] = False
        wait = self._recheck_end(entry, len(alert["checks"]) + 1) - (self.last_frame_t or 0.0)
        alert["next_check_wall"] = time.time() + max(0.0, wait)

    def _verified(self, payload: dict) -> None:
        alert_id, _, recheck = payload["id"].partition("/r")
        answer = {"status": "error" if "error" in payload else "done", **payload}
        with self.lock:
            alert = next((a for a in self.alerts if a["id"] == alert_id), None)
            if alert is None:
                return
            normal = answer["status"] == "done" and answer.get("tier") == "dismissed"
            if not recheck:
                alert["verifier"] = answer
                if alert["status"] == "checking":
                    if normal:
                        alert["status"] = "on_hold"
                        self._schedule_recheck_locked(alert)
                    else:
                        self._decide_locked(alert, "notified", {"confirmed": "Verifier sees an alert event", "uncertain": "Verifier is unsure"}.get(
                            answer.get("tier"), "Verifier failed") if answer["status"] == "done" else "Verifier failed")
            else:
                check = alert["checks"][int(recheck) - 1]
                check["verifier"], check["passed"] = answer, normal
                if alert["status"] == "on_hold":
                    if not normal:
                        self._decide_locked(alert, "notified", f"Recheck {recheck} did not see a normal activity")
                    elif int(recheck) >= self.config["rechecks"]:
                        self._decide_locked(alert, "cleared", f"{recheck} rechecks in a row saw a normal activity")
                    else:
                        self._schedule_recheck_locked(alert)
            self._save_alert(alert)

    def _watch_open_alerts(self) -> None:
        """Start due rechecks and notify on any timeout, so an open alert can never stall silently."""
        while True:
            time.sleep(0.25)
            with self.lock:
                now = time.monotonic()
                for alert in [a for a in self.alerts if a["id"] in self.open]:
                    entry = self.open[alert["id"]]
                    self._sample_rechecks_locked(entry)
                    if entry["collecting"]:
                        if now >= entry["deadline"] or self._after_ready_locked(entry):
                            self._send_first_check_locked(alert, entry)
                        continue
                    if entry["waiting"]:
                        if now < entry["deadline"]:
                            continue
                        self._decide_locked(alert, "notified", "Verifier did not answer in time")
                    elif len(entry["rechecks"]) > len(alert["checks"]):
                        if self.verifier is None or self.verifier.status.startswith("exited"):
                            self._decide_locked(alert, "notified", "Verifier did not answer in time")
                        else:
                            self._start_recheck_locked(alert, entry)
                    elif entry["session"] != self.session or not self.received or now - self.received[-1] > FRESH_WINDOW_S:
                        self._decide_locked(alert, "ended", "No new video to recheck")
                    else:
                        continue
                    self._save_alert(alert)

    def _sample_rechecks_locked(self, entry: dict) -> None:
        """Sample each recheck window as soon as its frames arrive, even while the verifier is still busy.

        A slow verifier then checks the same frames as a fast one, only later, and frames it has not read yet
        cannot drop out of the short frame buffer."""
        while len(entry["rechecks"]) < self.config["rechecks"] and entry["session"] == self.session and self.frames:
            end = self._recheck_end(entry, len(entry["rechecks"]) + 1)
            if self.frames[-1][0] < end:
                return
            entry["rechecks"].append({"t": end, "wall": time.time(),
                                      "frames": self._sample_locked(end - self.config["window_s"], end + 1e-3, NUM_FRAMES)})

    def _after_ready_locked(self, entry: dict) -> bool:
        """The frames after the alert window have arrived, or the source changed and they never will."""
        return entry["session"] != self.session or (bool(self.frames) and self.frames[-1][0] >= entry["end"] + self.config["window_s"])

    def _send_first_check_locked(self, alert: dict, entry: dict) -> None:
        after = (self._sample_locked(entry["end"], entry["end"] + self.config["window_s"] + 1e-3, AFTER_FRAMES)
                 if entry["session"] == self.session else [])
        paths, alert["frames"] = self._save_frames(self.alerts_dir / alert["id"], [*entry["frames"], *after])
        alert["after_frames"] = len(after)
        entry |= {"collecting": False, "waiting": True, "frames": None, "deadline": time.monotonic() + VERIFIER_TIMEOUT_S}
        self.verifier.submit(alert["id"], paths)
        self._save_alert(alert)

    def _start_recheck_locked(self, alert: dict, entry: dict) -> None:
        number = len(alert["checks"]) + 1
        sample = entry["rechecks"][number - 1]
        paths, urls = self._save_frames(self.alerts_dir / alert["id"] / f"r{number}", sample["frames"])
        alert["checks"].append({"number": number, "wall": sample["wall"], "t": sample["t"], "frames": urls,
                                "verifier": {"status": "pending"}, "passed": None})
        entry |= {"deadline": time.monotonic() + VERIFIER_TIMEOUT_S, "waiting": True}
        alert["next_check_wall"] = None
        self.verifier.submit(f"{alert['id']}/r{number}", paths)

    # ------------------------------------------------------------------ output
    def state(self) -> dict:
        with self.lock:
            now = time.monotonic()
            fps = sum(now - t < 2.0 for t in self.received) / 2.0
            return {"stage1": self.stage1_status, "config": self.config, "session": self.session,
                    "defaults": {"system": self.args.system_defaults, "user": self.args.user_defaults},
                    "phone": None if self.notifier is None else self.notifier.state() | {
                        "test": self.test_alert, "registered": sorted(set(self.registrations.values()))},
                    "input_fps": fps, "buffer_frames": len(self.frames), "latest": self.latest,
                    "history": list(self.history), "steps": list(self.steps), "alerts": self.alerts[-30:],
                    "verifier": None if self.verifier is None else
                    {"model": self.verifier.model, "status": self.verifier.status, "pending": self.verifier.pending,
                     "prompt": PROMPT.format(n=NUM_FRAMES)}}

    def input_image(self, key: str) -> bytes | None:
        return self.images.get(key)

    def update_config(self, changes: dict) -> None:
        settings = parse_settings(changes)
        with self.lock:
            self.config["thresholds"] |= settings.pop("thresholds", {})
            self.config |= settings

    def save_user_defaults(self) -> None:
        """Store the current settings as the user defaults that the next server start applies."""
        with self.lock:
            saved = json.loads(json.dumps(self.config))
        self.args.settings.write_text(json.dumps(saved, indent=2) + "\n")
        self.args.user_defaults = saved

    def reset_to_system_defaults(self) -> None:
        """Apply the defaults written in the code and forget the saved user defaults."""
        self.args.settings.unlink(missing_ok=True)
        self.args.user_defaults = {}
        self.update_config(self.args.system_defaults)


def handler(monitor: Monitor):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def _send(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            # Frame images never change once written (step- or alert-keyed), so let the browser cache them.
            self.send_header("Cache-Control", "max-age=3600" if content_type == "image/jpeg" else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send(json.dumps(payload).encode(), "application/json", status)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                return self._send((STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/api/state":
                return self._json(monitor.state())
            if path.startswith("/api/input/"):
                image = monitor.input_image(path.removeprefix("/api/input/"))
                if image is not None:
                    return self._send(image, "image/jpeg")
            if path.startswith("/alerts/"):
                parts = path.split("/")[2:]  # <alert id>/[r<n>/]<nn>.jpg
                if len(parts) in (2, 3) and parts[-1].endswith(".jpg") and ".." not in path:
                    file = monitor.alerts_dir.joinpath(*parts)
                    if file.is_file():
                        return self._send(file.read_bytes(), "image/jpeg")
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            url = urlparse(self.path)
            query = {key: values[0] for key, values in parse_qs(url.query).items()}
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            try:
                if url.path == "/api/frame":
                    monitor.add_frame(float(query["t"]), body, int(query["session"]))
                elif url.path == "/api/reset":
                    monitor.reset(int(query["session"]))
                elif url.path == "/api/verify":
                    return self._json({"id": monitor.verify_now()})
                elif url.path == "/api/config":
                    monitor.update_config(json.loads(body))
                elif url.path == "/api/config/save":
                    monitor.save_user_defaults()
                elif url.path == "/api/config/reset":
                    monitor.reset_to_system_defaults()
                elif url.path == "/api/notify/test":
                    monitor.send_test_notification()
                else:
                    return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (KeyError, ValueError) as error:
                return self._json({"error": repr(error)}, HTTPStatus.BAD_REQUEST)
            self._json({"ok": True})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8642)
    parser.add_argument("--crop-aware", type=Path, default=Path("artifacts/patch16_crop_aware/seed42"))
    parser.add_argument("--person-trained", type=Path, default=Path("artifacts/patch16_person_crop/seed42"))
    parser.add_argument("--pose-model", type=Path, default=Path("artifacts/hand_roi_models/pose_landmarker_lite.task"))
    parser.add_argument("--device", default=None, help="torch device; default mps when available")
    parser.add_argument("--verifier-model", default="mlx-community/Qwen3.5-4B-MLX-4bit", help="mlx-vlm model id, or 'none'")
    parser.add_argument("--verifier-max-side", type=int, default=768)
    parser.add_argument("--fall-threshold", type=float, default=DEFAULT_THRESHOLDS["A043"])
    parser.add_argument("--vomit-threshold", type=float, default=DEFAULT_THRESHOLDS["A048"])
    parser.add_argument("--window", type=float, default=2.5, help="seconds of video per inference step")
    parser.add_argument("--persist", type=int, default=1, help="consecutive steps above threshold to raise an alert")
    parser.add_argument("--cooldown", type=float, default=15.0, help="seconds between alerts")
    parser.add_argument("--recheck", type=float, default=15.0, help="seconds between rechecks of an alert on hold")
    parser.add_argument("--rechecks", type=int, default=3, help="consecutive normal-activity rechecks that clear an alert on hold")
    parser.add_argument("--min-interval", type=float, default=0.5, help="minimum seconds between inference steps")
    parser.add_argument("--alerts-dir", type=Path, default=Path("cache/demo_alerts"))
    parser.add_argument("--ntfy-topic", default=None, help="ntfy topic for phone notifications; default ntfy_topic in --local, off when neither")
    parser.add_argument("--ntfy-server", default="https://ntfy.sh")
    parser.add_argument("--share-port", type=int, default=8643, help="local port of the alert page server behind the tunnel")
    parser.add_argument("--share-hours", type=float, default=24.0, help="hours an alert page link keeps working")
    parser.add_argument("--no-tunnel", action="store_true", help="send text-only notifications, without an alert page")
    parser.add_argument("--local", type=Path, default=Path("demo_local.json"),
                        help="personal settings ignored by git: ntfy_topic, and the person and relatives for the alert page")
    parser.add_argument("--settings", type=Path, default=Path("demo_settings.json"),
                        help="user defaults saved from the Settings pane; explicit command-line options override them")
    # System defaults are the values written above; saved user defaults replace them, and explicit options win.
    system_defaults = settings_from_args(parser.parse_args([]))
    settings_path = parser.parse_known_args()[0].settings
    user_defaults = parse_settings(json.loads(settings_path.read_text())) if settings_path.is_file() else {}
    saved = user_defaults | user_defaults.get("thresholds", {})
    parser.set_defaults(**{dest: saved[key] for key, dest in (SETTING_ARGS | THRESHOLD_ARGS).items() if key in saved})
    args = parser.parse_args()
    system_defaults["verifier"] = args.verifier_model != "none"  # checks cannot be on without a verifier
    args.system_defaults, args.user_defaults = system_defaults, user_defaults
    monitor = Monitor(args)
    server = ThreadingHTTPServer((args.host, args.port), handler(monitor))
    print(f"Demo running at http://{args.host}:{args.port}  (the detector and the verifier load in the background)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if monitor.verifier:
            monitor.verifier.close()
        if monitor.notifier and monitor.notifier.share:
            monitor.notifier.share.close()


if __name__ == "__main__":
    main()
