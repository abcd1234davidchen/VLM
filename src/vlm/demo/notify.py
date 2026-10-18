"""Phone notifications for the live demo: an ntfy push that links to an alert page on this computer.

Only alerts that end as notified are pushed. Pushes carry only a title and a link, here to an alert page
that loops the frames the verifier saw, marks the alert as seen by the phone that opens it, and has a
demo emergency button. When the tunnel is up, a first push asks each phone to register a name, so a
"seen" says who saw it. Nothing is uploaded: a separate share server on localhost answers only paths
under random tokens, and a Cloudflare Quick Tunnel (cloudflared, no account or domain) makes that one
server reachable from the phone. The tunnel address changes on every start, so old links stop working
and phones register again after each restart.
"""

from __future__ import annotations

import json
import re
import secrets
import shutil
import subprocess
import threading
import time
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TUNNEL_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
DNS_QUERY = "https://cloudflare-dns.com/dns-query?type=A&name="  # DNS over HTTPS, JSON answers
DNS_FIRST_ASK_S = 5.0  # after the tunnel registers
DNS_WAIT_S = 60.0
STATIC = Path(__file__).with_name("static")
ACTIONS = ("seen", "emergency")  # what a phone can record on an alert page
MAX_NAME = 40
MAX_BODY = 1024
NO_BODY = "\u200b"  # ntfy replaces an empty or blank message with "triggered"; a zero-width space shows nothing


class AlertShare:
    """Serves alert pages and one registration page at unguessable paths through a Cloudflare Quick Tunnel.

    Alert pages: ``/a/<token>`` (page), ``/state``, ``/frames/<n>.jpg`` and ``POST /respond`` under it.
    Registration: ``/r/<token>`` (page), ``/state`` and ``POST /register`` under it. Pages post from
    JavaScript, so link previews and prefetches never count as a phone. ``app`` is the demo server and
    answers ``page_state(alert_id, device)``, ``record_response(alert_id, device, action)``,
    ``registration_state(device)`` and ``register(device, name)``.
    """

    def __init__(self, port: int, ttl_s: float, log_path: Path, app, tunnel: bool = True) -> None:
        self.ttl_s, self.app = ttl_s, app
        self.lock = threading.Lock()
        self.pages: dict[str, dict] = {}  # token -> {"alert_id", "frames", "expiry" (monotonic)}
        self.register_token = secrets.token_urlsafe(24)  # valid while this server runs
        self.public_url: str | None = None
        self.on_ready = []  # called once the public URL resolves
        self.process: subprocess.Popen | None = None
        self.server = ThreadingHTTPServer(("127.0.0.1", port), self._handler())
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        if not tunnel:
            self.status = "off"
        elif shutil.which("cloudflared") is None:
            self.status = "failed: cloudflared not found"
        else:
            self.status = "starting"
            self.log = log_path.open("a")
            self.process = subprocess.Popen(
                ["cloudflared", "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            threading.Thread(target=self._read_tunnel, daemon=True).start()

    @property
    def register_url(self) -> str | None:
        return f"{self.public_url}/r/{self.register_token}" if self.public_url else None

    def _read_tunnel(self) -> None:
        url = None
        for line in self.process.stderr:
            self.log.write(line)
            self.log.flush()
            match = TUNNEL_URL.search(line)
            url = url or (match.group(0) if match else None)
            # Ask DNS only once the tunnel is registered: an earlier "no such name" answer gets cached for a minute.
            if url and "Registered tunnel connection" in line and self.status == "starting":
                self.status = "waiting for DNS"
                threading.Thread(target=self._wait_for_dns, args=(url,), daemon=True).start()
        self.public_url, self.status = None, f"failed: cloudflared exited ({self.process.wait()})"

    def _wait_for_dns(self, url: str) -> None:
        """A new tunnel name takes about 10 s to resolve. Links are handed out only after that, so no
        resolver caches the name as missing."""
        request = urllib.request.Request(DNS_QUERY + url.removeprefix("https://"), headers={"Accept": "application/dns-json"})
        deadline = time.monotonic() + DNS_WAIT_S
        time.sleep(DNS_FIRST_ASK_S)
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(request, timeout=5) as response:
                    if json.loads(response.read()).get("Answer"):
                        break
            except Exception:  # keep waiting
                pass
            time.sleep(2)
        if self.status == "waiting for DNS":
            self.public_url, self.status = url, "ready"
            for callback in self.on_ready:
                callback()

    def add(self, alert_id: str, frames: list[bytes]) -> str | None:
        """Share one alert with its JPEG frames; returns the page URL, or None while the tunnel is not ready."""
        if self.public_url is None:
            return None
        token = secrets.token_urlsafe(24)
        with self.lock:
            now = time.monotonic()
            self.pages = {key: page for key, page in self.pages.items() if page["expiry"] > now}
            self.pages[token] = {"alert_id": alert_id, "frames": frames, "expiry": now + self.ttl_s}
        return f"{self.public_url}/a/{token}"

    def _page(self, token: str) -> dict | None:
        with self.lock:
            page = self.pages.get(token)
        return page if page and page["expiry"] > time.monotonic() else None

    def _handler(self):
        share = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def _send(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK, with_body: bool = True,
                      cache: str = "private, no-store") -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                # "private": nothing on the way (Cloudflare's edge included) may keep a copy.
                self.send_header("Cache-Control", cache)
                self.send_header("X-Robots-Tag", "noindex")
                self.send_header("Referrer-Policy", "no-referrer")
                self.end_headers()
                if with_body:
                    self.wfile.write(body)

            def _gone(self, with_body: bool = True) -> None:
                self._send(b"This link has expired.", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND, with_body)

            def _json(self, payload: object, with_body: bool = True) -> None:
                self._send(json.dumps(payload).encode(), "application/json", with_body=with_body)

            def _parts(self) -> tuple[list[str], dict]:
                path, _, query = self.path.partition("?")
                params = dict(item.partition("=")[::2] for item in query.split("&") if item)
                return path.strip("/").split("/"), params

            def _registration(self, parts: list[str], length: int) -> bool:
                return len(parts) == length and parts[0] == "r" and secrets.compare_digest(parts[1], share.register_token)

            def _get(self, with_body: bool) -> None:
                parts, params = self._parts()
                device = params.get("device", "")[:64]
                if self._registration(parts, 2):
                    return self._send((STATIC / "register.html").read_bytes(), "text/html; charset=utf-8", with_body=with_body)
                if self._registration(parts, 3) and parts[2] == "state":
                    return self._json(share.app.registration_state(device), with_body)
                page = share._page(parts[1]) if parts[0] == "a" and len(parts) >= 2 else None
                if page is None:
                    return self._gone(with_body)
                if len(parts) == 2:
                    return self._send((STATIC / "alert.html").read_bytes(), "text/html; charset=utf-8", with_body=with_body)
                if parts[2:] == ["state"]:
                    state = share.app.page_state(page["alert_id"], device)
                    if state is not None:
                        return self._json(state | {"frames": len(page["frames"])}, with_body)
                if len(parts) == 4 and parts[2] == "frames" and parts[3].removesuffix(".jpg").isdigit():
                    index = int(parts[3].removesuffix(".jpg"))
                    if index < len(page["frames"]):
                        # Only the phone may cache a frame, so the loop does not download it again.
                        return self._send(page["frames"][index], "image/jpeg", with_body=with_body, cache="private, max-age=86400, immutable")
                self._gone(with_body)

            def do_GET(self) -> None:
                self._get(True)

            def do_HEAD(self) -> None:
                self._get(False)

            def do_POST(self) -> None:
                parts, _ = self._parts()
                length = int(self.headers.get("Content-Length", 0))
                try:
                    body = json.loads(self.rfile.read(length)) if 0 < length <= MAX_BODY else None
                    device = str(body.get("device") or "")[:64]
                except (ValueError, AttributeError):
                    body, device = None, ""
                if body is None or not device:
                    return self._send(b"", "text/plain", HTTPStatus.BAD_REQUEST)
                if self._registration(parts, 3) and parts[2] == "register":
                    name = str(body.get("name") or "").strip()[:MAX_NAME]
                    if not name:
                        return self._send(b"", "text/plain", HTTPStatus.BAD_REQUEST)
                    share.app.register(device, name)
                    return self._json({})
                page = share._page(parts[1]) if parts[0] == "a" and len(parts) == 3 and parts[2] == "respond" else None
                if page is None:
                    return self._gone()
                if body.get("action") not in ACTIONS:
                    return self._send(b"", "text/plain", HTTPStatus.BAD_REQUEST)
                share.app.record_response(page["alert_id"], device, body["action"])
                self._json({})

        return Handler

    def close(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
        self.server.shutdown()


class Notifier:
    """Publishes to one ntfy topic; anyone subscribed to the topic in the ntfy app gets the push."""

    def __init__(self, topic: str, server_url: str, share: AlertShare | None) -> None:
        self.topic, self.server_url, self.share = topic, server_url.rstrip("/"), share
        self.registration: dict | None = None  # delivery record of the startup registration push
        if share:
            share.on_ready.append(self._ask_to_register)

    def state(self) -> dict:
        return {"link": self.share.status if self.share else "off", "registration": self.registration}

    def _ask_to_register(self) -> None:
        url = self.share.register_url
        self.registration = {"status": "sending"}
        self.registration = self._publish({
            "title": "Register this phone", "message": NO_BODY, "priority": 3, "tags": ["wave"], "click": url,
            "actions": [{"action": "view", "label": "Register", "url": url}]})

    def send(self, alert_id: str, title: str, load_frames, on_done) -> None:
        """Publish in the background. ``load_frames()`` returns the check's frames as JPEG bytes;
        ``on_done`` receives the delivery record."""
        threading.Thread(target=self._send, args=(alert_id, title, load_frames, on_done), daemon=True).start()

    def _send(self, alert_id: str, title: str, load_frames, on_done) -> None:
        page = None
        if self.share and self.share.public_url:
            try:
                frames = load_frames() or []
            except Exception as failure:  # still share the page
                print(f"Notification frames failed: {failure!r}", flush=True)
                frames = []
            page = self.share.add(alert_id, frames)
        payload = {"title": title, "message": NO_BODY, "priority": 5, "tags": ["warning"]}  # details are on the page
        if page:
            payload |= {"click": page, "actions": [{"action": "view", "label": "Open alert", "url": page}]}
        on_done(self._publish(payload) | {"page": page is not None})

    def _publish(self, payload: dict) -> dict:
        request = urllib.request.Request(self.server_url, data=json.dumps({"topic": self.topic} | payload).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
        error = None
        for attempt in range(2):
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    response.read()
                error = None
                break
            except Exception as failure:  # report and retry once
                error = repr(failure)
                time.sleep(2)
        return {"status": "failed" if error else "sent", "wall": time.time(), "error": error}
