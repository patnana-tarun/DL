"""Web UI + JSON API on the device (Python standard library only).

  GET  /                  operator UI (live video, measurements, events, VLM reports, chat)
  GET  /video.mjpg        annotated live video (MJPEG)
  GET  /snapshot.jpg      latest annotated frame
  GET  /api/state         measurements, pipeline stats, VLM status, recent reports
  GET  /api/series?seconds=600      occupancy time series
  GET  /api/events?minutes=10       stored events
  GET  /api/chat          chat log
  GET  /heatmap.jpg       foot-traffic heatmap over the latest frame
  Auth: with server.token and/or server.users ([{name, token, role}]) every request needs ?token=<token> or
  the header X-Token: <token>; the token selects the principal (server.token = admin). Without either, every
  client is "anonymous" with server.anonymous_role. Roles (policy.py): viewer = read and ask, operator = change
  analytics, admin = also remove all alerts. Chat requests are rate-limited per principal (HTTP 429).
  POST /api/chat          {"message": "..."} -> {"answer", "actions", ...}  ("/" commands or the VLM)
"""
import json
import logging
import threading

import cv2
import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import commands

log = logging.getLogger("edgevision.server")
UI = Path(__file__).parent / "ui" / "index.html"


class Server:
    def __init__(self, pipeline, cfg):
        self.p, self.cfg = pipeline, cfg
        handler = type("H", (_Handler,), {"pipeline": pipeline})
        self.httpd = ThreadingHTTPServer((cfg["host"], cfg["port"]), handler)
        self.httpd.daemon_threads = True
        self._t = threading.Thread(target=self.httpd.serve_forever, name="http", daemon=True)

    def start(self):
        self._t.start()
        host = "localhost" if self.cfg["host"] in ("0.0.0.0", "") else self.cfg["host"]
        log.info("UI on http://%s:%d", host, self.httpd.server_address[1])

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def state(p):
    snap = p.analytics.snapshot(recent_events=15)
    a = p.agent
    return {
        "measurements": snap,
        "pipeline": {"fps": round(p.fps_meas, 2), "frames": p.frames, "latency_ms": p.timer.summary(),
                     "source": p.cfg["source"]["uri"], "live": p.source.live, "dropped": p.source.dropped,
                     "detector": getattr(p.detector, "provider", "")},
        "vlm": None if a is None else {"online": a.online, "busy": a.busy, "pending": a.pending(),
                                       "last_latency_s": a.last_latency, "model": a.cfg["model"],
                                       "endpoint": a.cfg["endpoint"]},
        "observations": p.store.observations(6),
    }


def resolve_principal(cfg, token):
    """Token -> policy principal, or None if the token is required and wrong."""
    from .policy import principal
    users = cfg.get("users") or []
    if cfg.get("token") and token == cfg["token"]:
        return principal("admin", "admin")
    for u in users:
        if token == u["token"]:
            return principal(u.get("name", "user"), u["role"])
    if not cfg.get("token") and not users:
        return principal("anonymous", cfg.get("anonymous_role", "operator"))
    return None


def chat(p, message, who=None):
    message = (message or "").strip()
    if not message:
        return {"answer": "empty message"}
    r = commands.handle(message, p, who)
    if r is not None:
        p.store.chat("user", message); p.store.chat("assistant", r["answer"])
        return r
    if p.agent is None:
        return {"answer": "The VLM is disabled (vlm.enabled: false). Commands still work: try /help"}
    policy = getattr(p, "policy", None)
    if policy is not None:
        ok, why = policy.allow_request(who, p.agent.pending())
        if not ok:
            return {"answer": why, "error": True, "status": 429}
    fut = p.agent.submit(0, p.agent.ask, message, who) if policy is not None else p.agent.submit(0, p.agent.ask, message)
    try:
        return fut.result(timeout=p.agent.cfg["timeout_s"] + 10)
    except Exception as e:  # noqa: BLE001
        return {"answer": f"VLM error: {type(e).__name__}: {e}. Is the VLM server running at "
                          f"{p.agent.cfg['endpoint']}? Commands still work: try /help", "error": True}


class _Handler(BaseHTTPRequestHandler):
    pipeline = None
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        log.debug(fmt, *args)

    def _send(self, code, body, ctype="application/json"):
        if not isinstance(body, (bytes, bytearray)):
            body = json.dumps(body, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _authorised(self, q):
        """-> principal, or None after answering 401."""
        token = q.get("token", [""])[0] or self.headers.get("X-Token") or ""
        who = resolve_principal(self.pipeline.cfg["server"], token)
        if who is None:
            self._send(401, {"error": "token required"})
        return who

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path); q = parse_qs(u.query); p = self.pipeline
        if self._authorised(q) is None:
            return None
        try:
            if u.path in ("/", "/index.html"):
                return self._send(200, UI.read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/state":
                return self._send(200, state(p))
            if u.path == "/api/series":
                return self._send(200, p.analytics.series_list(float(q.get("seconds", ["600"])[0])))
            if u.path == "/api/events":
                return self._send(200, p.store.recent_events(float(q.get("minutes", ["10"])[0])))
            if u.path == "/api/chat":
                return self._send(200, p.store.chat_log(50))
            if u.path == "/snapshot.jpg":
                _, jpg = p.latest_jpeg(timeout=2.0)
                return self._send(200, jpg, "image/jpeg") if jpg else self._send(503, {"error": "no frame yet"})
            if u.path == "/heatmap.jpg":
                return self._heatmap()
            if u.path == "/video.mjpg":
                return self._mjpeg()
            return self._send(404, {"error": "not found"})
        except (BrokenPipeError, ConnectionResetError):
            return None

    def do_POST(self):  # noqa: N802
        u = urlparse(self.path)
        who = self._authorised(parse_qs(u.query))
        if who is None:
            return None
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, {"error": "invalid JSON"})
        if u.path == "/api/chat":
            r = chat(self.pipeline, body.get("message", ""), who)
            return self._send(r.pop("status", 200), r)
        return self._send(404, {"error": "not found"})

    def _heatmap(self):
        raw, _, _ = self.pipeline.latest_frames()
        if raw is None:
            return self._send(503, {"error": "no frame yet"})
        H, W = raw.shape[:2]
        h = self.pipeline.analytics.heatmap()
        h = cv2.GaussianBlur(cv2.resize(h, (W, H), interpolation=cv2.INTER_LINEAR), (0, 0), W / 64)
        h = (255 * h / max(float(h.max()), 1e-6)).astype(np.uint8)
        over = cv2.addWeighted(raw, 0.55, cv2.applyColorMap(h, cv2.COLORMAP_INFERNO), 0.45, 0)
        ok, buf = cv2.imencode(".jpg", over, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return self._send(200, buf.tobytes(), "image/jpeg")

    def _mjpeg(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        seq = -1
        p = self.pipeline
        while p.running:
            seq, jpg = p.latest_jpeg(seq, timeout=1.0)
            if not jpg:
                continue
            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                             + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
        self.close_connection = True
