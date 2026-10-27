"""The main loop: source -> detector -> tracker -> analytics (-> store, VLM, UI, re-stream).

Threads: decoder reader (stream.py), this inference loop, the VLM worker (vlm.py) and the HTTP server
(server.py). The loop never waits for the VLM: VLM jobs are queued and answered asynchronously.
"""
import logging
import threading
import time
from collections import deque

import cv2
import numpy as np

from . import annotate, detector
from .analytics import Analytics
from .config import resolve
from .store import Store
from .stream import VideoSink, VideoSource
from .tracker import Tracker

log = logging.getLogger("edgevision")


class StageTimer:
    """Per-stage latency (ms): recent window for the UI, full list for the run report."""

    def __init__(self):
        self.recent, self.all = {}, {}

    def add(self, stage, ms):
        self.recent.setdefault(stage, deque(maxlen=200)).append(ms)
        self.all.setdefault(stage, []).append(ms)

    def summary(self, full=False):
        src = self.all if full else self.recent
        return {k: {"mean": round(float(np.mean(v)), 2), "p50": round(float(np.median(v)), 2),
                    "p90": round(float(np.percentile(v, 90)), 2), "n": len(v)} for k, v in src.items() if len(v)}


class Pipeline:
    def __init__(self, cfg, record_tracks=False):
        self.cfg = cfg
        self.source = VideoSource(cfg["source"])
        W, H, fps = self.source.W, self.source.H, self.source.fps
        log.info("source %s (%s, %s), %dx%d @ %.2f fps", cfg["source"]["uri"], self.source.kind,
                 "live" if self.source.live else "file", W, H, fps)
        self.detector = detector.build(cfg["detector"])
        self.tracker = Tracker(cfg["tracker"], fps)
        self.store = Store(resolve(cfg["store"]["path"]))
        self.analytics = Analytics(cfg["analytics"], cfg["alerts"], W, H, fps, on_event=self._on_event,
                                   live=self.source.live)
        self.policy = None
        if (cfg.get("policy") or {}).get("enabled", False):
            from .policy import Policy
            self.policy = Policy(cfg["policy"], self.store)
        self.agent = None
        if cfg["vlm"]["enabled"]:
            from .vlm import SceneAgent
            self.agent = SceneAgent(cfg["vlm"], self.analytics, self.store, self.latest_frames, self.policy)
        self.timer = StageTimer()
        self._lock = threading.Lock()
        self._raw = self._ann = self._jpeg = None
        self._ts = None
        self._jpeg_seq = 0
        self._frame_cv = threading.Condition()
        self._alert_banner = (None, 0.0)
        self._series_n = 0
        self.fps_meas = 0.0
        self.frames = 0
        self.running = False
        self.record = [] if record_tracks else None
        self.sink = None
        if cfg["output"]["uri"]:
            self.sink = VideoSink(resolve(cfg["output"]["uri"]), W, H, cfg["output"]["fps"] or fps,
                                  cfg["output"]["bitrate"])
        self.server = None
        if cfg["server"]["enabled"]:
            from .server import Server
            self.server = Server(self, cfg["server"])
        self.need_annotation = bool(self.sink or self.server or self.agent)

    # ---------------------------------------------------------------- shared state
    def latest_frames(self):
        with self._lock:
            return self._raw, self._ann, self._ts

    def latest_jpeg(self, after_seq=-1, timeout=1.0):
        """Block until a JPEG newer than `after_seq` exists; -> (seq, bytes)."""
        with self._frame_cv:
            if self._jpeg_seq <= after_seq:
                self._frame_cv.wait(timeout)
            return self._jpeg_seq, self._jpeg

    def _on_event(self, e):
        self.store.event(e)
        if e["kind"] == "alert":
            self._alert_banner = (f"ALERT: {e['rule']}", time.time())
            log.warning("alert %s (%s=%s)", e["rule"], e["metric"], e["value"])
            if self.agent and self.cfg["vlm"]["verify_alerts"]:
                self.agent.submit(1, self.agent.verify, e)

    # ---------------------------------------------------------------- loop
    def run(self, max_frames=0):
        self.running = True
        if self.agent:
            self.agent.start()
        if self.server:
            self.server.start()
        vcfg = self.cfg["vlm"]
        last_obs = time.time()
        t_prev, t_start = None, time.time()
        live = self.source.live
        try:
            for fr in self.source:
                t0 = time.perf_counter()
                boxes, scores = self.detector(fr.image)
                t1 = time.perf_counter()
                tb, ids = self.tracker.update(boxes, scores, fr.ts if live else None)
                t2 = time.perf_counter()
                self.analytics.update(fr.ts, tb, ids)
                t3 = time.perf_counter()
                if self.record is not None:
                    self.record.append((fr.index, fr.ts, tb.copy(), ids.copy(), boxes, scores))
                self.timer.add("detect_ms", 1000 * (t1 - t0))
                self.timer.add("track_ms", 1000 * (t2 - t1))
                self.timer.add("analytics_ms", 1000 * (t3 - t2))
                if len(self.analytics.series) != self._series_n:      # a 1-s bin closed -> persist it
                    self._series_n = len(self.analytics.series)
                    if self._series_n:
                        self.store.series(self.analytics.series_list()[-1])
                if self.need_annotation:
                    banner, since = self._alert_banner
                    snap = self.analytics.snapshot(recent_events=0)
                    snap["zones_px"] = self.analytics.zones
                    ann = annotate.draw(fr.image, tb, ids, snap, self.fps_meas,
                                        banner if banner and time.time() - since < 5 else None)
                    t4 = time.perf_counter()
                    self.timer.add("annotate_ms", 1000 * (t4 - t3))
                    with self._lock:
                        self._raw, self._ann, self._ts = fr.image, ann, fr.ts
                    if self.server:
                        self._publish_jpeg(ann)
                    if self.sink:
                        self.sink.write(ann)
                now = time.time()
                if t_prev is not None:
                    dt = now - t_prev
                    self.fps_meas = 0.9 * self.fps_meas + 0.1 * (1 / dt) if self.fps_meas else 1 / dt
                t_prev = now
                self.timer.add("loop_ms", 1000 * (time.perf_counter() - t0))
                self.frames += 1
                if self.agent and vcfg["observe_every_s"] and now - last_obs >= vcfg["observe_every_s"] \
                        and self.agent.pending() == 0:
                    last_obs = now
                    self.agent.submit(2, self.agent.observe, "periodic")
                if self.frames % 18000 == 0:                  # ~10 min at 30 fps: enforce data retention
                    self.store.purge(self.cfg["store"].get("retention_h", 0))
                if self.frames % 300 == 0:
                    s = self.analytics.snapshot(recent_events=0)
                    log.info("t=%.1fs frames=%d fps=%.1f occupancy=%d unique=%d crossings=%s detect p50=%.1fms",
                             fr.ts, self.frames, self.fps_meas, s["occupancy"], s["unique_people"], s["crossings"],
                             self.timer.summary()["detect_ms"]["p50"])
                if max_frames and self.frames >= max_frames:
                    break
                if not self.running:
                    break
        finally:
            self.wall_s = time.time() - t_start
            self.stop()

    def _publish_jpeg(self, img):
        sc = self.cfg["server"]
        h, w = img.shape[:2]
        if w > sc["stream_max_width"]:
            img = cv2.resize(img, (sc["stream_max_width"], round(h * sc["stream_max_width"] / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, sc["jpeg_quality"]])
        with self._frame_cv:
            self._jpeg, self._jpeg_seq = buf.tobytes(), self._jpeg_seq + 1
            self._frame_cv.notify_all()

    def stop(self):
        if not self.running:
            return
        self.running = False
        self.source.stop()
        if self.sink:
            self.sink.close()
        if self.agent:
            self.agent.stop()
        if self.server:
            self.server.stop()

    def report(self):
        """Summary of a finished run (for benchmarks and the paper)."""
        snap = self.analytics.snapshot(recent_events=0)
        return {"frames": self.frames, "wall_s": round(getattr(self, "wall_s", 0), 2),
                "throughput_fps": round(self.frames / max(getattr(self, "wall_s", 1e-9), 1e-9), 2),
                "source": {"uri": self.cfg["source"]["uri"], "kind": self.source.kind, "live": self.source.live,
                           "fps": self.source.fps, "arrival_fps": self.source.arrival_fps,
                           "dropped": self.source.dropped},
                "detector": {"backend": self.cfg["detector"]["backend"], "weights": self.cfg["detector"]["weights"],
                             "provider": getattr(self.detector, "provider", None)},
                "latency_ms": self.timer.summary(full=True),
                "analytics": {k: snap[k] for k in ("t", "unique_people", "crossings", "crossing_directions",
                                                   "dwell_finished_median_s", "dwell_finished_p90_s")}}
