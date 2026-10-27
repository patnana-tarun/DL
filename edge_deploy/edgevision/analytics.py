"""Online crowd analytics from tracker output (one call per frame), plus rule-based alerts.

Definitions (kept identical to the offline study where one exists, research/analytics.py):
  occupancy        tracked people in the current frame
  unique people    tracks that lasted at least `min_track_s` (0.25 s, the pseudo-GT filter). An upper bound
                   on distinct people: a person whose track breaks and restarts is counted twice.
  dwell            first-to-last-seen time of a finished track
  line crossing    the foot point (bottom-centre of the box) changes side of the counting segment; a side
                   only counts once the foot is `line_hysteresis` box heights past the line, so jitter of a
                   person standing on the line is not a crossing. Directions "a_to_b" / "b_to_a" get readable
                   labels from the line geometry (e.g. "right to left" for the default vertical line).
  zones            foot point inside a polygon
  speed            foot-point speed in image pixels per second (includes camera motion on a moving camera)
All coordinates are image-plane; nothing here is a world-frame measurement.
"""
import math
import operator
import threading
import time
from collections import deque

import cv2
import numpy as np

OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le, "==": operator.eq}
METRICS = ("occupancy", "unique_people", "crossings_total", "crossings_a_to_b", "crossings_b_to_a",
           "crossings_last_min", "max_dwell_s", "loitering", "zone:<name>")
LOST_S = 2.0          # a track unseen this long is finished


class _Track:
    __slots__ = ("id", "first", "last", "n", "foot", "h", "side", "hist", "zones", "loiter", "counted")

    def __init__(self, tid, ts):
        self.id, self.first, self.last, self.n = tid, ts, ts, 0
        self.foot, self.h, self.side = None, 0.0, 0
        self.hist = deque(maxlen=64)            # (ts, x, y) foot points for speed
        self.zones, self.loiter, self.counted = set(), False, False


class Analytics:
    def __init__(self, cfg, alerts, width, height, fps, on_event=None, live=False):
        self.cfg, self.W, self.H, self.fps = cfg, width, height, fps
        self.on_event = on_event or (lambda e: None)
        self.lock = threading.RLock()
        self.min_frames = max(1, round(cfg["min_track_s"] * fps))
        self.live = live                        # live: frames can be dropped, so track length is measured in time
        self.tracks, self.finished_dwell = {}, deque(maxlen=5000)
        self.unique = 0
        self.cross = {"a_to_b": 0, "b_to_a": 0}
        self.cross_times = deque(maxlen=10000)
        gx, gy = cfg["heatmap_grid"]
        self.heat = np.zeros((gy, gx), np.float32)
        nbins = int(cfg["history_s"] / cfg["bin_s"])
        self.series = deque(maxlen=nbins)       # (t_start, occ_mean, occ_max, unique, crossings)
        self._bin = None
        self.rules = []
        for r in alerts:
            self.add_rule(r)
        self.set_line(cfg["line"])
        self.set_zones(cfg["zones"])
        self.ts, self.occupancy, self.frames = 0.0, 0, 0
        self.events = deque(maxlen=200)

    # ---------------------------------------------------------------- configuration (thread-safe, runtime)
    def set_line(self, line):
        (x1, y1), (x2, y2) = line
        if (x1, y1) == (x2, y2):
            raise ValueError("the counting line needs two different points")
        with self.lock:
            self.line_norm = [[float(x1), float(y1)], [float(x2), float(y2)]]
            self.p1 = np.array([x1 * self.W, y1 * self.H], float)
            self.p2 = np.array([x2 * self.W, y2 * self.H], float)
            for t in self.tracks.values():
                t.side = 0
            d = self.p2 - self.p1
            n = np.array([-d[1], d[0]])               # points to the positive ("b") side
            if abs(n[0]) >= abs(n[1]):
                pos, neg = ("right", "left") if n[0] > 0 else ("left", "right")
            else:
                pos, neg = ("bottom", "top") if n[1] > 0 else ("top", "bottom")
            self.direction_labels = {"a_to_b": f"{neg} to {pos}", "b_to_a": f"{pos} to {neg}"}

    def set_zones(self, zones):
        with self.lock:
            self.zones_norm = {k: [[float(x), float(y)] for x, y in v] for k, v in zones.items()}
            self.zones = {k: (np.array(v, np.float32) * [self.W, self.H]).astype(np.float32)
                          for k, v in self.zones_norm.items()}

    def add_rule(self, r):
        metric = r["metric"]
        if metric not in METRICS and not metric.startswith("zone:"):
            raise ValueError(f"unknown metric {metric!r}; use one of {', '.join(METRICS)}")
        if r["op"] not in OPS:
            raise ValueError(f"unknown operator {r['op']!r}; use one of {' '.join(OPS)}")
        rule = {"name": str(r.get("name") or f"{metric} {r['op']} {float(r['value']):g}"),
                "metric": metric, "op": r["op"],
                "value": float(r["value"]), "for_s": float(r.get("for_s", 0)), "cooldown_s": float(r.get("cooldown_s", 60)),
                "_since": None, "_last_fire": -1e9, "active": False}
        with self.lock:
            self.rules = [x for x in self.rules if x["name"] != rule["name"]] + [rule]
        return {k: v for k, v in rule.items() if not k.startswith("_")}

    def remove_rule(self, name=None):
        with self.lock:
            n = len(self.rules)
            self.rules = [] if name is None else [r for r in self.rules if r["name"] != name]
            return n - len(self.rules)

    def rules_public(self):
        with self.lock:
            return [{k: v for k, v in r.items() if not k.startswith("_")} for r in self.rules]

    # ---------------------------------------------------------------- per frame
    def _emit(self, kind, ts, **data):
        e = {"ts": round(ts, 2), "wall": time.time(), "kind": kind, **data}
        self.events.append(e)
        self.on_event(e)

    def _side(self, foot, h):
        d = self.p2 - self.p1
        L = math.hypot(*d) + 1e-9
        rel = foot - self.p1
        along = float(rel @ d) / (L * L)
        if along < 0 or along > 1:                  # beyond the ends of the segment
            return 0
        dist = float(d[0] * rel[1] - d[1] * rel[0]) / L     # signed distance, + = the "b" side (see set_line)
        if abs(dist) < self.cfg["line_hysteresis"] * h:
            return 0
        return 1 if dist > 0 else -1

    def update(self, ts, boxes, ids):
        with self.lock:
            self.ts, self.frames, self.occupancy = ts, self.frames + 1, len(ids)
            gy, gx = self.heat.shape
            for (x1, y1, x2, y2), tid in zip(boxes, ids):
                tid = int(tid)
                t = self.tracks.get(tid)
                if t is None:
                    t = self.tracks[tid] = _Track(tid, ts)
                t.last, t.n = ts, t.n + 1
                foot = np.array([(x1 + x2) / 2, y2], float); h = float(y2 - y1)
                t.foot, t.h = foot, h
                t.hist.append((ts, foot[0], foot[1]))
                self.heat[min(gy - 1, max(0, int(foot[1] / self.H * gy))),
                          min(gx - 1, max(0, int(foot[0] / self.W * gx)))] += 1
                long_enough = (ts - t.first >= self.cfg["min_track_s"] and t.n >= 2) if self.live \
                    else t.n >= self.min_frames
                if not t.counted and long_enough:
                    t.counted = True; self.unique += 1
                    self._emit("person_enter", ts, track=tid, unique=self.unique)
                s = self._side(foot, h)
                if s:
                    if t.side and s != t.side and t.counted:
                        direction = "a_to_b" if t.side < 0 < s else "b_to_a"
                        self.cross[direction] += 1; self.cross_times.append(ts)
                        self._emit("line_cross", ts, track=tid, direction=direction, **self.cross)
                    t.side = s
                inside = {z for z, poly in self.zones.items()
                          if cv2.pointPolygonTest(poly, (float(foot[0]), float(foot[1])), False) >= 0}
                for z in inside - t.zones:
                    self._emit("zone_enter", ts, track=tid, zone=z)
                for z in t.zones - inside:
                    self._emit("zone_exit", ts, track=tid, zone=z)
                t.zones = inside
                if t.counted and not t.loiter and ts - t.first >= self.cfg["loiter_s"]:
                    t.loiter = True
                    self._emit("loitering", ts, track=tid, dwell_s=round(ts - t.first, 1))
            for tid in [k for k, t in self.tracks.items() if ts - t.last > LOST_S]:
                t = self.tracks.pop(tid)
                if t.counted:
                    dwell = t.last - t.first + 1 / self.fps
                    self.finished_dwell.append(dwell)
                    self._emit("person_exit", ts, track=tid, dwell_s=round(dwell, 2))
            self._accumulate_bin(ts)
            self._check_rules(ts)

    def _accumulate_bin(self, ts):
        b = self.cfg["bin_s"]
        start = math.floor(ts / b) * b
        if self._bin is None or self._bin[0] != start:
            if self._bin is not None:
                s, occ, n, cross0 = self._bin
                self.series.append((s, float(np.mean(occ)), int(max(occ)), self.unique,
                                    sum(self.cross.values()) - cross0))
            self._bin = (start, [], 0, sum(self.cross.values()))
        self._bin[1].append(self.occupancy)

    def metric(self, name):
        if name == "occupancy":
            return self.occupancy
        if name == "unique_people":
            return self.unique
        if name == "crossings_total":
            return sum(self.cross.values())
        if name in ("crossings_a_to_b", "crossings_b_to_a"):
            return self.cross[name.split("_", 1)[1]]
        if name == "crossings_last_min":
            return sum(1 for t in self.cross_times if self.ts - t <= 60)
        if name == "max_dwell_s":
            return max((self.ts - t.first for t in self.tracks.values() if t.counted), default=0.0)
        if name == "loitering":
            return sum(1 for t in self.tracks.values() if t.loiter and self.ts - t.last <= 1.0)
        if name.startswith("zone:"):
            z = name[5:]
            return sum(1 for t in self.tracks.values() if z in t.zones and self.ts - t.last <= 1.0 / self.fps + 1e-6)
        raise KeyError(name)

    def _check_rules(self, ts):
        for r in self.rules:
            try:
                ok = OPS[r["op"]](self.metric(r["metric"]), r["value"])
            except KeyError:
                ok = False
            if not ok:
                r["_since"], r["active"] = None, False
                continue
            r["_since"] = ts if r["_since"] is None else r["_since"]
            if ts - r["_since"] >= r["for_s"] and ts - r["_last_fire"] >= r["cooldown_s"]:
                r["_last_fire"], r["active"] = ts, True
                self._emit("alert", ts, rule=r["name"], metric=r["metric"], value=self.metric(r["metric"]),
                           threshold=r["value"], op=r["op"])

    # ---------------------------------------------------------------- read side
    def snapshot(self, recent_events=10):
        """Compact JSON-able state, used by the UI, the VLM context and the API."""
        with self.lock:
            ts = self.ts
            series = list(self.series)

            def window(sec):
                rows = [r for r in series if ts - r[0] <= sec]
                return {"mean": round(float(np.mean([r[1] for r in rows])), 2) if rows else None,
                        "max": int(max(r[2] for r in rows)) if rows else None}
            w1, w5 = window(60), window(300)
            prev = [r[1] for r in series if 60 < ts - r[0] <= 120]
            trend = None
            if prev and w1["mean"] is not None:
                diff = w1["mean"] - float(np.mean(prev))
                trend = "rising" if diff > 1 else "falling" if diff < -1 else "steady"
            active = sorted(((t.id, ts - t.first) for t in self.tracks.values() if t.counted and ts - t.last <= 1.0),
                            key=lambda x: -x[1])
            fd = list(self.finished_dwell)
            occ = self.occupancy
            level = "empty" if occ == 0 else "low" if occ <= 5 else "moderate" if occ <= 15 else "high" if occ <= 30 \
                else "very high"
            spd = []
            for t in self.tracks.values():
                if len(t.hist) >= 2 and ts - t.last <= 1.0:
                    (t0, x0, y0), (t1, x1, y1) = t.hist[max(0, len(t.hist) - 1 - int(self.fps / 2))], t.hist[-1]
                    if t1 > t0:
                        spd.append(math.hypot(x1 - x0, y1 - y0) / (t1 - t0))
            return {
                "t": round(ts, 2), "frames": self.frames, "occupancy": occ, "crowd_level": level,
                "unique_people": self.unique, "crossings": dict(self.cross),
                "crossing_directions": dict(self.direction_labels),
                "crossings_last_min": self.metric("crossings_last_min"),
                "occupancy_last_1min": w1, "occupancy_last_5min": w5, "trend": trend,
                "active_tracks": len(active), "longest_present": [{"track": i, "dwell_s": round(d, 1)} for i, d in active[:5]],
                "dwell_finished_median_s": round(float(np.median(fd)), 2) if fd else None,
                "dwell_finished_p90_s": round(float(np.percentile(fd, 90)), 2) if fd else None,
                "mean_speed_px_s": round(float(np.mean(spd)), 1) if spd else None,
                "zones": {z: self.metric(f"zone:{z}") for z in self.zones},
                "loitering": self.metric("loitering"),
                "line": self.line_norm, "rules": self.rules_public(),
                "recent_events": [{k: v for k, v in e.items() if k != "wall"}
                                  for e in (list(self.events)[-recent_events:] if recent_events > 0 else [])],
            }

    def series_list(self, seconds=None):
        with self.lock:
            rows = list(self.series)
        if seconds:
            rows = [r for r in rows if self.ts - r[0] <= seconds]
        return [{"t": r[0], "occ_mean": round(r[1], 2), "occ_max": r[2], "unique": r[3], "crossings": r[4]} for r in rows]

    def heatmap(self):
        with self.lock:
            return self.heat.copy()
