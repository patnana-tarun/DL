import copy
import json

import numpy as np
import pytest

from edgevision.analytics import LOST_S, METRICS, Analytics
from edgevision.config import DEFAULTS

W, H = 1000, 500


def make(fps=10, alerts=(), **over):
    cfg = copy.deepcopy(DEFAULTS["analytics"])
    cfg.update(over)
    events = []
    an = Analytics(cfg, [dict(a) for a in alerts], W, H, fps, on_event=events.append)
    return an, events


def box(fx, fy, w=40, h=100):
    """Box whose foot point (bottom centre) is (fx, fy)."""
    return [fx - w / 2, fy - h, fx + w / 2, fy]


def step(an, ts, people):
    """people: {track_id: (foot_x, foot_y)}"""
    ids = list(people)
    boxes = np.array([box(*people[i]) for i in ids], np.float32).reshape(-1, 4)
    an.update(ts, boxes, np.array(ids, int))


def walk(an, fps, paths, start=0):
    """paths: {tid: [(fx, fy), ...]} (None = absent in that frame). Returns the last ts."""
    n = max(len(p) for p in paths.values())
    for i in range(n):
        step(an, (start + i) / fps, {t: p[i] for t, p in paths.items() if i < len(p) and p[i] is not None})
    return (start + n - 1) / fps


def line_path(x0, x1, fy, step_px=10):
    n = int(abs(x1 - x0) / step_px) + 1
    return [(x0 + np.sign(x1 - x0) * step_px * k, fy) for k in range(n)]


def kinds(events, kind):
    return [e for e in events if e["kind"] == kind]


# ---------------------------------------------------------------- counting line
@pytest.mark.parametrize("line, a_to_b, b_to_a", [
    ([[0.5, 0.0], [0.5, 1.0]], "right to left", "left to right"),          # the default
    ([[0.5, 1.0], [0.5, 0.0]], "left to right", "right to left"),
    ([[0.0, 0.5], [1.0, 0.5]], "top to bottom", "bottom to top"),
    ([[0.2, 0.2], [0.6, 0.8]], "top to bottom", "bottom to top"),          # steep diagonal
    ([[0.1, 0.4], [0.9, 0.6]], "top to bottom", "bottom to top"),          # shallow diagonal
])
def test_direction_labels(line, a_to_b, b_to_a):
    an, _ = make(line=line)
    assert an.direction_labels == {"a_to_b": a_to_b, "b_to_a": b_to_a}


def test_crossings_both_directions_default_line():
    an, ev = make()
    walk(an, 10, {1: line_path(300, 700, 300) + line_path(700, 300, 300)[1:],    # there and back
                  2: line_path(700, 300, 450)})
    assert an.cross == {"a_to_b": 2, "b_to_a": 1}
    crosses = kinds(ev, "line_cross")
    assert [(e["track"], e["direction"]) for e in crosses] == [(1, "b_to_a"), (2, "a_to_b"), (1, "a_to_b")]
    labels = an.snapshot()["crossing_directions"]
    assert labels[crosses[0]["direction"]] == "left to right"                    # track 1 walked left -> right
    assert labels[crosses[1]["direction"]] == "right to left"
    assert an.metric("crossings_total") == 3
    assert an.metric("crossings_a_to_b") == 2 and an.metric("crossings_b_to_a") == 1
    assert an.metric("crossings_last_min") == 3
    assert crosses[-1]["a_to_b"] == 2 and crosses[-1]["b_to_a"] == 1               # running totals in events


def test_hysteresis_ignores_jitter_on_the_line():
    jitter = [(450, 300)] * 5 + [(480, 300), (520, 300)] * 25                     # +-20 px, band is +-30 px
    an, ev = make()
    walk(an, 10, {1: jitter})
    assert an.cross == {"a_to_b": 0, "b_to_a": 0} and not kinds(ev, "line_cross")
    an0, _ = make(line_hysteresis=0.0)                                            # without hysteresis: counted
    walk(an0, 10, {1: jitter})
    assert sum(an0.cross.values()) > 20


def test_wiggle_inside_band_then_cross_counts_once():
    path = line_path(300, 490, 300) + [(510, 300), (490, 300), (515, 300), (485, 300)] * 5 + line_path(500, 700, 300)
    an, _ = make()
    walk(an, 10, {1: path})
    assert an.cross == {"a_to_b": 0, "b_to_a": 1}


def test_partial_vertical_segment():
    an, _ = make(line=[[0.5, 0.0], [0.5, 0.5]])                                 # top half only (y 0-250)
    walk(an, 10, {1: line_path(300, 700, 450),                                   # passes below the segment
                  2: line_path(300, 700, 200)})                                   # passes through it
    assert an.cross == {"a_to_b": 0, "b_to_a": 1}


def test_diagonal_segment_extent():
    an, ev = make(line=[[0.2, 0.2], [0.6, 0.8]])                                 # (200,100)-(600,400) px
    walk(an, 10, {1: line_path(200, 600, 250),                                   # crosses the middle
                  2: line_path(300, 900, 480)})                                   # crosses the extension only
    assert [e["track"] for e in kinds(ev, "line_cross")] == [1]
    assert an.cross == {"a_to_b": 0, "b_to_a": 1}
    assert an.direction_labels["b_to_a"] == "bottom to top"


def test_short_track_crossing_not_counted():
    an, _ = make(fps=10, min_track_s=1.0)                                        # needs 10 frames
    walk(an, 10, {1: line_path(440, 560, 300, step_px=40)})                      # 4 frames
    assert sum(an.cross.values()) == 0 and an.unique == 0


def test_set_line_at_runtime():
    an, _ = make()
    walk(an, 10, {1: line_path(300, 450, 300)})
    an.set_line([[0.0, 0.5], [1.0, 0.5]])                                        # now horizontal
    assert an.line_norm == [[0.0, 0.5], [1.0, 0.5]]
    assert an.direction_labels["a_to_b"] == "top to bottom"
    assert all(t.side == 0 for t in an.tracks.values())                           # sides re-learned
    walk(an, 10, {1: [(450, y) for y in range(200, 420, 10)]}, start=16)          # walks down across y=250
    assert an.cross == {"a_to_b": 1, "b_to_a": 0}


def test_degenerate_line_rejected():
    an, _ = make()
    with pytest.raises(ValueError):
        an.set_line([[0.3, 0.3], [0.3, 0.3]])
    assert an.line_norm == [[0.5, 0.0], [0.5, 1.0]]                               # unchanged


# ---------------------------------------------------------------- people, dwell, zones
def test_unique_people_needs_min_track_duration():
    an, ev = make(fps=10, min_track_s=0.3)                                       # 3 frames
    assert an.min_frames == 3
    walk(an, 10, {1: [(100, 300)] * 2, 2: [(300, 300)] * 3, 3: [(500, 300)] * 20})
    assert an.unique == 2
    assert sorted(e["track"] for e in kinds(ev, "person_enter")) == [2, 3]
    assert an.metric("unique_people") == 2


def test_unique_people_default_at_30fps():
    an, _ = make(fps=30)                                                         # 0.25 s -> 8 frames
    walk(an, 30, {1: [(100, 300)] * 7, 2: [(300, 300)] * 8})
    assert an.unique == 1


def test_person_exit_reports_dwell():
    an, ev = make(fps=10)
    walk(an, 10, {1: [(100, 300)] * 11, 2: [(600, 300)]})                        # 1: ts 0.0-1.0; 2: one frame
    for i in range(11, 40):
        step(an, i / 10, {})
    exits = kinds(ev, "person_exit")
    assert len(exits) == 1 and exits[0]["track"] == 1                           # track 2 was never counted
    assert exits[0]["dwell_s"] == pytest.approx(1.1)                             # last - first + 1 frame
    assert exits[0]["ts"] == pytest.approx(3.1)                                  # unseen > LOST_S (2 s)
    assert LOST_S == 2.0
    assert not an.tracks
    snap = an.snapshot()
    assert snap["dwell_finished_median_s"] == pytest.approx(1.1) and snap["dwell_finished_p90_s"] == pytest.approx(1.1)


def test_zones_enter_exit_and_metric():
    an, ev = make(fps=10, zones={"door": [[0, 0], [0.3, 0], [0.3, 1], [0, 1]]})   # x 0-300 px
    path = [(100 + 20 * i, 300) for i in range(21)]                               # 100 -> 500
    zone_counts = []
    for i, p in enumerate(path):
        step(an, i / 10, {1: p, 2: (50, 400)})                                    # 2 stays inside
        zone_counts.append(an.metric("zone:door"))
    enters, exits = kinds(ev, "zone_enter"), kinds(ev, "zone_exit")
    assert sorted(e["track"] for e in enters) == [1, 2] and all(e["zone"] == "door" for e in enters)
    assert [(e["track"], e["ts"]) for e in exits] == [(1, 1.1)]                    # foot x 320 at frame 11
    assert zone_counts[:11] == [2] * 11 and zone_counts[11:] == [1] * 10
    assert an.snapshot()["zones"] == {"door": 1}
    step(an, 2.3, {1: (500, 300)})                                               # track 2 missing (last seen 2.0)
    assert an.metric("zone:door") == 0


def test_set_zones_runtime_and_unknown_zone_metric():
    an, _ = make()
    an.set_zones({"a": [[0, 0], [1, 0], [1, 1], [0, 1]]})
    step(an, 0.0, {1: (500, 300)})
    assert an.metric("zone:a") == 1 and an.metric("zone:nope") == 0
    with pytest.raises(KeyError):
        an.metric("no_such_metric")


def test_loitering_and_max_dwell():
    an, ev = make(fps=10, loiter_s=1.0)
    walk(an, 10, {1: [(100, 300)] * 16})                                         # 1.5 s
    lo = kinds(ev, "loitering")
    assert len(lo) == 1 and lo[0]["track"] == 1 and lo[0]["ts"] == pytest.approx(1.0)
    assert an.metric("loitering") == 1
    assert an.metric("max_dwell_s") == pytest.approx(1.5)


def test_speed_and_heatmap():
    an, _ = make(fps=10)
    walk(an, 10, {1: line_path(100, 400, 300, step_px=10)})                     # 100 px/s
    snap = an.snapshot()
    assert snap["mean_speed_px_s"] == pytest.approx(100.0, rel=1e-3)
    hm = an.heatmap()
    assert hm.shape == (18, 32) and hm.sum() == 31
    assert hm[int(300 / H * 18)].sum() == 31                                      # all foot points in one row


# ---------------------------------------------------------------- alerts
RULE = {"name": "busy", "metric": "occupancy", "op": ">", "value": 1, "for_s": 1.0, "cooldown_s": 5}


def occupancy_run(an, fps, schedule):
    """schedule: list of (n_frames, occupancy)."""
    i = 0
    for n, occ in schedule:
        for _ in range(n):
            step(an, i / fps, {k: (100 + 100 * k, 300) for k in range(occ)})
            i += 1


def test_alert_for_s_and_cooldown():
    an, ev = make(fps=8, alerts=[RULE])                                          # ts multiples of 1/8: exact
    occupancy_run(an, 8, [(81, 2), (8, 1), (16, 2)])                             # 0-10 s, 10.1-11 s, 11.1-13 s
    alerts = kinds(ev, "alert")
    assert [a["ts"] for a in alerts] == [1.0, 6.0, 12.12]                       # 12.125 rounded to 2 dp
    assert all(a["rule"] == "busy" and a["value"] == 2 and a["threshold"] == 1.0 and a["op"] == ">"
               for a in alerts)
    assert an.rules_public()[0]["active"]


def test_alert_cooldown_blocks_refire():
    an, ev = make(fps=8, alerts=[RULE])
    occupancy_run(an, 8, [(13, 2), (4, 1), (60, 2)])                             # true 0-1.5, false, true 2.125+
    assert [a["ts"] for a in kinds(ev, "alert")] == [1.0, 6.0]                   # 3.125 blocked by the cooldown


def test_alert_condition_must_hold_continuously():
    an, ev = make(fps=8, alerts=[RULE])
    occupancy_run(an, 8, [(7, 2), (1, 0), (7, 2), (1, 0), (7, 2)])               # never 1 s in a row
    assert kinds(ev, "alert") == []
    assert not an.rules_public()[0]["active"]


@pytest.mark.parametrize("op, value, occ, fires", [
    (">=", 2, 2, True), (">=", 3, 2, False), ("<", 1, 0, True), ("<=", 0, 0, True), ("==", 2, 2, True),
    ("==", 2, 1, False), (">", 2, 2, False),
])
def test_alert_operators(op, value, occ, fires):
    an, ev = make(fps=8, alerts=[{"name": "r", "metric": "occupancy", "op": op, "value": value, "for_s": 0}])
    occupancy_run(an, 8, [(3, occ)])
    assert bool(kinds(ev, "alert")) == fires


def test_zone_alert():
    an, ev = make(fps=8, zones={"z": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]]},
                  alerts=[{"name": "zone busy", "metric": "zone:z", "op": ">=", "value": 2, "for_s": 0.5}])
    for i in range(10):
        step(an, i / 8, {1: (100, 300), 2: (200, 300), 3: (800, 300)})
    assert [a["ts"] for a in kinds(ev, "alert")] == [0.5]


def test_default_alert_rules_from_config():
    an, _ = make(alerts=DEFAULTS["alerts"])
    assert [r["name"] for r in an.rules_public()] == ["crowding"]


def test_add_rule_validation_and_replace():
    an, _ = make()
    with pytest.raises(ValueError, match="unknown metric"):
        an.add_rule({"metric": "temperature", "op": ">", "value": 1})
    with pytest.raises(ValueError, match="unknown operator"):
        an.add_rule({"metric": "occupancy", "op": "!=", "value": 1})
    r = an.add_rule({"metric": "occupancy", "op": ">", "value": 3})
    assert r == {"name": "occupancy > 3", "metric": "occupancy", "op": ">", "value": 3.0, "for_s": 0.0,
                 "cooldown_s": 60.0, "active": False}
    an.add_rule({"metric": "zone:door", "op": ">", "value": 0, "name": "door"})
    an.add_rule({"metric": "unique_people", "op": ">=", "value": 10, "name": "door"})   # same name replaces
    names = [r["name"] for r in an.rules_public()]
    assert names == ["occupancy > 3", "door"]
    assert an.rules_public()[-1]["metric"] == "unique_people"
    assert all(not k.startswith("_") for r in an.rules_public() for k in r)
    for m in METRICS:
        if m != "zone:<name>":
            an.add_rule({"metric": m, "op": ">", "value": 1, "name": m})


def test_remove_rule():
    an, _ = make()
    an.add_rule({"metric": "occupancy", "op": ">", "value": 3, "name": "a"})
    an.add_rule({"metric": "occupancy", "op": ">", "value": 4, "name": "b"})
    assert an.remove_rule("a") == 1
    assert an.remove_rule("a") == 0
    assert [r["name"] for r in an.rules_public()] == ["b"]
    assert an.remove_rule() == 1 and an.rules_public() == []


# ---------------------------------------------------------------- read side
SNAP_KEYS = {"t", "frames", "occupancy", "crowd_level", "unique_people", "crossings", "crossing_directions",
             "crossings_last_min", "occupancy_last_1min", "occupancy_last_5min", "trend", "active_tracks",
             "longest_present", "dwell_finished_median_s", "dwell_finished_p90_s", "mean_speed_px_s", "zones",
             "loitering", "line", "rules", "recent_events"}


def test_snapshot_empty():
    an, _ = make()
    snap = an.snapshot()
    assert set(snap) == SNAP_KEYS
    assert snap["occupancy"] == 0 and snap["crowd_level"] == "empty" and snap["recent_events"] == []
    json.dumps(snap, allow_nan=False)


def test_snapshot_keys_and_json():
    an, _ = make(fps=10, zones={"door": [[0, 0], [0.3, 0], [0.3, 1], [0, 1]]},
                 alerts=[{"name": "any", "metric": "occupancy", "op": ">", "value": 0, "for_s": 0}])
    walk(an, 10, {1: line_path(100, 700, 300), 2: line_path(900, 300, 450), 3: [(50, 100)] * 25})
    snap = an.snapshot(recent_events=4)
    assert set(snap) == SNAP_KEYS
    s = json.dumps(snap, allow_nan=False)
    assert json.loads(s)["unique_people"] == 3
    assert len(snap["recent_events"]) == 4 and all("wall" not in e for e in snap["recent_events"])
    assert snap["occupancy"] == 2 and snap["crowd_level"] == "low"
    assert snap["crossings"] == {"a_to_b": 1, "b_to_a": 1}
    assert snap["active_tracks"] == 2 and snap["longest_present"][0]["dwell_s"] == pytest.approx(6.0)
    assert snap["rules"][0]["name"] == "any" and snap["line"] == [[0.5, 0.0], [0.5, 1.0]]
    assert snap["frames"] == 61 and snap["t"] == pytest.approx(6.0)
    assert an.snapshot(recent_events=0)["recent_events"] == []


@pytest.mark.parametrize("occ, level", [(0, "empty"), (5, "low"), (6, "moderate"), (15, "moderate"),
                                        (16, "high"), (30, "high"), (31, "very high")])
def test_crowd_level(occ, level):
    an, _ = make()
    step(an, 0.0, {k: (10 + 30 * k, 300) for k in range(occ)})
    assert an.snapshot()["crowd_level"] == level


def test_series_bins():
    an, _ = make(fps=10)
    for i in range(41):                                                          # ts 0.0 .. 4.0
        s = i // 10
        if s == 0:
            people = {1: (100, 300)}
        elif s == 1:
            people = {1: (100, 300), 2: (300, 300)} if i < 15 else {2: (300, 300)}
        elif s == 2:
            people = {}
        else:
            people = {3: (500, 300), 4: (600, 300), 5: (700, 300)}
        step(an, i / 10, people)
    rows = an.series_list()
    assert [r["t"] for r in rows] == [0.0, 1.0, 2.0, 3.0]                        # bin 4 still open
    assert [r["occ_mean"] for r in rows] == [1.0, 1.5, 0.0, 3.0]
    assert [r["occ_max"] for r in rows] == [1, 2, 0, 3]
    assert [r["unique"] for r in rows] == [1, 2, 2, 5]
    assert [r["crossings"] for r in rows] == [0, 0, 0, 0]
    assert [r["t"] for r in an.series_list(seconds=1.5)] == [3.0]
    snap = an.snapshot()
    assert snap["occupancy_last_1min"] == {"mean": 1.38, "max": 3}


def test_series_crossings_per_bin():
    an, _ = make(fps=10)
    walk(an, 10, {1: line_path(300, 700, 300), 2: [None] * 25 + line_path(700, 300, 450)})
    rows = an.series_list()
    assert sum(r["crossings"] for r in rows) == sum(an.cross.values()) == 2
    assert [r["t"] for r in rows if r["crossings"]] == [2.0, 4.0]


def test_trend_rising():
    an, _ = make(fps=2)
    for i in range(2 * 130):
        ts = i / 2
        occ = 2 if ts < 65 else 10
        step(an, ts, {k: (20 + 40 * k, 300) for k in range(occ)})
    assert an.snapshot()["trend"] == "rising"


def test_history_limit():
    an, _ = make(fps=1, history_s=5, bin_s=1.0)
    for i in range(20):
        step(an, float(i), {})
    assert len(an.series_list()) == 5
