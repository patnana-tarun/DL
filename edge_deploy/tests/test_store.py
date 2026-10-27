import json
import threading
import time

import pytest

from edgevision.store import Store


@pytest.fixture(params=["memory", "file"])
def store(request, tmp_path):
    return Store(":memory:" if request.param == "memory" else tmp_path / "sub" / "ev.db")


def test_file_store_creates_parent_and_persists(tmp_path):
    p = tmp_path / "a" / "b" / "ev.db"
    s = Store(p)
    s.event({"ts": 1.0, "kind": "alert", "rule": "x"})
    s.chat("user", "hi")
    assert p.exists()
    s2 = Store(p)                                              # a new process / restart sees the data
    assert [e["kind"] for e in s2.recent_events()] == ["alert"]
    assert s2.chat_log()[0]["text"] == "hi"


def test_event_round_trip(store):
    e = {"ts": 12.5, "wall": time.time(), "kind": "line_cross", "track": 7, "direction": "a_to_b",
         "a_to_b": 3, "b_to_a": 1, "nested": {"x": [1, 2]}}
    store.event(e)
    store.event({"ts": None, "kind": "config", "change": "set_alert"})              # no wall -> now
    ev = store.recent_events()
    assert len(ev) == 2
    assert ev[0] == e                                                                # oldest first, all fields back
    assert ev[1]["kind"] == "config" and ev[1]["ts"] is None and ev[1]["wall"] == pytest.approx(time.time(), abs=5)


def test_recent_events_filters(store):
    now = time.time()
    store.event({"ts": 0, "wall": now - 3600, "kind": "alert"})                     # an hour ago
    for i, k in enumerate(["alert", "line_cross", "person_enter", "alert", "line_cross"]):
        store.event({"ts": i, "wall": now - 10 + i, "kind": k, "i": i})
    assert len(store.recent_events(minutes=10)) == 5
    assert len(store.recent_events(minutes=120)) == 6
    al = store.recent_events(10, kinds=["alert"])
    assert [e["i"] for e in al] == [0, 3]
    assert [e["kind"] for e in store.recent_events(10, kinds=["alert", "line_cross"])] == \
        ["alert", "line_cross", "alert", "line_cross"]
    assert store.recent_events(10, kinds=["nope"]) == []
    assert [e["i"] for e in store.recent_events(10, limit=2)] == [3, 4]            # newest N, oldest first
    assert store.event_counts(10) == {"alert": 2, "line_cross": 2, "person_enter": 1}
    assert store.event_counts(120)["alert"] == 3


def test_series_and_summary(store):
    assert store.series_summary(10) == {"minutes": 10, "samples": 0}
    rows = [{"t": 0.0, "occ_mean": 1.0, "occ_max": 2, "unique": 3, "crossings": 1},
            {"t": 1.0, "occ_mean": 2.0, "occ_max": 4, "unique": 5, "crossings": 0},
            {"t": 2.0, "occ_mean": 3.5, "occ_max": 5, "unique": 8, "crossings": 2}]
    for r in rows:
        store.series(r)
    store.series({"t": -100.0, "occ_mean": 50.0, "occ_max": 90, "unique": 0, "crossings": 9},
                 wall=time.time() - 7200)                                            # outside the window
    s = store.series_summary(10)
    assert s == {"minutes": 10, "samples": 3, "occupancy_mean": pytest.approx(2.17, abs=0.005),
                 "occupancy_max": 5, "new_unique_people": 5, "crossings": 3}
    assert store.series_summary(180)["samples"] == 4


def test_observations_round_trip(store):
    for i in range(7):
        store.observation(float(i), "periodic" if i % 2 else "alert:crowding", f"report {i}", {"risk": "low"},
                          latency_s=0.5 + i)
    obs = store.observations(limit=3)
    assert [o["text"] for o in obs] == ["report 4", "report 5", "report 6"]
    assert obs[-1] == {"wall": pytest.approx(time.time(), abs=5), "ts": 6.0, "trigger": "alert:crowding",
                       "text": "report 6", "latency_s": 6.5}
    store.observation(None, "user", "no data")
    assert store.observations(1)[0]["latency_s"] is None


def test_chat_round_trip(store):
    store.chat("user", "how many?")
    store.chat("assistant", "3 people", {"actions": [{"tool": "set_alert"}]})
    log = store.chat_log()
    assert [(m["role"], m["text"]) for m in log] == [("user", "how many?"), ("assistant", "3 people")]
    for i in range(60):
        store.chat("user", f"m{i}")
    assert len(store.chat_log()) == 50 and store.chat_log(5)[-1]["text"] == "m59"


def test_unicode_and_quotes(store):
    store.chat("user", "it's \"quoted\" — ünïcode ✓")
    store.event({"ts": 0, "kind": "vlm_risk", "summary": "Robert'); DROP TABLE events;--"})
    assert store.chat_log()[0]["text"] == "it's \"quoted\" — ünïcode ✓"
    assert store.recent_events()[0]["summary"].startswith("Robert')")


def test_thread_safety(store):
    n_threads, n = 8, 60
    errors = []

    def writer(k):
        try:
            for i in range(n):
                store.event({"ts": i, "kind": f"k{k}", "i": i})
                store.series({"t": i, "occ_mean": 1.0, "occ_max": 1, "unique": i, "crossings": 0})
                if i % 10 == 0:
                    store.chat("user", f"{k}-{i}")
                    store.observation(i, "periodic", "x")
                    store.recent_events(1, limit=5)
                    store.series_summary(1)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=writer, args=(k,)) for k in range(n_threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(60)
    assert not errors, errors
    assert sum(store.event_counts(10).values()) == n_threads * n
    assert store.series_summary(10)["samples"] == n_threads * n
    assert len(store.chat_log(1000)) == n_threads * n // 10
    assert all(json.dumps(e) for e in store.recent_events(10, limit=1000))


def test_purge_keeps_recent_rows_and_the_audit_chain():
    import time as _t
    from edgevision.store import Store as _S
    s = _S(":memory:")
    s.event({"kind": "x", "wall": _t.time() - 7200}); s.event({"kind": "y"})
    s.chat("user", "old"); s.audit({"tool": "t"})
    with s._conn() as c:
        c.execute("UPDATE chat SET wall = wall - 7200")
    assert s.purge(1) == 2
    assert [e["kind"] for e in s.recent_events(10 ** 6)] == ["y"] and s.chat_log() == []
    assert s.verify_audit()["ok"] and s.verify_audit()["rows"] == 1
    assert s.purge(0) == 0
