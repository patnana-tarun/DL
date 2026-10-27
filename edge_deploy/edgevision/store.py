"""Persistent store (SQLite): analytics events, time series, VLM observations, the chat log and the
hash-chained audit log of every tool decision (policy.py).

One connection per thread (sqlite3 objects are not shared across threads); WAL mode lets the UI and the
VLM read while the pipeline writes.
"""
import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, wall REAL, ts REAL, kind TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS series (wall REAL, ts REAL, occ_mean REAL, occ_max INTEGER, unique_people INTEGER,
                                   crossings INTEGER);
CREATE TABLE IF NOT EXISTS observations (id INTEGER PRIMARY KEY, wall REAL, ts REAL, trigger TEXT, text TEXT,
                                         data TEXT, latency_s REAL);
CREATE TABLE IF NOT EXISTS chat (id INTEGER PRIMARY KEY, wall REAL, role TEXT, text TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS audit (seq INTEGER PRIMARY KEY, wall REAL, data TEXT, prev TEXT, hash TEXT);
CREATE INDEX IF NOT EXISTS ev_wall ON events(wall);
CREATE INDEX IF NOT EXISTS se_wall ON series(wall);
"""


class Store:
    def __init__(self, path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._mem = sqlite3.connect(":memory:", check_same_thread=False) if self.path == ":memory:" else None
        self._mem_lock = threading.Lock()
        self._audit_lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self):
        if self._mem is not None:
            return _Locked(self._mem, self._mem_lock)
        c = getattr(self._local, "c", None)
        if c is None:
            c = self._local.c = sqlite3.connect(self.path, timeout=10)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
        return c

    # ---------------------------------------------------------------- writes
    def event(self, e):
        with self._conn() as c:
            c.execute("INSERT INTO events(wall, ts, kind, data) VALUES (?,?,?,?)",
                      (e.get("wall", time.time()), e.get("ts"), e["kind"],
                       json.dumps({k: v for k, v in e.items() if k not in ("wall", "ts", "kind")})))

    def series(self, row, wall=None):
        with self._conn() as c:
            c.execute("INSERT INTO series VALUES (?,?,?,?,?,?)",
                      (wall or time.time(), row["t"], row["occ_mean"], row["occ_max"], row["unique"], row["crossings"]))

    def observation(self, ts, trigger, text, data=None, latency_s=None):
        with self._conn() as c:
            c.execute("INSERT INTO observations(wall, ts, trigger, text, data, latency_s) VALUES (?,?,?,?,?,?)",
                      (time.time(), ts, trigger, text, json.dumps(data or {}), latency_s))

    def chat(self, role, text, data=None):
        with self._conn() as c:
            c.execute("INSERT INTO chat(wall, role, text, data) VALUES (?,?,?,?)",
                      (time.time(), role, text, json.dumps(data or {})))

    def purge(self, retention_h):
        """Delete events, series, scene reports and chat older than retention_h hours (privacy: data
        minimisation). The audit log is kept: deleting from it would break the hash chain on purpose."""
        if not retention_h:
            return 0
        cut, n = time.time() - 3600 * retention_h, 0
        with self._conn() as c:
            for t in ("events", "series", "observations", "chat"):
                n += c.execute(f"DELETE FROM {t} WHERE wall < ?", (cut,)).rowcount
        return n

    def audit(self, rec):
        """Append one decision; hash = SHA-256(prev_hash + canonical JSON of (seq, wall, data))."""
        data = json.dumps(rec, sort_keys=True, separators=(",", ":"), default=str)
        with self._audit_lock, self._conn() as c:
            last = c.execute("SELECT seq, hash FROM audit ORDER BY seq DESC LIMIT 1").fetchone()
            seq, prev = (last[0] + 1, last[1]) if last else (1, GENESIS)
            wall = round(time.time(), 6)
            h = _chain(prev, seq, wall, data)
            c.execute("INSERT INTO audit(seq, wall, data, prev, hash) VALUES (?,?,?,?,?)", (seq, wall, data, prev, h))
        return h

    def audit_log(self, limit=100):
        with self._conn() as c:
            rows = c.execute("SELECT seq, wall, data, hash FROM audit ORDER BY seq DESC LIMIT ?", (limit,)).fetchall()
        return [{"seq": s, "wall": w, **json.loads(d), "hash": h} for s, w, d, h in rows][::-1]

    def verify_audit(self):
        """-> {"ok", "rows", "head", "first_bad"}: recomputes the whole chain (edits, deletions, reordering)."""
        with self._conn() as c:
            rows = c.execute("SELECT seq, wall, data, prev, hash FROM audit ORDER BY seq").fetchall()
        prev, expect = GENESIS, 1
        for seq, wall, data, p, h in rows:
            if seq != expect or p != prev or h != _chain(prev, seq, wall, data):
                return {"ok": False, "rows": len(rows), "head": prev, "first_bad": seq}
            prev, expect = h, expect + 1
        return {"ok": True, "rows": len(rows), "head": prev, "first_bad": None}

    # ---------------------------------------------------------------- reads
    def recent_events(self, minutes=10, kinds=None, limit=200):
        q = "SELECT wall, ts, kind, data FROM events WHERE wall >= ?"
        args = [time.time() - 60 * minutes]
        if kinds:
            q += f" AND kind IN ({','.join('?' * len(kinds))})"; args += list(kinds)
        q += " ORDER BY id DESC LIMIT ?"; args.append(limit)
        with self._conn() as c:
            rows = c.execute(q, args).fetchall()
        return [{"wall": w, "ts": t, "kind": k, **json.loads(d)} for w, t, k, d in rows][::-1]

    def event_counts(self, minutes=10):
        with self._conn() as c:
            rows = c.execute("SELECT kind, COUNT(*) FROM events WHERE wall >= ? GROUP BY kind",
                             (time.time() - 60 * minutes,)).fetchall()
        return dict(rows)

    def series_summary(self, minutes=10):
        with self._conn() as c:
            r = c.execute("SELECT COUNT(*), AVG(occ_mean), MAX(occ_max), MIN(unique_people), MAX(unique_people), "
                          "SUM(crossings) FROM series WHERE wall >= ?", (time.time() - 60 * minutes,)).fetchone()
        n, avg, mx, u0, u1, cr = r
        if not n:
            return {"minutes": minutes, "samples": 0}
        return {"minutes": minutes, "samples": n, "occupancy_mean": round(avg, 2), "occupancy_max": mx,
                "new_unique_people": (u1 - u0) if u0 is not None else None, "crossings": cr}

    def observations(self, limit=5):
        with self._conn() as c:
            rows = c.execute("SELECT wall, ts, trigger, text, latency_s FROM observations ORDER BY id DESC LIMIT ?",
                             (limit,)).fetchall()
        return [{"wall": w, "ts": t, "trigger": g, "text": x, "latency_s": l} for w, t, g, x, l in rows][::-1]

    def chat_log(self, limit=50):
        with self._conn() as c:
            rows = c.execute("SELECT wall, role, text FROM chat ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"wall": w, "role": r, "text": t} for w, r, t in rows][::-1]


GENESIS = "0" * 64


def _chain(prev, seq, wall, data):
    return hashlib.sha256(f"{prev}|{seq}|{wall!r}|{data}".encode()).hexdigest()


class _Locked:
    """Context manager giving serialised access to a shared in-memory connection (tests)."""

    def __init__(self, c, lock):
        self.c, self.lock = c, lock

    def __enter__(self):
        self.lock.acquire()
        self.c.__enter__()
        return self.c

    def __exit__(self, *a):
        try:
            return self.c.__exit__(*a)
        finally:
            self.lock.release()
