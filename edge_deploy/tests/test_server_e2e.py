"""End to end: the real Pipeline (ffmpeg decoder -> ONNX detector on CPU -> ByteTrack -> analytics -> store ->
HTTP server) on a 4-second excerpt of videos/aa1.mp4, queried over HTTP while it runs."""
import http.client
import json
import threading
import time
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest
from conftest import AA1, ONNX_384, free_port

from edgevision import config
from edgevision.pipeline import Pipeline

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not (AA1.exists() and ONNX_384.exists()), reason="aa1.mp4 or ONNX model missing")]


def get(port, path, timeout=10):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def post(port, path, obj, timeout=30):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def read_mjpeg_frame(port, timeout=10):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        c.request("GET", "/video.mjpg")
        r = c.getresponse()
        assert r.status == 200 and r.getheader("Content-Type").startswith("multipart/x-mixed-replace")
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += r.read1(4096) if hasattr(r, "read1") else r.read(1)
        head, rest = buf.split(b"\r\n\r\n", 1)
        assert head.startswith(b"--frame")
        n = int([ln for ln in head.split(b"\r\n") if ln.lower().startswith(b"content-length")][0].split(b":")[1])
        while len(rest) < n:
            rest += r.read(n - len(rest))
        return rest[:n]
    finally:
        c.close()


class _Slow:
    """Detector wrapper adding 20 ms per frame; every attribute (provider, cfg, ...) is the real detector's."""

    def __init__(self, det):
        self._det = det

    def __getattr__(self, name):
        return getattr(self._det, name)

    def __call__(self, img):
        out = self._det(img)
        time.sleep(0.02)
        return out


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    port = free_port()
    db = tmp_path_factory.mktemp("e2e") / "edgevision.db"
    cfg = config.load(config.BASE / "configs" / "desktop.yaml", [
        f"source.uri={AA1}", "source.duration_s=4", "detector.weights=models/yolo26n_person_384x640.onnx",
        "detector.providers=[cpu]", "vlm.enabled=false", f"server.port={port}", "server.host=127.0.0.1",
        f"store.path={db}", "log_level=WARNING"])
    p = Pipeline(cfg)
    # keep the run alive long enough for the HTTP checks even on a fast machine (~30 ms/frame minimum)
    p.detector = _Slow(p.detector)
    errors = []

    def target():
        try:
            p.run()
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    th = threading.Thread(target=target, daemon=True)
    th.start()
    t = time.time()
    while p._jpeg is None and time.time() - t < 60:          # first annotated frame published
        time.sleep(0.05)
    assert p._jpeg is not None, "no frame published within 60 s"
    return {"p": p, "port": port, "thread": th, "errors": errors, "db": db, "cfg": cfg}


@pytest.fixture(scope="module")
def live_results(run):
    """All HTTP interactions happen while the pipeline runs; tests below check the recorded results."""
    port, p = run["port"], run["p"]
    res = {}
    res["index"] = get(port, "/")
    res["state1"] = json.loads(get(port, "/api/state")[2])
    res["snapshot"] = get(port, "/snapshot.jpg")
    res["mjpeg"] = read_mjpeg_frame(port)
    res["stats"] = post(port, "/api/chat", {"message": "/stats"})
    res["alert"] = post(port, "/api/chat", {"message": "/alert occupancy > 1"})
    res["state2"] = json.loads(get(port, "/api/state")[2])
    res["alert0"] = post(port, "/api/chat", {"message": "/alert occupancy >= 1 for 0"})   # fires right away
    res["vlm"] = post(port, "/api/chat", {"message": "what is happening?"})
    res["empty"] = post(port, "/api/chat", {"message": "   "})
    res["heatmap"] = get(port, "/heatmap.jpg")
    p.cfg["server"]["token"] = "s3cret"                    # token auth switched on at runtime
    try:
        get(port, "/api/state"); res["no_token"] = 200
    except urllib.error.HTTPError as e:
        res["no_token"] = e.code
    res["with_token"] = get(port, "/api/state?token=s3cret")[0]
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/chat", data=b'{"message": "/alerts"}',
                                 headers={"Content-Type": "application/json", "X-Token": "s3cret"})
    with urllib.request.urlopen(req, timeout=10) as r:
        res["header_token"] = r.status
    p.cfg["server"]["token"] = ""
    t = time.time()
    while p.analytics.ts < 2.2 and p.running and time.time() - t < 60:
        time.sleep(0.05)
    res["series"] = json.loads(get(port, "/api/series?seconds=600")[2])
    res["events"] = json.loads(get(port, "/api/events?minutes=10")[2])
    res["chat_log"] = json.loads(get(port, "/api/chat")[2])
    try:
        get(port, "/nope")
    except urllib.error.HTTPError as e:
        res["404"] = e.code
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/chat", data=b"{bad json",
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10)
    except urllib.error.HTTPError as e:
        res["400"] = e.code
    res["still_running"] = p.running
    run["thread"].join(180)
    res["finished"] = not run["thread"].is_alive()
    return res


def test_ran_while_queried(live_results):
    assert live_results["still_running"], "the pipeline finished before the HTTP checks were done"


def test_index_html(live_results):
    status, ctype, body = live_results["index"]
    assert status == 200 and ctype.startswith("text/html") and b"<html" in body.lower()


def test_state(live_results, run):
    s = live_results["state1"]
    assert set(s) == {"measurements", "pipeline", "vlm", "observations"}
    assert s["vlm"] is None and s["observations"] == []
    assert s["pipeline"]["frames"] >= 1 and s["pipeline"]["live"] is False and s["pipeline"]["dropped"] == 0
    assert s["pipeline"]["detector"] == "CPUExecutionProvider" and s["pipeline"]["source"] == str(AA1)
    assert "detect_ms" in s["pipeline"]["latency_ms"]
    m = s["measurements"]
    assert m["occupancy"] >= 0 and m["rules"][0]["name"] == "crowding"


def test_snapshot_is_valid_jpeg(live_results):
    status, ctype, body = live_results["snapshot"]
    assert status == 200 and ctype == "image/jpeg" and body[:2] == b"\xff\xd8" and body[-2:] == b"\xff\xd9"
    img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (540, 960, 3)                               # 1280x720 scaled to stream_max_width


def test_mjpeg_frame(live_results):
    jpg = live_results["mjpeg"]
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.shape == (540, 960, 3)


def test_chat_commands(live_results):
    st = json.loads(live_results["stats"]["answer"])
    assert "occupancy" in st and "unique_people" in st
    assert live_results["alert"]["answer"] == "alert added: occupancy > 1"
    names = [r["name"] for r in live_results["state2"]["measurements"]["rules"]]
    assert names == ["crowding", "occupancy > 1"]


def test_chat_vlm_disabled(live_results):
    assert "VLM is disabled" in live_results["vlm"]["answer"]
    assert live_results["empty"]["answer"] == "empty message"


def test_chat_log_and_errors(live_results):
    log = live_results["chat_log"]
    assert [m["text"] for m in log if m["role"] == "user"] == ["/stats", "/alert occupancy > 1",
                                                                "/alert occupancy >= 1 for 0", "/alerts"]
    assert live_results["404"] == 404 and live_results["400"] == 400


def test_series_and_events(live_results, run):
    rows = live_results["series"]
    assert len(rows) >= 2 and rows[0]["t"] == 0.0
    assert set(rows[0]) == {"t", "occ_mean", "occ_max", "unique", "crossings"}
    kinds = {e["kind"] for e in live_results["events"]}
    assert "config" in kinds and "person_enter" in kinds                # people are walking in aa1


def test_pipeline_finishes_with_report(live_results, run):
    assert live_results["finished"], "pipeline did not finish"
    assert not run["errors"], run["errors"]
    p = run["p"]
    rep = p.report()
    fps = p.source.fps
    assert fps == pytest.approx(60000 / 1001, rel=1e-3)
    assert rep["frames"] > 0 and abs(rep["frames"] - 4 * fps) <= 2
    for stage in ("detect_ms", "track_ms", "analytics_ms", "annotate_ms", "loop_ms"):
        assert rep["latency_ms"][stage]["n"] == rep["frames"] and rep["latency_ms"][stage]["p50"] > 0
    assert rep["source"] == {"uri": str(AA1), "kind": "file", "live": False, "fps": fps, "arrival_fps": None,
                             "dropped": 0}
    assert rep["detector"]["provider"] == "CPUExecutionProvider"
    assert rep["analytics"]["t"] == pytest.approx((rep["frames"] - 1) / fps, abs=0.02)
    assert rep["analytics"]["unique_people"] > 0
    assert rep["throughput_fps"] > 0 and rep["wall_s"] > 0
    json.dumps(rep)
    # the alert added over HTTP was evaluated by the running pipeline and persisted
    alerts = p.store.recent_events(10, kinds=["alert"])
    assert [a["rule"] for a in alerts if a["rule"] == "occupancy >= 1"] == ["occupancy >= 1"]   # cooldown 60 s
    assert p.store.series_summary(10)["samples"] >= 3
    assert run["db"].exists()


def test_server_stopped_after_run(live_results, run):
    with pytest.raises(OSError):
        get(run["port"], "/api/state", timeout=2)


def test_heatmap_endpoint(live_results):
    code, ctype, body = live_results["heatmap"]
    img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    assert code == 200 and ctype == "image/jpeg" and img is not None and img.shape[1] > 0


def test_token_auth(live_results):
    assert live_results["no_token"] == 401
    assert live_results["with_token"] == 200 and live_results["header_token"] == 200
