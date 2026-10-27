"""Closed-loop demonstration of the full edge system on a live stream, recorded for the paper.

A video is published as a live RTSP camera (MediaMTX + ffmpeg -re). EdgeVision runs with the web server and
the Qwen3-VL agent (llama-server must be running: edge_deploy/scripts/start_vlm.sh). A scripted operator then
talks to the system only through the HTTP API, exactly as the UI does:
  1. a scene question                                  -> VLM answer grounded in the live measurements
  2. "alert me if more than N people ..."              -> VLM tool call set_alert -> rule active in analytics
  3. wait for the rule to fire                         -> automatic VLM scene report for the alert (alert_context)
  4. a history question                                -> get_history tool call
  5. "make the counting line horizontal ..."           -> set_counting_line tool call -> crossings re-defined
  6. /stats, then a periodic scene report is requested -> observation stored
Every request, answer, tool call, event and latency is saved.

    python research/edge_demo.py [--video videos/aa1.mp4] [--seconds 150]
Writes results/edge/demo/{transcript.json, events.json, observations.json, report.json, snapshot.jpg}.
"""
import argparse
import json
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import paths as P

sys.path.insert(0, str(P.DEPLOY))
from edgevision import config as ecfg  # noqa: E402
from edgevision.pipeline import Pipeline  # noqa: E402

OUT = Path("results/edge/demo")


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def api(port, path, body=None, timeout=300):
    url = f"http://127.0.0.1:{port}{path}"
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    return json.loads(data) if path.startswith("/api") else data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=P.VIDEO_AA1)
    ap.add_argument("--seconds", type=float, default=150)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    db = OUT / "demo.db"
    if db.exists():
        db.unlink()                                       # the demo's own scratch database from a previous run
    rtsp, http = free_port(), free_port()
    cfgf = OUT / "mediamtx.yml"
    cfgf.write_text(f"rtspAddress: :{rtsp}\nrtmp: no\nhls: no\nwebrtc: no\nsrt: no\napi: no\nmetrics: no\n"
                    "pprof: no\nplayback: no\npaths:\n  all_others:\n")
    mtx = subprocess.Popen([str(P.DEPLOY / "tools/bin/mediamtx"), str(cfgf)], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
    time.sleep(1.5)
    url = f"rtsp://127.0.0.1:{rtsp}/cam1"
    pub = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-re", "-stream_loop", "-1", "-i", a.video,
                            "-an", "-c:v", "copy", "-f", "rtsp", "-rtsp_transport", "tcp", url])
    time.sleep(2.0)
    cfg = ecfg.load(P.DEPLOY / "configs/desktop.yaml", [
        f"source.uri={url}", "source.fps=15",
        "detector.backend=ultralytics", f"detector.weights={(P.ROOT / P.YOLO_FT).as_posix()}",
        f"server.port={http}", "server.host=127.0.0.1", f"store.path={db.resolve().as_posix()}",
        "vlm.observe_every_s=45", "alerts=[]", "log_level=WARNING"])
    p = Pipeline(cfg)
    th = threading.Thread(target=p.run, daemon=True); th.start()
    t0 = time.time()
    for _ in range(100):
        try:
            if api(http, "/api/state")["pipeline"]["frames"] > 30:
                break
        except OSError:
            pass
        time.sleep(0.3)
    transcript = []

    def say(msg, note=""):
        t = time.time()
        r = api(http, "/api/chat", {"message": msg})
        st = api(http, "/api/state")["measurements"]
        row = {"t_s": round(t - t0, 1), "user": msg, "answer": r.get("answer"), "actions": r.get("actions", []),
               "latency_s": round(time.time() - t, 2), "occupancy_at_answer": st["occupancy"],
               "unique_at_answer": st["unique_people"], "note": note}
        transcript.append(row)
        print(json.dumps(row, default=str)[:600], flush=True)
        return r

    try:
        time.sleep(8)
        say("What is happening in the scene right now? How many people are in view?", "scene question")
        occ = api(http, "/api/state")["measurements"]["occupancy"]
        thr = max(3, occ - 3)          # a threshold the scene will exceed soon, so the alert path is exercised
        say(f"Alert me when more than {thr} people are in view for 2 seconds.", f"expect set_alert(occupancy > {thr})")
        fired = None
        for _ in range(120):
            ev = api(http, "/api/events?minutes=30")
            if any(e["kind"] == "alert_context" for e in ev):
                fired = [e for e in ev if e["kind"] in ("alert", "alert_context")]
                break
            time.sleep(1)
        transcript.append({"t_s": round(time.time() - t0, 1), "system": "alert fired; VLM scene report attached",
                           "events": fired})
        say("How busy has it been over the last minute?", "expect get_history")
        say("Make the counting line a horizontal line across the middle of the image.",
            "expect set_counting_line(0,0.5,1,0.5)")
        time.sleep(10)
        say("/stats", "deterministic command")
        say("/observe", "on-demand VLM scene report")
        say("Remove all alerts.", "expect remove_alert(all)")
        while time.time() - t0 < a.seconds:
            time.sleep(1)
        (OUT / "snapshot.jpg").write_bytes(api(http, "/snapshot.jpg"))
        state = api(http, "/api/state")
    finally:
        p.running = False; p.source.stop(); th.join(timeout=15)
        pub.terminate(); mtx.terminate()
    rep = p.report()
    c = sqlite3.connect(db)
    events = [dict(zip(("wall", "ts", "kind", "data"), r)) for r in c.execute("SELECT wall, ts, kind, data FROM events")]
    obs = [dict(zip(("wall", "ts", "trigger", "text", "latency_s"), r))
           for r in c.execute("SELECT wall, ts, trigger, text, latency_s FROM observations")]
    tools_expected = {"set_alert": 1, "get_history": 1, "set_counting_line": 1, "remove_alert": 1}
    tools_called = {}
    for row in transcript:
        for act in row.get("actions", []) or []:
            tools_called[act["tool"]] = tools_called.get(act["tool"], 0) + 1
    vlm_lat = [r["latency_s"] for r in transcript if "user" in r and not r["user"].startswith("/stats")]
    summary = {"video": a.video, "stream": "RTSP (MediaMTX), real time, analysed at 15 fps",
               "detector": "YOLO26n fine-tuned (Ultralytics GPU backend on the desktop)", "vlm": cfg["vlm"]["model"],
               "duration_s": round(time.time() - t0, 1), "pipeline": rep, "tools_expected": tools_expected,
               "tools_called": tools_called, "alert_fired_and_described": bool(fired),
               "chat_latency_s": {"median": sorted(vlm_lat)[len(vlm_lat) // 2] if vlm_lat else None, "all": vlm_lat},
               "observations": len(obs), "events_by_kind": {}, "final_state": state["measurements"]}
    for e in events:
        summary["events_by_kind"][e["kind"]] = summary["events_by_kind"].get(e["kind"], 0) + 1
    for name, obj in (("transcript", transcript), ("events", events), ("observations", obs), ("report", summary)):
        (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1, default=str))
    print(json.dumps({k: summary[k] for k in ("tools_expected", "tools_called", "alert_fired_and_described",
                                              "chat_latency_s", "observations", "events_by_kind")}, indent=1))


if __name__ == "__main__":
    main()
