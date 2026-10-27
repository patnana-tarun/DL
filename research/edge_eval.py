"""End-to-end evaluation of the deployed edge pipeline (edge_deploy/edgevision), not of the research code.

The pipeline is run exactly as on the device (FFmpeg decoder process -> ONNX Runtime detector -> ByteTrack ->
online analytics), on the tracking hold-out of aa1.mp4 (first 30 s, never used for training), and its tracks
are scored against the same teacher pseudo-GT as research/track.py.

  files    file source, 736x1280 ONNX, frame stride 1 / 2 / 4 (59.9 / 30 / 15 fps analysis rate), plus the two
           smaller Jetson inputs (544x960, 384x640) at stride 4. Scored: HOTA / MOTA / IDF1 on the held-out half
           (15-30 s) and on 0-30 s, unique people (>= 0.25 s tracks) vs pseudo-GT, per-stage latency.
           Stride s keeps original frames 0, s, 2s, ... so the pseudo-GT is subsampled at the same frames.
  live     the video is published as a live RTSP camera (tools/stream_videos.py -> MediaMTX, real time) and the
           pipeline reads rtsp:// in live mode (stale frames dropped). Reported: delivered vs dropped frames,
           throughput, latency, analytics. Live frames have no exact frame index, so no HOTA here.

    python research/edge_eval.py files|live|all
Writes results/edge/edge_eval.json (+ per-run reports and track dumps in results/edge/runs/).
"""
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

import paths as P
import track

sys.path.insert(0, str(P.DEPLOY))
from edgevision import config as ecfg  # noqa: E402
from edgevision.pipeline import Pipeline  # noqa: E402

OUT = Path("results/edge"); RUNS = OUT / "runs"
SECONDS = 30.0
BASE = ["vlm.enabled=false", "server.enabled=false", "output.uri=", "log_level=WARNING",
        "detector.providers=[cpu]", f"source.uri={(P.ROOT / P.VIDEO_AA1).as_posix()}", f"source.duration_s={SECONDS}"]
FILE_RUNS = {   # name: (onnx, stride)
    "736x1280_s1": ("models/yolo26n_person_736x1280.onnx", 1),
    "736x1280_s2": ("models/yolo26n_person_736x1280.onnx", 2),
    "736x1280_s4": ("models/yolo26n_person_736x1280.onnx", 4),
    "544x960_s4": ("models/yolo26n_person_544x960.onnx", 4),
    "384x640_s4": ("models/yolo26n_person_384x640.onnx", 4),
}


def run_pipeline(name, overrides):
    cfg = ecfg.load(None, BASE + [f"store.path={(RUNS / name).resolve().as_posix()}.db"] + overrides)
    p = Pipeline(cfg, record_tracks=True)
    p.run()
    rep = p.report()
    (RUNS / f"{name}.json").write_text(json.dumps(rep, indent=1))
    return p, rep


def unique_ids(tracks, min_frames):
    n = {}
    for _, ids in tracks:
        for j in ids:
            n[int(j)] = n.get(int(j), 0) + 1
    return sum(1 for v in n.values() if v >= min_frames)


def files():
    fps, cache = track.load_cache()
    gt_full = track.build_gt(fps, cache["teacher"])
    n_full = len(gt_full)
    res = {}
    for name, (onnx, stride) in FILE_RUNS.items():
        t = time.time()
        p, rep = run_pipeline(name, [f"detector.weights={onnx}", f"source.stride={stride}"])
        pred = [(tb, ids) for _, _, tb, ids, _, _ in p.record]
        gt = gt_full[::stride][:len(pred)]
        n = min(len(pred), len(gt)); half = (n_full // 2 + stride - 1) // stride
        m_hold = track.score(gt, pred, half, n); m_full = track.score(gt, pred, 0, n)
        min_frames = max(1, round(track.GT_MIN_LEN / stride))
        res[name] = {"onnx": onnx, "stride": stride, "analysis_fps": round(fps / stride, 2), "frames": n,
                     "heldout_15-30s": m_hold, "full_0-30s": m_full,
                     "unique_people": unique_ids(pred[:n], min_frames), "gt_unique_people": unique_ids(gt[:n], min_frames),
                     "pipeline_unique_people": rep["analytics"]["unique_people"],
                     "latency_ms": rep["latency_ms"], "provider": rep["detector"]["provider"],
                     "wall_s": round(time.time() - t, 1)}
        print(f"{name}: frames {n}  HOTA held-out {m_hold['HOTA']:.3f} MOTA {m_hold['MOTA']:.3f} IDF1 {m_hold['IDF1']:.3f}  "
              f"unique {res[name]['unique_people']} (GT {res[name]['gt_unique_people']})  "
              f"detect p50 {rep['latency_ms']['detect_ms']['p50']:.1f} ms", flush=True)
        np.savez_compressed(RUNS / f"{name}_tracks.npz", **{f"{i}": np.c_[b, ids] for i, (b, ids) in enumerate(pred)})
    ref = json.load(open("results/tracking/tracking_metrics.json"))["YOLO26n"]["tuned"]["heldout_15-30s"]
    res["_reference_research_offline"] = {"what": "research/track.py, PyTorch FP16 fast path, cv2 decode, stride 1",
                                         "heldout_15-30s": ref}
    return res


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def live():
    """Publish aa1 as a real-time RTSP camera and analyse it live (GPU Ultralytics backend, like a device with a
    GPU; the CPU ONNX path cannot keep up with 60 fps and would only show frame dropping)."""
    port = free_port()
    cfgf = RUNS / "mediamtx_eval.yml"
    cfgf.write_text(f"rtspAddress: :{port}\nrtmp: no\nhls: no\nwebrtc: no\nsrt: no\napi: no\nmetrics: no\n"
                    "pprof: no\nplayback: no\npaths:\n  all_others:\n")
    mtx = subprocess.Popen([str(P.DEPLOY / "tools/bin/mediamtx"), str(cfgf)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.5)
    url = f"rtsp://127.0.0.1:{port}/cam1"
    pub = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-re", "-t", str(SECONDS + 5),
                            "-i", P.VIDEO_AA1, "-an", "-c:v", "copy", "-f", "rtsp", "-rtsp_transport", "tcp", url])
    time.sleep(2.0)
    res = {}
    try:
        n_frames = int(SECONDS * 59.94)
        for name, det in (("rtsp_live_gpu_pt1280", ["detector.backend=ultralytics", f"detector.weights={(P.ROOT / P.YOLO_FT).as_posix()}"]),):
            cfg_over = [o for o in BASE if not o.startswith(("source.uri", "source.duration_s"))]
            cfg = ecfg.load(None, cfg_over + [f"source.uri={url}", f"store.path={(RUNS / name).resolve().as_posix()}.db",
                                              "source.reconnect_s=0.5"] + det)
            p = Pipeline(cfg)
            t0 = time.time()
            th = threading.Thread(target=p.run, daemon=True); th.start()
            th.join(timeout=SECONDS + 4)
            p.running = False; p.source.stop(); th.join(timeout=10)
            rep = p.report(); wall = time.time() - t0
            res[name] = {"url": "rtsp (MediaMTX, ffmpeg -re publisher, -c copy)", "wall_s": round(wall, 1),
                         "frames_analysed": rep["frames"], "frames_dropped_stale": rep["source"]["dropped"],
                         "source_frames_published_approx": n_frames,
                         "throughput_fps": round(rep["frames"] / wall, 2), "latency_ms": rep["latency_ms"],
                         "analytics": rep["analytics"], "detector": rep["detector"]}
            (RUNS / f"{name}.json").write_text(json.dumps(rep, indent=1))
            print(name, json.dumps({k: res[name][k] for k in ("frames_analysed", "frames_dropped_stale", "throughput_fps")}), flush=True)
    finally:
        pub.terminate(); mtx.terminate(); pub.wait(5); mtx.wait(5)
    return res


if __name__ == "__main__":
    RUNS.mkdir(parents=True, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    path = OUT / "edge_eval.json"
    out = json.load(open(path)) if path.exists() else {}
    if mode in ("files", "all"):
        out["files"] = files()
    if mode in ("live", "all"):
        out["live"] = live()
    out["protocol"] = __doc__.split("\n\n")[1].strip()
    json.dump(out, open(path, "w"), indent=1, default=float)
