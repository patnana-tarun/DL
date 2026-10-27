"""Sustained on-device benchmark: detector-only vs VLM-only vs both at once (review item 2).

Run it ON THE JETSON (or a desktop, for comparison) with the VLM server up (scripts/start_vlm.sh):

    python tools/bench.py --config configs/jetson.yaml --video ../videos/aa1.mp4 --duration 1800
    python tools/bench.py --config configs/desktop.yaml --video ../videos/aa1.mp4 --duration 600 \
        --set detector.backend=ultralytics --set detector.weights=../runs/yolo26n/weights/best.pt

Protocol (MLPerf-edge style; every mode in the same power mode, one after the other):
  det     the deployed pipeline (decoder -> detector -> ByteTrack -> analytics) on the video played as a
          LIVE camera at its native rate (looped; stale frames are dropped as on a real stream)
  vlm     scene-report requests (frame + measurements, fixed max_tokens) every --vlm-interval s, nothing else
  both    det + vlm at the same time: the contention the deployment actually sees
Warm-up (--warmup s) is discarded. Per mode:
  frame latency   capture -> analytics done, p50 / p95 / p99 / max (ms); stage medians
  deadline        share of frames slower than the source frame interval; drop rate = dropped / delivered
  throughput      analysed fps
  VLM             request latency p50 / p95, image+prompt and decode times / speeds from llama.cpp timings
  system          sampled every --sample-ms: power (Jetson: VDD_IN from tegrastats; desktop: nvidia-smi
                  board power), GPU utilisation / clock, temperatures, RAM (+ largest free block on Jetson)
  energy          J per analysed frame = mean power x window / frames (whole module on Jetson, GPU only on desktop)
  drift           first vs last 5 minutes of frame latency and power (thermal throttling), if --duration >= 900
Records the platform (device-tree model, L4T release, nvpmodel mode, jetson_clocks), package versions and the
exact config. -> --out JSON (default results/bench_<host>_<time>.json) + raw samples (CSV next to it).
"""
import argparse
import csv
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from edgevision import config as C  # noqa: E402
from edgevision.pipeline import Pipeline  # noqa: E402
from edgevision.vlm import OBSERVE, SYSTEM, VLMClient, encode_image  # noqa: E402


# ---------------------------------------------------------------- system sampling
class Sampler(threading.Thread):
    """tegrastats on a Jetson, nvidia-smi + /proc/meminfo elsewhere. Rows: dict(t, power_w, gpu_util, ...)."""

    def __init__(self, period_ms):
        super().__init__(daemon=True)
        self.period, self.rows, self._halt = period_ms, [], threading.Event()
        self.jetson = shutil.which("tegrastats") is not None
        self.proc = None

    def run(self):
        if self.jetson:
            self.proc = subprocess.Popen(["tegrastats", "--interval", str(self.period)], stdout=subprocess.PIPE,
                                         text=True)
            for line in self.proc.stdout:
                if self._halt.is_set():
                    break
                self.rows.append({"t": time.time(), **parse_tegrastats(line)})
            return
        while not self._halt.is_set():
            r = {"t": time.time(), **mem_info()}
            try:
                q = subprocess.run(["nvidia-smi", "--query-gpu=power.draw,utilization.gpu,clocks.sm,temperature.gpu,"
                                    "memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                   timeout=5).stdout.split(",")
                r.update(power_w=float(q[0]), gpu_util=float(q[1]), gpu_clock_mhz=float(q[2]),
                         temp_gpu_c=float(q[3]), gpu_mem_mb=float(q[4]))
            except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
                pass
            self.rows.append(r)
            self._halt.wait(self.period / 1000)

    def stop(self):
        self._halt.set()
        if self.proc:
            self.proc.terminate()


def parse_tegrastats(line):
    """One tegrastats line -> numbers (fields differ a little between JetPack releases; missing ones are skipped)."""
    r = {}
    m = re.search(r"RAM (\d+)/(\d+)MB \(lfb (\d+)x(\d+)MB\)", line)
    if m:
        r.update(ram_used_mb=int(m[1]), ram_total_mb=int(m[2]), largest_free_block_mb=int(m[3]) * int(m[4]))
    m = re.search(r"SWAP (\d+)/(\d+)MB", line)
    if m:
        r["swap_used_mb"] = int(m[1])
    m = re.search(r"GR3D_FREQ (\d+)%", line)
    if m:
        r["gpu_util"] = int(m[1])
    m = re.search(r"CPU \[([^\]]+)\]", line)
    if m:
        loads = [int(x.split("%")[0]) for x in m[1].split(",") if "%" in x]
        r["cpu_util_mean"] = float(np.mean(loads)) if loads else None
    for name, val in re.findall(r"(\w+)@(-?[\d.]+)C", line):
        r[f"temp_{name.lower()}_c"] = float(val)
    for name, cur, avg in re.findall(r"(VDD_\w+) (\d+)mW/(\d+)mW", line):
        r[f"{name.lower()}_mw"] = int(cur)
    if "vdd_in_mw" in r:
        r["power_w"] = r["vdd_in_mw"] / 1000
    return r


def mem_info():
    try:
        d = dict(re.findall(r"(\w+):\s+(\d+) kB", Path("/proc/meminfo").read_text()))
        return {"ram_used_mb": (int(d["MemTotal"]) - int(d["MemAvailable"])) // 1024,
                "ram_total_mb": int(d["MemTotal"]) // 1024}
    except (OSError, KeyError):
        return {}


def platform_info():
    def sh(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10, shell=True).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            return ""
    info = {"host": socket.gethostname(), "machine": platform.machine(), "python": platform.python_version(),
            "kernel": platform.release()}
    if Path("/proc/device-tree/model").exists():
        info["device"] = Path("/proc/device-tree/model").read_text().strip("\x00 \n")
    if Path("/etc/nv_tegra_release").exists():
        info["l4t"] = Path("/etc/nv_tegra_release").read_text().splitlines()[0]
    info["nvpmodel"] = sh("nvpmodel -q 2>/dev/null")
    info["jetson_clocks"] = sh("jetson_clocks --show 2>/dev/null | head -5")
    info["gpu"] = sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null")
    for mod in ("onnxruntime", "tensorrt", "torch", "ultralytics", "cv2"):
        try:
            info[f"v_{mod}"] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            pass
    return info


# ---------------------------------------------------------------- load generators
class VLMLoad(threading.Thread):
    def __init__(self, cfg, frame_fn, context_fn, interval_s, max_tokens):
        super().__init__(daemon=True)
        self.client = VLMClient({**cfg, "timeout_s": 300})
        self.frame_fn, self.context_fn, self.interval, self.max_tokens = frame_fn, context_fn, interval_s, max_tokens
        self.image_side = cfg["image_max_side"]
        self.rows, self._halt = [], threading.Event()

    def run(self):
        while not self._halt.is_set():
            t0 = time.time()
            img = self.frame_fn()
            if img is not None:
                msgs = [{"role": "system", "content": SYSTEM + "\n\n" + self.context_fn()},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": encode_image(img, self.image_side)}},
                            {"type": "text", "text": OBSERVE}]}]
                try:
                    _, info = self.client.chat(msgs, max_tokens=self.max_tokens, temperature=0.0)
                    tm = info.get("timings") or {}
                    self.rows.append({"t": t0, "latency_s": info["latency_s"], "prompt_tokens": info["prompt_tokens"],
                                      "completion_tokens": info["completion_tokens"],
                                      "prompt_ms": tm.get("prompt_ms"), "predicted_ms": tm.get("predicted_ms"),
                                      "prompt_tok_s": tm.get("prompt_per_second"),
                                      "decode_tok_s": tm.get("predicted_per_second")})
                except Exception as e:  # noqa: BLE001
                    self.rows.append({"t": t0, "error": f"{type(e).__name__}: {e}"})
            self._halt.wait(max(0.0, self.interval - (time.time() - t0)))

    def stop(self):
        self._halt.set()


def q(v, p):
    return round(float(np.percentile(v, p)), 2) if len(v) else None


def run_mode(mode, cfg, a):
    cfg = json.loads(json.dumps(cfg))
    cfg["vlm"]["enabled"] = False                         # load is generated here, not by the pipeline
    cfg["server"]["enabled"] = False
    cfg["store"]["path"] = str(Path(a.out).with_suffix("")) + f"_{mode}.db"
    sampler = Sampler(a.sample_ms); sampler.start()
    frames, stop_at = [], time.time() + a.warmup + a.duration
    p = vload = None
    holder = {}
    if mode in ("det", "both"):
        p = Pipeline(cfg)
        real = p.source

        def frames_until_stop():                           # remember each frame's capture time; stop on time
            for fr in real:
                if time.time() > stop_at or not p.running:
                    return
                holder["wall"] = fr.wall
                yield fr
        p.source = _Proxy(real, frames_until_stop)
        orig_add = p.timer.add

        def add(stage, ms):
            orig_add(stage, ms)
            if stage == "loop_ms":
                now = time.time()
                frames.append((now, 1000 * (now - holder["wall"]), ms, p.timer.all["detect_ms"][-1]))
        p.timer.add = add
        p.need_annotation = True          # same per-frame work in every mode (the VLM needs the latest frame)
        th = threading.Thread(target=p.run, daemon=True); th.start()
    if mode in ("vlm", "both"):
        import cv2
        base = cv2.imread(a.vlm_frame) if a.vlm_frame else None
        frame_fn = (lambda: p.latest_frames()[0]) if p else (lambda: base)
        ctx = (lambda: json.dumps(p.analytics.snapshot(recent_events=5), default=str)) if p else (lambda: "{}")
        vload = VLMLoad(cfg["vlm"], frame_fn, ctx, a.vlm_interval, a.vlm_max_tokens); vload.start()
    while time.time() < stop_at:
        time.sleep(1)
    if vload:
        vload.stop(); vload.join(timeout=320)
    if p:
        p.running = False; th.join(timeout=30); real.stop()
    sampler.stop()
    t_meas = stop_at - a.duration
    out = {"mode": mode}
    if p:
        fr = np.array([f for f in frames if f[0] >= t_meas]).reshape(-1, 4)
        period_ms = 1000 / max(p.source.fps or 30, 1)
        out["frames"] = {"analysed": int(len(fr)), "fps": round(len(fr) / a.duration, 2),
                         "dropped_total": int(p.source.dropped),
                         "drop_rate": round(p.source.dropped / max(p.source.dropped + p.frames, 1), 4),
                         "e2e_ms": {k: q(fr[:, 1], v) for k, v in (("p50", 50), ("p95", 95), ("p99", 99), ("max", 100))},
                         "loop_ms": {k: q(fr[:, 2], v) for k, v in (("p50", 50), ("p95", 95), ("p99", 99))},
                         "detect_ms": {k: q(fr[:, 3], v) for k, v in (("p50", 50), ("p95", 95), ("p99", 99))},
                         "deadline_ms": round(period_ms, 2),
                         "deadline_miss_rate": round(float((fr[:, 2] > period_ms).mean()), 4) if len(fr) else None,
                         "stages_p50": {k: v["p50"] for k, v in p.timer.summary(full=True).items()}}
        if a.duration >= 900 and len(fr):
            first, last = fr[fr[:, 0] < t_meas + 300], fr[fr[:, 0] > stop_at - 300]
            out["drift"] = {"e2e_p50_first5min": q(first[:, 1], 50), "e2e_p50_last5min": q(last[:, 1], 50)}
    if vload:
        ok = [r for r in vload.rows if "error" not in r and r["t"] >= t_meas]
        out["vlm"] = {"requests": len(ok), "errors": sum("error" in r for r in vload.rows),
                      "latency_s": {"p50": q([r["latency_s"] for r in ok], 50), "p95": q([r["latency_s"] for r in ok], 95)},
                      "prompt_ms_p50": q([r["prompt_ms"] for r in ok if r["prompt_ms"]], 50),
                      "decode_tok_s_p50": q([r["decode_tok_s"] for r in ok if r["decode_tok_s"]], 50),
                      "completion_tokens_p50": q([r["completion_tokens"] for r in ok if r["completion_tokens"]], 50),
                      "max_tokens": a.vlm_max_tokens, "interval_s": a.vlm_interval}
    rows = [r for r in sampler.rows if r["t"] >= t_meas]
    sysm = {}
    for k in sorted({k for r in rows for k in r} - {"t"}):
        v = [r[k] for r in rows if r.get(k) is not None]
        if v:
            sysm[k] = {"mean": round(float(np.mean(v)), 2), "max": round(float(np.max(v)), 2)}
    out["system"] = sysm
    if "power_w" in sysm and p and out["frames"]["analysed"]:
        out["energy_j_per_frame"] = round(sysm["power_w"]["mean"] * a.duration / out["frames"]["analysed"], 4)
    if a.duration >= 900 and rows:
        pw = lambda sel: round(float(np.mean([r["power_w"] for r in sel if "power_w" in r] or [np.nan])), 2)  # noqa: E731
        out.setdefault("drift", {}).update(power_w_first5min=pw([r for r in rows if r["t"] < t_meas + 300]),
                                           power_w_last5min=pw([r for r in rows if r["t"] > stop_at - 300]))
    with open(Path(a.out).with_suffix("").as_posix() + f"_{mode}_samples.csv", "w", newline="") as f:
        keys = sorted({k for r in sampler.rows for k in r})
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(sampler.rows)
    return out


class _Proxy:
    """Looks like the VideoSource to Pipeline.run (attributes pass through), iterates the wrapped generator."""

    def __init__(self, real, gen_fn):
        object.__setattr__(self, "_real", real); object.__setattr__(self, "_gen", gen_fn)

    def __iter__(self):
        return self._gen()

    def __getattr__(self, k):
        return getattr(self._real, k)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--video", required=True, help="played as a live camera at its native rate, looped")
    ap.add_argument("--set", action="append", default=[], help="config override key=value")
    ap.add_argument("--modes", default="det,vlm,both")
    ap.add_argument("--duration", type=float, default=600, help="measured seconds per mode (MLPerf: >= 600)")
    ap.add_argument("--warmup", type=float, default=30)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--vlm-interval", type=float, default=10.0)
    ap.add_argument("--vlm-max-tokens", type=int, default=128)
    ap.add_argument("--vlm-frame", default="", help="image for vlm-only mode (default: first video frame)")
    ap.add_argument("--sample-ms", type=int, default=500)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    a.out = a.out or str(C.BASE / "results" / f"bench_{socket.gethostname()}_{stamp}.json")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    over = [f"source.uri={os.path.abspath(a.video)}", "source.realtime=true", "source.loop=true", *a.set]
    cfg = C.load(a.config, over)
    if not a.vlm_frame:
        import cv2
        cap = cv2.VideoCapture(a.video); ok, img = cap.read(); cap.release()
        a.vlm_frame = str(Path(a.out).with_suffix("")) + "_vlm_frame.jpg"; cv2.imwrite(a.vlm_frame, img)
    res = {"platform": platform_info(), "args": vars(a), "config": cfg, "runs": []}
    for r in range(a.repeats):
        for mode in a.modes.split(","):
            print(f"== repeat {r + 1}/{a.repeats} mode {mode}: {a.warmup:.0f} s warm-up + {a.duration:.0f} s", flush=True)
            res["runs"].append({"repeat": r, **run_mode(mode, cfg, a)})
            print(json.dumps(res["runs"][-1], indent=1), flush=True)
            json.dump(res, open(a.out, "w"), indent=1, default=str)
    print("->", a.out)


if __name__ == "__main__":
    main()
