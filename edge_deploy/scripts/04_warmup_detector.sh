#!/usr/bin/env bash
# Step 4: build and cache the TensorRT engine of the person detector, then measure its latency.
#
#   scripts/04_warmup_detector.sh
#   CONFIG=configs/jetson.yaml scripts/04_warmup_detector.sh detector.weights=models/yolo26n_person_736x1280.onnx
#
# The first run on a new board builds a TensorRT FP16 engine from the ONNX model (several minutes on an
# Orin Nano; the screen looks frozen, that is normal). The engine is stored in models/trt_cache/ and every
# later start loads it in seconds. Extra arguments are config overrides (key=value), like --set in edgevision.
#
# The latency printed here is measured on a black frame (no detections, so NMS is almost free): it is the
# preprocess + inference time of YOUR board in its CURRENT power mode. Run it again after changing nvpmodel.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${CONFIG:-configs/jetson.yaml}
RUNS=${RUNS:-50}
PYBIN=${PYTHON:-.venv/bin/python}            # PYTHON=/path/to/python to use another interpreter
[[ -x $PYBIN ]] || PYBIN=python3

echo "==> warming up the detector from $CONFIG (overrides: ${*:-none})"
env -u PYTHONPATH "$PYBIN" - "$CONFIG" "$RUNS" "$@" <<'EOF'
import logging, sys, time
import numpy as np
from edgevision import config
from edgevision.detector import OnnxDetector

logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
cfg_path, runs, overrides = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
cfg = config.load(cfg_path, overrides)
d = cfg["detector"]
if d["backend"] != "onnxruntime":
    sys.exit(f"detector.backend is {d['backend']!r}; this warm-up is for the onnxruntime backend")
W, H = cfg["source"]["width"], cfg["source"]["height"]
print(f"model {d['weights']}  providers wanted {d['providers']}  fp16 {d['fp16']}  trt_cache {config.resolve(d['trt_cache'])}")
print("creating the session (first time on this board: TensorRT engine build, can take several minutes)...", flush=True)
t = time.time()
det = OnnxDetector(d)
t_load = time.time() - t
frame = np.zeros((H, W, 3), np.uint8)
t = time.time()
det(frame)                                  # the TensorRT EP may finish building on the first run
t_first = time.time() - t
for _ in range(5):
    det(frame)
ms = []
for _ in range(runs):
    t = time.perf_counter(); det(frame); ms.append(1000 * (time.perf_counter() - t))
ms = np.array(ms)
print()
print(f"provider      : {det.provider}")
print(f"input (h, w)  : {det.hw}   frame {W}x{H}")
print(f"session load  : {t_load:.1f} s   first inference: {t_first:.1f} s")
print(f"latency       : median {np.median(ms):.1f} ms, p90 {np.percentile(ms, 90):.1f} ms over {runs} runs "
      f"(~{1000 / np.median(ms):.0f} detector-only fps on this board, measured now)")
if det.provider != "TensorrtExecutionProvider":
    print("\nWARNING: not running on TensorRT. Check `onnxruntime.get_available_providers()` and README step 10.")
    sys.exit(2)
EOF
echo
echo "cached engines:"
if [[ -d models/trt_cache ]]; then find models/trt_cache -maxdepth 1 -type f -printf '  %10s bytes  %p\n'
else echo "  (none: TensorRT not used)"; fi
echo
echo "OK. Next: scripts/05_run.sh"
