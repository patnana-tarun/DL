"""Configuration: built-in defaults, deep-merged with a YAML file and optional `key.path=value` overrides.

Relative paths in the config (weights, database, VLM files) are resolved against the edge_deploy/
folder, so the whole folder can be copied anywhere (e.g. to a Jetson) and still work.
"""
import copy
from pathlib import Path

import yaml

from .policy import DEFAULTS as POLICY_DEFAULTS

BASE = Path(__file__).resolve().parent.parent          # edge_deploy/

DEFAULTS = {
    "source": {
        "uri": "rtsp://127.0.0.1:8554/cam1",  # file path, rtsp://, udp://, tcp://, srt://, http(s)://,
                                               # /dev/videoN (USB camera) or csi://N (Jetson CSI camera)
        "width": 1280, "height": 720,          # frames are scaled (aspect kept, padded) to exactly this size
        "fps": 0,                              # resample to this rate with ffmpeg (0 = keep source rate)
        "stride": 1,                           # or keep every Nth decoded frame (exact frame mapping)
        "realtime": None,                      # None = auto: live sources drop stale frames, files never drop
        "loop": False,                         # files only: restart at the end
        "rtsp_transport": "tcp",
        "reconnect_s": 2.0,                    # live sources: wait this long, then reconnect
        "start_s": 0.0, "duration_s": 0.0,     # files only: analyse a time window (0 = whole file)
        "hwaccel": "",                         # e.g. "cuda" on a desktop with NVDEC-enabled ffmpeg
        "decoder": "ffmpeg",                   # ffmpeg | gstreamer (network streams; hardware decode on Jetson)
        "gst_decoder": "nvv4l2decoder",        # GStreamer H.264 decoder element: nvv4l2decoder (Jetson) | avdec_h264
        "input_args": [],                      # extra ffmpeg input arguments, passed verbatim
    },
    "detector": {
        "backend": "onnxruntime",              # onnxruntime | ultralytics
        "weights": "models/yolo26n_wise_person_736x1280.onnx",
        "providers": ["tensorrt", "cuda", "cpu"],   # onnxruntime: first available wins
        "fp16": True,
        "conf": 0.05,                          # keep low-score boxes: ByteTrack's second stage uses them
        "iou": 0.7,                            # detector NMS IoU
        "max_det": 300,
        "trt_cache": "models/trt_cache",
    },
    "tracker": {                               # tuned on aa1 0-15 s, see research/track.py
        "min_conf": 0.2, "nms_iou": 0.6,
        "track_activation_threshold": 0.7, "high_conf_det_threshold": 0.4,
        "lost_track_buffer": 30, "minimum_iou_threshold": 0.4, "minimum_consecutive_frames": 3,
    },
    "analytics": {
        "line": [[0.5, 0.0], [0.5, 1.0]],      # counting line, normalised image coords (x, y)
        "line_hysteresis": 0.3,                # fraction of box height past the line before a side counts
        "zones": {},                           # name: [[x, y], ...] normalised polygon
        "min_track_s": 0.25,                   # tracks shorter than this are not counted as people
        "bin_s": 1.0,                          # time-series resolution
        "history_s": 3600,                     # in-memory time-series length
        "heatmap_grid": [32, 18],
        "loiter_s": 30.0,                      # dwell longer than this raises a loitering event
    },
    "alerts": [                                # rules evaluated every frame; editable at runtime (UI / VLM)
        {"name": "crowding", "metric": "occupancy", "op": ">", "value": 20, "for_s": 3, "cooldown_s": 60},
    ],
    "vlm": {
        "enabled": True,
        "endpoint": "http://127.0.0.1:8080/v1",     # any OpenAI-compatible server: llama.cpp, Ollama, vLLM
        "model": "qwen3-vl-2b",
        "api_key": "",
        "timeout_s": 180,
        "max_tokens": 384,
        "temperature": 0.2,
        "image_max_side": 896,
        "observe_every_s": 120,                # periodic scene report (0 = off)
        "verify_alerts": True,                 # ask the VLM to look at the frame when an alert fires
        "history_turns": 6,
        "use_tools": True,                     # let the VLM call analytics tools (closed loop)
        "image": "annotated",                  # frame sent to the VLM: annotated (boxes + IDs) | raw
        "verify_mode": "describe",             # on alert: describe (scene report, no verdict; default) |
                                               # blind / informed (yes/no verdict; unreliable for counts, E5)
    },
    "server": {"enabled": True, "host": "0.0.0.0", "port": 8000, "jpeg_quality": 80, "stream_max_width": 960,
               "token": "",                    # set a token to require ?token=... / X-Token header on every request
               "users": [],                    # [{name, token, role: viewer|operator|admin}]; `token` above = admin
               "anonymous_role": "operator"},  # role when no token/users are configured (trusted LAN only)
    "policy": copy.deepcopy(POLICY_DEFAULTS),  # trusted tool-call layer, see edgevision/policy.py
    "output": {"uri": "", "fps": 15, "bitrate": "2M"},   # annotated re-stream / recording (rtsp://, udp://, .mp4)
    "store": {"path": "data/edgevision.db",
              "retention_h": 72},              # delete events / series / reports / chat older than this (0 = keep)
    "log_level": "INFO",
}


def _merge(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def _parse_value(s):
    return yaml.safe_load(s)


def load(path=None, overrides=()):
    """-> config dict. `overrides` are strings like "source.uri=videos/aa1.mp4" or "vlm.enabled=false"."""
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as f:
            _merge(cfg, yaml.safe_load(f) or {})
    for o in overrides:
        key, _, val = o.partition("=")
        if not _:
            raise ValueError(f"override {o!r} is not key=value")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = _parse_value(val)
    validate(cfg)
    return cfg


def resolve(p):
    """Resolve a config path relative to edge_deploy/ (absolute paths and URLs pass through)."""
    if not p or "://" in str(p):
        return p
    q = Path(p).expanduser()
    return str(q if q.is_absolute() else BASE / q)


def validate(cfg):
    s, d = cfg["source"], cfg["detector"]
    if s["width"] <= 0 or s["height"] <= 0:
        raise ValueError("source.width/height must be positive")
    if s["fps"] and s["stride"] != 1:
        raise ValueError("use either source.fps or source.stride, not both")
    if s.get("decoder", "ffmpeg") not in ("ffmpeg", "gstreamer"):
        raise ValueError(f"unknown source.decoder {s['decoder']!r}")
    if cfg["vlm"].get("verify_mode", "describe") not in ("describe", "blind", "informed"):
        raise ValueError("vlm.verify_mode must be describe, blind or informed")
    if d["backend"] not in ("onnxruntime", "ultralytics"):
        raise ValueError(f"unknown detector.backend {d['backend']!r}")
    for r in cfg["alerts"]:
        missing = {"name", "metric", "op", "value"} - set(r)
        if missing:
            raise ValueError(f"alert {r} misses {sorted(missing)}")
    from .policy import ROLES
    srv = cfg["server"]
    for u in srv.get("users") or []:
        if not u.get("token") or u.get("role") not in ROLES:
            raise ValueError(f"server.users entry {u.get('name')!r} needs a token and a role in {sorted(ROLES)}")
    if srv.get("anonymous_role", "operator") not in ROLES:
        raise ValueError("server.anonymous_role must be viewer, operator or admin")
    (x1, y1), (x2, y2) = cfg["analytics"]["line"]
    if (x1, y1) == (x2, y2):
        raise ValueError("analytics.line needs two different points")
