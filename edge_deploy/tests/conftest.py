"""Shared helpers for the EdgeVision tests. Adds edge_deploy/ to sys.path so `import edgevision` works."""
import copy
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

EDGE = Path(__file__).resolve().parent.parent          # edge_deploy/
ROOT = EDGE.parent                                     # project root
if str(EDGE) not in sys.path:
    sys.path.insert(0, str(EDGE))

from edgevision import config  # noqa: E402

MEDIAMTX = EDGE / "tools" / "bin" / "mediamtx"
ONNX_384 = EDGE / "models" / "yolo26n_person_384x640.onnx"
AA1 = ROOT / "videos" / "aa1.mp4"

needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
                                  reason="ffmpeg/ffprobe not on PATH")


def free_port(kind=socket.SOCK_STREAM):
    """A port that is free right now on 127.0.0.1 (TCP by default, UDP with SOCK_DGRAM)."""
    s = socket.socket(socket.AF_INET, kind)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def wait_port(port, timeout=10.0):
    t = time.time() + timeout
    while time.time() < t:
        with socket.socket() as s:
            s.settimeout(0.3)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.1)
    return False


def make_video(path, size="320x240", rate=30, seconds=2, extra=()):
    """Synthetic H.264 test video (ffmpeg testsrc)."""
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "lavfi",
                    "-i", f"testsrc=size={size}:rate={rate}", "-t", str(seconds), "-c:v", "libx264",
                    "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-g", str(rate), *extra, str(path)],
                   check=True, timeout=60)
    return Path(path)


def ffprobe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          "stream=width,height,nb_read_frames,codec_name", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, timeout=60, check=True).stdout.strip()
    codec, w, h, n = out.split(",")[:4]
    return {"codec": codec, "width": int(w), "height": int(h), "frames": int(n)}


def source_cfg(uri, **kw):
    c = copy.deepcopy(config.DEFAULTS["source"])
    c.update(uri=str(uri), width=320, height=240)
    c.update(kw)
    return c


def collect(src, max_frames=None, max_seconds=None, timeout=60.0, per_frame=None):
    """Iterate a VideoSource in a helper thread so a hang fails the test instead of blocking it.

    max_seconds counts from the first frame (a live decoder first probes the stream for a few seconds).
    -> (frames, finished) where finished says whether the iterator ended by itself."""
    frames, done = [], threading.Event()
    state = {"error": None}

    def run():
        t0 = None
        try:
            for fr in src:
                t0 = t0 or time.time()
                frames.append(fr)
                if per_frame:
                    per_frame(fr)
                if (max_frames and len(frames) >= max_frames) or (max_seconds and time.time() - t0 >= max_seconds):
                    break
        except Exception as e:  # noqa: BLE001
            state["error"] = e
        finally:
            done.set()

    th = threading.Thread(target=run, daemon=True)
    th.start()
    finished = done.wait(timeout)
    if state["error"]:
        raise state["error"]
    return frames, finished


@pytest.fixture(scope="session")
def videos(tmp_path_factory):
    """Synthetic videos: 4:3 (320x240), wide (320x120) and tall (120x240), 30 fps, 2 s each."""
    d = tmp_path_factory.mktemp("videos")
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")
    return {"std": make_video(d / "std.mp4"),
            "wide": make_video(d / "wide.mp4", size="320x120"),
            "tall": make_video(d / "tall.mp4", size="120x240")}
