"""Video in / video out through FFmpeg (or GStreamer for Jetson CSI cameras) subprocess pipes.

Every source is decoded by an external process that writes raw BGR frames of a fixed size to stdout,
so files, network streams and cameras all look the same to the pipeline:

    file.mp4 | rtsp://... | udp://... | tcp://... | srt://... | http(s)://...   -> ffmpeg
    /dev/videoN                                                                -> ffmpeg (v4l2)
    csi://N                                                                    -> gst-launch-1.0 (nvarguscamerasrc)

Live sources keep only the newest frame (the analyser never falls behind the camera; dropped frames are
counted). Files are read without dropping, so an analysis of a file is deterministic.
"""
import logging
import queue
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

log = logging.getLogger("edgevision.stream")
LIVE_SCHEMES = ("rtsp://", "rtsps://", "rtmp://", "udp://", "tcp://", "srt://", "http://", "https://", "rtp://")


@dataclass
class Frame:
    index: int            # 0-based index of the delivered frame
    ts: float             # seconds: media time for files, time since start for live sources
    image: np.ndarray     # HxWx3 uint8 BGR
    wall: float = field(default_factory=time.time)


def kind_of(uri):
    u = str(uri)
    if u.startswith("csi://"):
        return "csi"
    if u.startswith("/dev/video") or u.startswith("v4l2://"):
        return "v4l2"
    if u.startswith(LIVE_SCHEMES):
        return "network"
    return "file"


def probe_fps(uri):
    """Average frame rate of a file via ffprobe (None if unknown)."""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=avg_frame_rate,r_frame_rate", "-of", "csv=p=0", str(uri)],
                             capture_output=True, text=True, timeout=20).stdout.strip().split("\n")[0]
        for r in out.split(",")[::-1]:          # prefer avg_frame_rate (correct for VFR phone video)
            if r and r != "0/0":
                return float(Fraction(r))
    except Exception as e:  # noqa: BLE001
        log.warning("ffprobe failed for %s: %s", uri, e)
    return None


def probe_pts(uri):
    """Presentation timestamps (s) of every video frame of a file, in display order (None if unknown).
    Needed for variable-frame-rate files (phone video): index / mean fps can be off by many seconds there."""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
                              "-of", "csv=p=0", str(uri)], capture_output=True, text=True, timeout=120).stdout
        pts = np.sort(np.array([float(x) for x in out.split() if x and x != "N/A"]))
        return pts if len(pts) > 1 else None
    except Exception as e:  # noqa: BLE001
        log.warning("ffprobe pts failed for %s: %s", uri, e)
    return None


def media_ts(pts, start_s=0.0, duration_s=0.0, stride=1):
    """Timestamps of the frames the decoder delivers, relative to the first one: the frames at or after start_s
    (ffmpeg -ss before -i is frame-accurate), within duration_s, every `stride`-th (select=not(mod(n,stride)))."""
    p = np.asarray(pts, float)
    p = p[p >= p[0] + start_s - 1e-3] if start_s else p
    if duration_s and len(p):
        p = p[p < p[0] + duration_s - 1e-6]
    p = p[::max(1, int(stride))]
    return p - p[0] if len(p) else p


def build_command(cfg):
    """-> argv of the decoder process for a source config."""
    uri, W, H = str(cfg["uri"]), int(cfg["width"]), int(cfg["height"])
    k = kind_of(uri)
    if k == "csi":
        sensor = uri.split("://", 1)[1] or "0"
        fps = int(cfg["fps"] or 30)
        pipe = (f"nvarguscamerasrc sensor-id={sensor} ! video/x-raw(memory:NVMM),width={W},height={H},"
                f"framerate={fps}/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
                f"video/x-raw,format=BGR ! fdsink fd=1 sync=false")
        return ["gst-launch-1.0", "-q", *shlex.split(pipe)]
    if k == "network" and cfg.get("decoder") == "gstreamer":
        return gst_network_command(cfg)

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    if cfg.get("hwaccel"):
        cmd += ["-hwaccel", cfg["hwaccel"]]
    if k == "network":
        cmd += ["-fflags", "nobuffer", "-flags", "low_delay"]
        if uri.startswith(("rtsp://", "rtsps://")):
            cmd += ["-rtsp_transport", cfg.get("rtsp_transport", "tcp")]
    if k == "v4l2":
        dev = uri.replace("v4l2://", "")
        cmd += ["-f", "v4l2", "-video_size", f"{W}x{H}"]
        if cfg["fps"]:
            cmd += ["-framerate", str(cfg["fps"])]
        uri = dev
    if k == "file":
        if cfg.get("loop"):
            cmd += ["-stream_loop", "-1"]
        if cfg.get("realtime"):
            cmd += ["-re"]
        if cfg.get("start_s"):
            cmd += ["-ss", str(cfg["start_s"])]
    cmd += list(cfg.get("input_args") or [])
    cmd += ["-i", uri]
    if k == "file" and cfg.get("duration_s"):
        cmd += ["-t", str(cfg["duration_s"])]
    vf = []
    if cfg.get("stride", 1) > 1:
        vf.append(f"select=not(mod(n\\,{int(cfg['stride'])}))")
    if cfg["fps"] and k != "v4l2":
        vf.append(f"fps={cfg['fps']}")
    vf.append(f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2")
    cmd += ["-an", "-sn", "-dn", "-vf", ",".join(vf)]
    if not cfg["fps"]:      # deliver decoded frames as they are: rawvideo output otherwise defaults to constant frame
        cmd += ["-fps_mode", "passthrough"]   # rate and duplicates / drops frames of variable-frame-rate input
    cmd += ["-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
    return cmd


def gst_network_command(cfg):
    """RTSP / UDP H.264 through GStreamer, so the Jetson's hardware decoder (nvv4l2decoder + nvvidconv) does the
    work instead of the CPU. Elsewhere set source.gst_decoder: avdec_h264 (software)."""
    uri, W, H = str(cfg["uri"]), int(cfg["width"]), int(cfg["height"])
    dec = cfg.get("gst_decoder", "nvv4l2decoder")
    if uri.startswith(("rtsp://", "rtsps://")):
        src = (f"rtspsrc location={uri} latency=100 protocols={cfg.get('rtsp_transport', 'tcp')} drop-on-latency=true "
               f"! rtph264depay ! h264parse")
    elif uri.startswith("udp://"):
        host_port = uri.split("://", 1)[1].split("?")[0]
        src = f"udpsrc port={host_port.rsplit(':', 1)[1]} ! tsdemux ! h264parse"
    else:
        raise ValueError(f"source.decoder=gstreamer supports rtsp:// and udp:// (got {uri})")
    conv = "nvvidconv" if dec.startswith("nv") else "videoconvert ! videoscale"
    fmt = "BGRx" if dec.startswith("nv") else "BGR"
    pipe = (f"{src} ! {dec} ! {conv} ! video/x-raw,format={fmt},width={W},height={H} ! videoconvert ! "
            f"video/x-raw,format=BGR ! fdsink fd=1 sync=false")
    return ["gst-launch-1.0", "-q", *shlex.split(pipe)]


class VideoSource:
    """Iterate decoded frames: `for frame in VideoSource(cfg): ...`. Thread-safe `stop()`."""

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.kind = kind_of(cfg["uri"])
        self.live = self.kind != "file" if cfg.get("realtime") is None else bool(cfg["realtime"])
        self.W, self.H = int(cfg["width"]), int(cfg["height"])
        self.nbytes = self.W * self.H * 3
        self.src_fps = probe_fps(cfg["uri"]) if self.kind == "file" else None
        if cfg["fps"]:
            self.fps = float(cfg["fps"])
        elif self.src_fps:
            self.fps = self.src_fps / max(1, int(cfg.get("stride", 1)))
        else:
            self.fps = 30.0 / max(1, int(cfg.get("stride", 1)))   # live default until arrivals are measured
        self.ts_table = None                # files: real media timestamps of the delivered frames (VFR-safe)
        if self.kind == "file" and not cfg["fps"]:
            pts = probe_pts(cfg["uri"])
            if pts is not None:
                self.ts_table = media_ts(pts, cfg.get("start_s") or 0.0, cfg.get("duration_s") or 0.0,
                                         cfg.get("stride", 1))
        self.dropped = 0
        self.arrival_fps = None             # live sources: measured decode rate (EMA), before stale-frame dropping
        self.delivered = 0
        self._q = queue.Queue(maxsize=1 if self.live else 8)
        self._stop = threading.Event()
        self._proc = None
        self._thread = threading.Thread(target=self._reader, name="source", daemon=True)
        self._t0 = None

    def _file_ts(self, idx):
        t = self.ts_table
        if t is None or not len(t):
            return idx / self.fps
        n = len(t)                           # past the end only when looping: continue the media clock
        return float(t[idx % n] + (idx // n) * (t[-1] + 1 / self.fps))

    # ---------------------------------------------------------------- process management
    def _spawn(self):
        cmd = build_command(self.cfg)
        log.info("decoder: %s", " ".join(shlex.quote(c) for c in cmd))
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=self.nbytes * 2)

    def _read_exact(self, f):
        buf = bytearray(self.nbytes); view = memoryview(buf); got = 0
        while got < self.nbytes:
            n = f.readinto(view[got:])
            if not n:
                return None
            got += n
        return buf

    def _reader(self):
        idx = failures = 0
        while not self._stop.is_set():
            p = self._proc = self._spawn()          # local ref: stop() may reset self._proc at any time
            got_frames = False
            while not self._stop.is_set():
                buf = self._read_exact(p.stdout)
                if buf is None:
                    break
                got_frames = True
                img = np.frombuffer(buf, np.uint8).reshape(self.H, self.W, 3)
                now = time.monotonic()
                if self._t0 is None:
                    self._t0 = now
                ts = (now - self._t0) if self.live else self._file_ts(idx)
                if self.live and idx:
                    inst = idx / max(now - self._t0, 1e-6)
                    self.arrival_fps = inst if self.arrival_fps is None or idx < 30 else 0.95 * self.arrival_fps + 0.05 * inst
                fr = Frame(idx, ts, img)
                idx += 1
                if self.live:                       # keep only the newest frame
                    try:
                        self._q.get_nowait(); self.dropped += 1
                    except queue.Empty:
                        pass
                    self._q.put(fr)
                else:
                    while not self._stop.is_set():
                        try:
                            self._q.put(fr, timeout=0.2); break
                        except queue.Full:
                            continue
            if self._stop.is_set() and p.poll() is None:
                p.kill()                            # stopping: do not wait for the decoder to exit by itself
            err = p.stderr.read().decode(errors="replace").strip() if p.stderr else ""
            self._kill()
            if self._stop.is_set():
                break
            if self.kind == "file" and not self.cfg.get("loop"):
                if err:
                    log.warning("decoder: %s", err[-500:])
                break
            # back off (x2 per failure that delivered no frame, max 30 s) so a wrong URL does not spin
            failures = 0 if got_frames else failures + 1
            wait = min(30.0, self.cfg["reconnect_s"] * 2 ** min(failures, 5))
            log.warning("stream ended (%s); reconnecting in %.1fs", err[-200:] or "EOF", wait)
            self._stop.wait(wait)
        while True:                                 # end marker; after stop() the consumer may be gone
            try:
                self._q.put(None, timeout=0.2); break
            except queue.Full:
                if self._stop.is_set():
                    try:
                        self._q.get_nowait()
                    except queue.Empty:
                        pass

    def _kill(self):
        p, self._proc = self._proc, None
        if p and p.poll() is None:
            p.kill()
        if p:
            p.wait(timeout=5)

    # ---------------------------------------------------------------- public API
    def start(self):
        self._thread.start()
        return self

    def __iter__(self):
        if not self._thread.is_alive() and self.delivered == 0:
            self.start()
        while True:
            fr = self._q.get()
            if fr is None:
                return
            self.delivered += 1
            yield fr

    def stop(self):
        self._stop.set()
        self._kill()


class VideoSink:
    """Encode BGR frames with ffmpeg (H.264) to a file or a live stream (rtsp://, udp://, srt://, rtmp://)."""

    def __init__(self, uri, width, height, fps=15, bitrate="2M"):
        uri = str(uri)
        fmt = {"rtsp": "rtsp", "rtmp": "flv", "udp": "mpegts", "srt": "mpegts", "tcp": "mpegts"}.get(uri.split("://")[0])
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0", "-c:v", "libx264", "-preset", "ultrafast",
               "-tune", "zerolatency", "-b:v", bitrate, "-pix_fmt", "yuv420p", "-g", str(int(fps) * 2)]
        if fmt:
            cmd += ["-f", fmt] + (["-rtsp_transport", "tcp"] if fmt == "rtsp" else [])
        else:
            Path(uri).parent.mkdir(parents=True, exist_ok=True)
            cmd += ["-movflags", "+faststart"]
        cmd.append(uri)
        log.info("encoder: %s", " ".join(cmd))
        self.size = (width, height)
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, img):
        if img.shape[1::-1] != self.size:
            raise ValueError(f"frame {img.shape[1::-1]} != sink size {self.size}")
        try:
            self.p.stdin.write(np.ascontiguousarray(img).tobytes())
        except BrokenPipeError:
            log.error("encoder exited: %s", self.p.stderr.read().decode(errors="replace")[-300:])
            raise

    def close(self):
        if self.p.stdin:
            self.p.stdin.close()
        self.p.wait(timeout=30)
