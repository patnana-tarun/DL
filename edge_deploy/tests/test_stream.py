import socket
import subprocess
import threading
import time

import numpy as np
import pytest
from conftest import MEDIAMTX, collect, ffprobe, free_port, needs_ffmpeg, source_cfg, wait_port

from edgevision import stream
from edgevision.stream import VideoSink, VideoSource, build_command, kind_of

pytestmark = needs_ffmpeg


# ---------------------------------------------------------------- helpers
def read_file(uri, **kw):
    src = VideoSource(source_cfg(uri, **kw))
    frames, finished = collect(src, timeout=60)
    assert finished, "file source did not reach EOF"
    return src, frames


def _publisher(url_args, seconds=30, size="320x240", rate=30):
    return subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-re", "-f", "lavfi",
                             "-i", f"testsrc=size={size}:rate={rate}", "-t", str(seconds), "-c:v", "libx264",
                             "-preset", "ultrafast", "-tune", "zerolatency", "-g", str(rate), "-bf", "0",
                             "-pix_fmt", "yuv420p", *url_args],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _terminate(p):
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill(); p.wait(timeout=5)


# ---------------------------------------------------------------- kinds and commands
@pytest.mark.parametrize("uri, kind", [
    ("videos/aa1.mp4", "file"), ("/abs/x.mkv", "file"), ("rtsp://h:8554/cam1", "network"),
    ("rtsps://h/cam", "network"), ("udp://127.0.0.1:5000", "network"), ("tcp://h:1", "network"),
    ("srt://h:1", "network"), ("http://h/x.m3u8", "network"), ("rtmp://h/live", "network"),
    ("/dev/video0", "v4l2"), ("v4l2:///dev/video2", "v4l2"), ("csi://0", "csi"),
])
def test_kind_of(uri, kind):
    assert kind_of(uri) == kind


def _vf(cmd):
    return cmd[cmd.index("-vf") + 1]


def test_build_command_file_options():
    cmd = build_command(source_cfg("in.mp4", loop=True, realtime=True, start_s=1.5, duration_s=4,
                                   hwaccel="cuda", input_args=["-thread_queue_size", "64"]))
    i = cmd.index("-i")
    assert cmd[0] == "ffmpeg" and cmd[i + 1] == "in.mp4"
    assert cmd.index("-stream_loop") < i and cmd[cmd.index("-stream_loop") + 1] == "-1"
    assert cmd.index("-re") < i
    assert cmd.index("-ss") < i and cmd[cmd.index("-ss") + 1] == "1.5"
    assert cmd.index("-t") > i and cmd[cmd.index("-t") + 1] == "4"
    assert cmd[cmd.index("-hwaccel") + 1] == "cuda"
    assert cmd.index("-thread_queue_size") < i
    assert cmd[-5:] == ["-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
    assert "scale=320:240:force_original_aspect_ratio=decrease,pad=320:240:(ow-iw)/2:(oh-ih)/2" in _vf(cmd)
    assert "-fflags" not in cmd and "-rtsp_transport" not in cmd


def test_build_command_plain_file_has_no_extras():
    cmd = build_command(source_cfg("in.mp4"))
    for flag in ("-stream_loop", "-re", "-ss", "-t", "-hwaccel"):
        assert flag not in cmd
    assert _vf(cmd).startswith("scale=")
    # without passthrough, rawvideo output resamples variable-frame-rate input to a constant rate (duplicated frames)
    assert cmd[cmd.index("-fps_mode") + 1] == "passthrough"


def test_media_ts_follows_real_timestamps_of_vfr_files():
    from edgevision.stream import media_ts
    pts = [0.0, 1 / 60, 2 / 60, 2 / 60 + 1 / 45, 2 / 60 + 2 / 45, 2 / 60 + 3 / 45]     # 60 fps, then 45 fps
    assert np.allclose(media_ts(pts), pts)
    assert np.allclose(media_ts(pts, stride=2), [0.0, 2 / 60, 2 / 60 + 2 / 45])
    assert np.allclose(media_ts(pts, start_s=0.03), [0.0, 1 / 45, 2 / 45, 3 / 45])
    assert np.allclose(media_ts(pts, duration_s=0.05), [0.0, 1 / 60, 2 / 60])


def test_build_command_stride_and_fps():
    cmd = build_command(source_cfg("in.mp4", stride=3))
    assert _vf(cmd).startswith("select=not(mod(n\\,3)),scale=")
    assert cmd[cmd.index("-fps_mode") + 1] == "passthrough"
    cmd = build_command(source_cfg("in.mp4", fps=12.5))
    assert _vf(cmd).startswith("fps=12.5,scale=")
    assert "-fps_mode" not in cmd


def test_build_command_rtsp():
    cmd = build_command(source_cfg("rtsp://10.1.2.3:8554/cam1", rtsp_transport="udp"))
    i = cmd.index("-i")
    assert cmd[i + 1] == "rtsp://10.1.2.3:8554/cam1"
    assert cmd[cmd.index("-rtsp_transport") + 1] == "udp" and cmd.index("-rtsp_transport") < i
    assert cmd[cmd.index("-fflags") + 1] == "nobuffer" and cmd[cmd.index("-flags") + 1] == "low_delay"
    for flag in ("-re", "-stream_loop", "-ss", "-t"):                 # file-only options never leak
        assert flag not in build_command(source_cfg("rtsp://h/c", realtime=True, loop=True, start_s=3,
                                                    duration_s=2))


def test_build_command_udp():
    cmd = build_command(source_cfg("udp://127.0.0.1:5000"))
    assert cmd[cmd.index("-i") + 1] == "udp://127.0.0.1:5000"
    assert "-rtsp_transport" not in cmd and "nobuffer" in cmd


def test_build_command_v4l2():
    cmd = build_command(source_cfg("/dev/video0", fps=15))
    i = cmd.index("-i")
    assert cmd[i + 1] == "/dev/video0"
    assert cmd[cmd.index("-f") + 1] == "v4l2" and cmd.index("-f") < i
    assert cmd[cmd.index("-video_size") + 1] == "320x240"
    assert cmd[cmd.index("-framerate") + 1] == "15"
    assert "fps=" not in _vf(cmd)                     # the camera delivers the rate; no resampling filter
    cmd = build_command(source_cfg("v4l2:///dev/video2"))
    assert cmd[cmd.index("-i") + 1] == "/dev/video2" and "-framerate" not in cmd


def test_build_command_csi():
    cmd = build_command(source_cfg("csi://1", width=1280, height=720))
    assert cmd[:2] == ["gst-launch-1.0", "-q"]
    s = " ".join(cmd)
    assert "nvarguscamerasrc" in cmd and "sensor-id=1" in cmd
    assert "width=1280,height=720,framerate=30/1" in s
    assert "format=BGR" in s and cmd[-3:] == ["fdsink", "fd=1", "sync=false"]
    assert "framerate=15/1" in " ".join(build_command(source_cfg("csi://0", fps=15)))
    assert "sensor-id=0" in build_command(source_cfg("csi://"))


def test_constructing_camera_sources_does_not_spawn(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("no process may be started")
    monkeypatch.setattr(stream.subprocess, "Popen", boom)
    monkeypatch.setattr(stream.subprocess, "run", boom)
    for uri, kind in (("csi://0", "csi"), ("/dev/video0", "v4l2"), ("rtsp://h/c", "network")):
        s = VideoSource(source_cfg(uri))
        assert s.kind == kind and s.live and s.fps == 30.0 and s.src_fps is None
    s = VideoSource(source_cfg("csi://0", stride=2))
    assert s.fps == 15.0
    s = VideoSource(source_cfg("rtsp://h/c", realtime=False))
    assert not s.live


# ---------------------------------------------------------------- files
def test_probe_fps(videos):
    assert stream.probe_fps(videos["std"]) == pytest.approx(30.0)
    assert stream.probe_fps("/nonexistent/file.mp4") is None


def test_file_exact_frames_shape_and_ts(videos):
    src, frames = read_file(videos["std"])
    assert src.kind == "file" and not src.live and src.fps == pytest.approx(30.0)
    assert len(frames) == 60
    assert [f.index for f in frames] == list(range(60))
    for f in frames:
        assert f.image.shape == (240, 320, 3) and f.image.dtype == np.uint8
        assert f.ts == pytest.approx(f.index / 30.0, abs=1e-5)   # media timestamps (ffprobe, µs)
    assert not np.array_equal(frames[0].image, frames[30].image)      # testsrc changes over time
    assert src.dropped == 0 and src.delivered == 60


def test_file_resize_to_configured_size(videos):
    src, frames = read_file(videos["std"], width=640, height=480)
    assert len(frames) == 60 and frames[0].image.shape == (480, 640, 3)


def test_stride_two_halves_frames_with_exact_mapping(videos):
    _, full = read_file(videos["std"])
    src, half = read_file(videos["std"], stride=2)
    assert src.fps == pytest.approx(15.0)
    assert len(half) == 30
    for k, f in enumerate(half):
        assert f.index == k and f.ts == pytest.approx(k / 15.0, abs=1e-5)
        assert np.array_equal(f.image, full[2 * k].image), f"stride frame {k} != decoded frame {2 * k}"


def test_fps_resample(videos):
    src, frames = read_file(videos["std"], fps=10)
    assert src.fps == 10.0
    assert len(frames) == 20
    assert [f.ts for f in frames] == pytest.approx([i / 10 for i in range(20)])


def test_time_window(videos):
    _, full = read_file(videos["std"])
    src, frames = read_file(videos["std"], start_s=0.5, duration_s=1.0)
    assert len(frames) == 30
    assert frames[0].ts == 0.0 and frames[-1].ts == pytest.approx(29 / 30)
    diff = np.abs(frames[0].image.astype(int) - full[15].image.astype(int)).mean()
    assert diff < 2.0, "the window should start at media time 0.5 s (frame 15)"
    _, tail = read_file(videos["std"], start_s=1.5)
    assert len(tail) == 15


def _is_pad(region, tol=20):
    return region.max() <= tol


def test_letterbox_wide_source_padded_top_bottom(videos):
    _, frames = read_file(videos["wide"])                   # 320x120 -> 320x240: 60 px bars top and bottom
    img = frames[10].image
    assert len(frames) == 60 and img.shape == (240, 320, 3)
    assert _is_pad(img[:58]) and _is_pad(img[183:])
    assert img[62:178].mean() > 40                          # picture content in the middle
    assert img[62:178].std() > 20


def test_letterbox_tall_source_padded_left_right(videos):
    _, frames = read_file(videos["tall"])                   # 120x240 -> 320x240: 100 px bars left and right
    img = frames[10].image
    assert _is_pad(img[:, :98]) and _is_pad(img[:, 222:])
    assert img[:, 102:218].mean() > 40


def test_file_mode_never_drops_with_slow_consumer(videos):
    src = VideoSource(source_cfg(videos["std"]))
    frames, finished = collect(src, timeout=60, per_frame=lambda fr: time.sleep(0.03))
    assert finished
    assert len(frames) == 60 and [f.index for f in frames] == list(range(60))
    assert src.dropped == 0


def test_stop_from_another_thread_ends_iteration(videos):
    """stop() while the consumer iterates (e.g. Ctrl-C handler) must end the iteration, not hang it."""
    src = VideoSource(source_cfg(videos["std"]))
    frames, finished = [], threading.Event()

    def consume():
        for fr in src:
            frames.append(fr)
            time.sleep(0.05)                               # slow consumer: the reader waits on a full queue
        finished.set()

    th = threading.Thread(target=consume, daemon=True)
    th.start()
    t = time.time()
    while len(frames) < 3 and time.time() - t < 20:
        time.sleep(0.01)
    assert len(frames) >= 3
    src.stop()
    assert finished.wait(10), "iteration did not end after stop()"
    assert len(frames) < 60
    src._thread.join(5)
    assert not src._thread.is_alive(), "reader thread still running after stop()"


def test_stop_after_abandoned_iteration_releases_reader(videos):
    """A consumer that breaks out early (max_frames) and calls stop() must not leak the reader thread."""
    src = VideoSource(source_cfg(videos["std"]))
    frames, finished = collect(src, max_frames=5, timeout=20)
    assert finished and len(frames) == 5
    time.sleep(0.3)                                       # reader fills the queue and blocks on it
    src.stop()
    src._thread.join(5)
    assert not src._thread.is_alive(), "reader thread blocked after stop()"


def test_missing_file_ends_cleanly(tmp_path):
    src = VideoSource(source_cfg(tmp_path / "missing.mp4"))
    frames, finished = collect(src, timeout=20)
    assert finished and frames == []


# ---------------------------------------------------------------- live network sources
def test_live_udp_stream():
    port = free_port(socket.SOCK_DGRAM)
    src = VideoSource(source_cfg(f"udp://127.0.0.1:{port}"))
    assert src.live and src.kind == "network"                # realtime None -> auto -> live
    pub = None
    try:
        src.start()
        time.sleep(0.5)                                      # let the receiver bind first
        pub = _publisher(["-f", "mpegts", f"udp://127.0.0.1:{port}?pkt_size=1316"])
        frames, _ = collect(src, max_seconds=2.0, timeout=25)
        assert len(frames) >= 10, f"only {len(frames)} frames in ~2 s"
        assert all(f.image.shape == (240, 320, 3) for f in frames)
        ts = [f.ts for f in frames]
        assert ts == sorted(ts) and ts[0] >= 0
        idx = [f.index for f in frames]
        assert idx == sorted(idx) and len(set(idx)) == len(idx)
        # live ts is wall time since the first frame; ~2 s of reading after that
        assert ts[-1] == pytest.approx(frames[-1].wall - frames[0].wall + ts[0], abs=0.3)
        assert src.delivered == len(frames)
    finally:
        src.stop()
        if pub:
            _terminate(pub)
    src._thread.join(5)
    assert not src._thread.is_alive()


def test_live_source_drops_stale_frames_for_slow_consumer():
    port = free_port(socket.SOCK_DGRAM)
    src = VideoSource(source_cfg(f"udp://127.0.0.1:{port}"))
    pub = None
    try:
        src.start()
        time.sleep(0.5)
        pub = _publisher(["-f", "mpegts", f"udp://127.0.0.1:{port}?pkt_size=1316"])
        frames, _ = collect(src, max_seconds=3.0, timeout=25, per_frame=lambda fr: time.sleep(0.2))
        assert len(frames) >= 3
        assert src.dropped > 0, "a live source must drop frames the consumer cannot keep up with"
        assert frames[-1].index - frames[0].index > len(frames)
    finally:
        src.stop()
        if pub:
            _terminate(pub)


@pytest.mark.slow
@pytest.mark.skipif(not MEDIAMTX.exists(), reason="mediamtx binary missing")
def test_live_rtsp_via_mediamtx(tmp_path):
    port = free_port()
    cfg = tmp_path / "mediamtx.yml"
    cfg.write_text(f"""logLevel: warn
logDestinations: [stdout]
rtsp: true
rtspAddress: 127.0.0.1:{port}
rtspTransports: [tcp]
rtspEncryption: "no"
rtmp: false
hls: false
webrtc: false
srt: false
moq: false
api: false
metrics: false
pprof: false
playback: false
paths:
  all_others:
""")
    mtx = subprocess.Popen([str(MEDIAMTX), str(cfg)], cwd=tmp_path, stdin=subprocess.DEVNULL,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    pub = src = None
    try:
        assert wait_port(port, 10), f"mediamtx did not listen on {port}: {mtx.poll()}"
        url = f"rtsp://127.0.0.1:{port}/test"
        pub = _publisher(["-f", "rtsp", "-rtsp_transport", "tcp", url])
        time.sleep(1.5)                                      # publisher announces the path
        assert pub.poll() is None, pub.stderr.read().decode()
        src = VideoSource(source_cfg(url, reconnect_s=0.5))
        assert src.live and src.kind == "network"
        frames, _ = collect(src, max_seconds=2.0, timeout=30)
        assert len(frames) >= 10
        assert frames[0].image.shape == (240, 320, 3)
        assert frames[0].image.std() > 20                   # real picture, not padding
    finally:
        if src:
            src.stop()
        if pub:
            _terminate(pub)
        _terminate(mtx)


# ---------------------------------------------------------------- sink
def test_video_sink_mp4(tmp_path):
    out = tmp_path / "sub" / "out.mp4"
    sink = VideoSink(out, 320, 240, fps=15)
    rng = np.random.default_rng(0)
    for i in range(30):
        img = np.full((240, 320, 3), i * 8, np.uint8)
        img[100:140, i * 10:i * 10 + 20] = rng.integers(0, 255, 3, dtype=np.uint8)
        sink.write(img)
    sink.close()
    info = ffprobe(out)
    assert info == {"codec": "h264", "width": 320, "height": 240, "frames": 30}
    assert stream.probe_fps(out) == pytest.approx(15.0)


def test_video_sink_rejects_wrong_size(tmp_path):
    sink = VideoSink(tmp_path / "o.mp4", 320, 240)
    with pytest.raises(ValueError, match="sink size"):
        sink.write(np.zeros((240, 321, 3), np.uint8))
    sink.write(np.zeros((240, 320, 3), np.uint8))
    sink.close()
    assert ffprobe(tmp_path / "o.mp4")["frames"] == 1


def test_sink_then_source_round_trip(tmp_path):
    out = tmp_path / "rt.mp4"
    sink = VideoSink(out, 320, 240, fps=10)
    for i in range(12):
        sink.write(np.full((240, 320, 3), 20 * i, np.uint8))
    sink.close()
    src, frames = read_file(out)
    assert src.fps == pytest.approx(10.0) and len(frames) == 12
    assert [round(float(f.image.mean()) / 20) for f in frames] == list(range(12))
