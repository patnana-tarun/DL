"""Publish video files as live network streams with FFmpeg, as if they were IP cameras.

    python tools/stream_videos.py ../videos/aa1.mp4 ../videos/video_1.mp4            # RTSP via MediaMTX
    python tools/stream_videos.py ../videos/aa1.mp4 --protocol udp --port 5000       # no server needed
    python tools/stream_videos.py ../videos/aa1.mp4 --protocol tcp --port 5000       # analyser connects to us

Protocols (the URL each stream is available at is printed on start):
  rtsp  ffmpeg pushes to an RTSP server (MediaMTX, started automatically if `mediamtx` is on PATH or in
        tools/bin/); any number of clients read rtsp://HOST:8554/cam<N>.  Closest to a real IP camera.
  udp   MPEG-TS over UDP to HOST:PORT+N (push; one receiver).
  tcp   MPEG-TS over TCP, ffmpeg listens on PORT+N and serves one client.
  srt   MPEG-TS over SRT, ffmpeg listens on PORT+N (low latency over lossy networks).
Each file is played in real time (-re) and looped forever (--once to play once). Video is passed through
without re-encoding (-c copy) unless --reencode (useful to lower resolution / bitrate, or for
sources with B-frames that a low-latency receiver dislikes).
"""
import argparse
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def find_mediamtx():
    for p in (shutil.which("mediamtx"), HERE / "bin" / "mediamtx"):
        if p and Path(p).exists():
            return str(p)
    return None


def port_open(host, port):
    import socket
    s = socket.socket(); s.settimeout(0.5)
    try:
        return s.connect_ex((host, port)) == 0
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--protocol", choices=["rtsp", "udp", "tcp", "srt"], default="rtsp")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=None, help="rtsp: server port (8554); others: first port (5000)")
    ap.add_argument("--once", action="store_true", help="play each file once instead of looping")
    ap.add_argument("--reencode", action="store_true", help="re-encode to H.264 (zerolatency) instead of copying")
    ap.add_argument("--scale", default="", help="with --reencode: e.g. 1280:720")
    a = ap.parse_args()
    port = a.port or (8554 if a.protocol == "rtsp" else 5000)
    procs = []
    if a.protocol == "rtsp" and not port_open(a.host, port):
        mtx = find_mediamtx()
        if not mtx:
            sys.exit("No RTSP server on %s:%d and `mediamtx` not found. Install it (see README) or use "
                     "--protocol udp|tcp|srt." % (a.host, port))
        cfgfile = HERE / "mediamtx.yml"
        procs.append(subprocess.Popen([mtx, str(cfgfile)] if cfgfile.exists() else [mtx]))
        for _ in range(50):
            if port_open(a.host, port):
                break
            time.sleep(0.1)
    for i, v in enumerate(a.videos, 1):
        if not Path(v).exists():
            sys.exit(f"not found: {v}")
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin", "-re"]
        if not a.once:
            cmd += ["-stream_loop", "-1"]
        cmd += ["-i", v, "-an"]
        if a.reencode:
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-g", "30", "-bf", "0"]
            if a.scale:
                cmd += ["-vf", f"scale={a.scale}"]
        else:
            cmd += ["-c:v", "copy"]
        if a.protocol == "rtsp":
            url = f"rtsp://{a.host}:{port}/cam{i}"
            cmd += ["-f", "rtsp", "-rtsp_transport", "tcp", url]
        elif a.protocol == "udp":
            url = f"udp://{a.host}:{port + i - 1}"
            cmd += ["-f", "mpegts", url + "?pkt_size=1316"]
        elif a.protocol == "tcp":
            url = f"tcp://{a.host}:{port + i - 1}"
            cmd += ["-f", "mpegts", f"tcp://0.0.0.0:{port + i - 1}?listen=1"]
        else:
            url = f"srt://{a.host}:{port + i - 1}"
            cmd += ["-f", "mpegts", f"srt://0.0.0.0:{port + i - 1}?mode=listener"]
        procs.append(subprocess.Popen(cmd))
        print(f"cam{i}: {v}  ->  {url}", flush=True)

    def stop(*_):
        for p in procs:
            p.terminate()
        sys.exit(0)
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    while True:
        time.sleep(1)
        streams = procs[1:] if a.protocol == "rtsp" and len(procs) > len(a.videos) else procs
        if all(p.poll() is not None for p in streams):
            print("all streams ended"); stop()


if __name__ == "__main__":
    main()
