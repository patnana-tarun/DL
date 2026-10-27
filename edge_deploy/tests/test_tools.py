"""tools/chat_cli.py and tools/stream_videos.py."""
import importlib.util
import json
import signal
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from conftest import EDGE, collect, free_port, make_video, needs_ffmpeg, source_cfg

from edgevision.stream import VideoSource

TOOLS = EDGE / "tools"


def load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def chat_server():
    seen = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, body))
            out = json.dumps({"answer": f"echo {body['message']}", "latency_s": 0.5,
                              "actions": [{"tool": "set_alert", "args": {"value": 3}, "result": {"ok": True}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    port = free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}", seen
    httpd.shutdown()
    httpd.server_close()


def test_chat_cli_ask_and_show(chat_server, capsys):
    cli = load("chat_cli")
    url, seen = chat_server
    r = cli.ask(url + "/", "/stats")
    assert seen == [("/api/chat", {"message": "/stats"})]
    assert r["answer"] == "echo /stats"
    cli.show(r)
    out = capsys.readouterr().out
    assert "echo /stats" in out and "[action] set_alert" in out and "(0.5 s)" in out


def test_chat_cli_one_shot(chat_server):
    url, _ = chat_server
    p = subprocess.run([sys.executable, str(TOOLS / "chat_cli.py"), "--url", url, "-q", "hello"],
                       capture_output=True, text=True, timeout=30)
    assert p.returncode == 0 and "echo hello" in p.stdout


def test_chat_cli_interactive_quit(chat_server):
    url, seen = chat_server
    p = subprocess.run([sys.executable, str(TOOLS / "chat_cli.py"), "--url", url], input="hi\n/quit\n",
                       capture_output=True, text=True, timeout=30)
    assert p.returncode == 0 and "echo hi" in p.stdout and len(seen) == 1


def test_stream_videos_helpers():
    sv = load("stream_videos")
    assert sv.find_mediamtx() is not None                     # tools/bin/mediamtx ships with the folder
    port = free_port()
    assert not sv.port_open("127.0.0.1", port)
    with socket.socket() as s:
        s.bind(("127.0.0.1", port)); s.listen(1)
        assert sv.port_open("127.0.0.1", port)


def test_stream_videos_missing_file():
    p = subprocess.run([sys.executable, str(TOOLS / "stream_videos.py"), "/nonexistent.mp4", "--protocol", "udp",
                        "--port", str(free_port(socket.SOCK_DGRAM))], capture_output=True, text=True, timeout=30)
    assert p.returncode != 0 and "not found" in p.stderr


@needs_ffmpeg
def test_stream_videos_udp_feeds_video_source(tmp_path):
    video = make_video(tmp_path / "v.mp4", seconds=3)
    port = free_port(socket.SOCK_DGRAM)
    src = VideoSource(source_cfg(f"udp://127.0.0.1:{port}"))
    src.start()
    pub = subprocess.Popen([sys.executable, str(TOOLS / "stream_videos.py"), str(video), "--protocol", "udp",
                            "--port", str(port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        frames, _ = collect(src, max_seconds=1.5, timeout=30)
        assert len(frames) >= 10 and frames[0].image.shape == (240, 320, 3)
    finally:
        src.stop()
        pub.send_signal(signal.SIGTERM)
        out, _ = pub.communicate(timeout=10)
    assert f"cam1: {video}  ->  udp://127.0.0.1:{port}" in out
    assert pub.returncode == 0
