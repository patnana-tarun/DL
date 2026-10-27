"""Local server for the blind annotation tool (standard library only; binds to 127.0.0.1).

    env -u PYTHONPATH .venv/bin/python research/annotation/serve.py [--port 8765]
    open http://127.0.0.1:8765/?task=det_main&annotator=A

GET  /                       annotator.html
GET  /tasks/<task>/...       manifest and images
GET  /labels?task=&annotator=   this annotator's saved work (never anyone else's)
POST /labels?task=&annotator=   {file: [{x1,y1,x2,y2,ignore,occ,tid}], ...} -> research/annotation/labels/<annotator>/<task>.json
"""
import argparse
import json
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
SAFE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(HERE), **k)

    def log_message(self, *a):
        pass

    def _target(self):
        q = parse_qs(urlparse(self.path).query)
        task, who = q.get("task", [""])[0], q.get("annotator", [""])[0]
        if not (SAFE.match(task) and SAFE.match(who)):
            return None
        return HERE / "labels" / who / f"{task}.json"

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)
        if u.path == "/":
            self.path = "/annotator.html"
        elif u.path == "/labels":
            t = self._target()
            return self._json(400, {"error": "bad task/annotator"}) if t is None else \
                self._json(200, json.loads(t.read_text()) if t.exists() else {})
        elif not (u.path.startswith("/tasks/") or u.path == "/annotator.html"):
            return self._json(404, {"error": "not found"})
        return super().do_GET()

    def do_POST(self):  # noqa: N802
        t = self._target()
        if urlparse(self.path).path != "/labels" or t is None:
            return self._json(400, {"error": "bad request"})
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        t.parent.mkdir(parents=True, exist_ok=True)
        tmp = t.with_suffix(".tmp"); tmp.write_text(json.dumps(body)); tmp.replace(t)
        return self._json(200, {"ok": True, "frames": len(body)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    print(f"annotation tool: http://127.0.0.1:{a.port}/?task=det_main&annotator=A")
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
