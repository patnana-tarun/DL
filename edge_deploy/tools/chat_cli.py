"""Talk to the running EdgeVision pipeline (and its VLM) from a terminal, e.g. over SSH on the Jetson.

    python tools/chat_cli.py                       # http://127.0.0.1:8000
    python tools/chat_cli.py --url http://jetson.local:8000
    python tools/chat_cli.py -q "how many people are in view?"
    python tools/chat_cli.py --token SECRET        # when server.token is set
Type a question, or a /command (/help lists them). Ctrl-D or /quit to exit.
"""
import argparse
import json
import sys
import urllib.request


def ask(url, msg, timeout=300, token=""):
    req = urllib.request.Request(url.rstrip("/") + "/api/chat", data=json.dumps({"message": msg}).encode(),
                                 headers={"Content-Type": "application/json", **({"X-Token": token} if token else {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def show(r):
    print(r.get("answer", ""))
    for a in r.get("actions") or []:
        print(f"  [action] {a.get('tool')} {json.dumps(a.get('args', ''))} -> {json.dumps(a.get('result'))[:200]}")
    if r.get("latency_s"):
        print(f"  ({r['latency_s']} s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("-q", "--question", default="")
    ap.add_argument("--token", default="", help="server.token, if the server requires one")
    a = ap.parse_args()
    if a.question:
        show(ask(a.url, a.question, token=a.token)); return
    print(f"connected to {a.url} - ask a question or type /help")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if q in ("/quit", "/exit"):
            break
        if q:
            try:
                show(ask(a.url, q, token=a.token))
            except OSError as e:
                print(f"error: {e} (is the pipeline running?)", file=sys.stderr)


if __name__ == "__main__":
    main()
