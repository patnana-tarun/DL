"""Vision-language model in the loop (Qwen3-VL through any OpenAI-compatible server).

Closed loop
  detector + tracker -> analytics --(numbers, events)--> VLM --(tool calls)--> analytics configuration
  1. observe   every `observe_every_s` the VLM gets the current frame + the analytics numbers and writes
               a short structured scene report (stored, shown in the UI).
  2. on alert  when an alert rule fires, the VLM looks at the frame and writes a scene report for it
               (`alert_context` event). The rule itself is decided by the detector numbers: with a yes/no
               verdict prompt, Qwen3-VL-2B confirmed 100 % of fabricated count alerts when shown the measured
               value and was below chance without it (results/vlm/e5_verify.json), so verdict modes
               ("blind", "informed") are opt-in research options, not the default.
  3. ask       the user chats with the VLM in two stages. Stage 1 (routing) sees the question, the chat
               history and the analytics context as text only, and may call tools to read history or change
               the analytics (add / remove alert rules, move the counting line); changes apply on the next
               frame, so the conversation steers what the pipeline watches for. Stage 2 (answer) sees the live
               frame, the updated measurements and the tool results, with no tools offered.
               Why two stages: with the frame attached, Qwen3-VL-2B called read-only tools on 75 % of plain
               questions; text-only routing cut that to 6 % (results/vlm/e3_tools.json).
Every tool call is decided by the trusted policy layer (policy.py) when one is attached: role, argument
validation, an intent gate (a change must be asked for in the user's own words), rate limits and /confirm for
removing or weakening alerts, all hash-chained in the audit log. With policy.isolate_untrusted, stage 1 never
sees text the VLM wrote from an image (scene reports, earlier answers, VLM fields of events), so words in the
camera view cannot reach the component that can change settings.
The analytics numbers are the source of truth for counts (a 2B VLM miscounts crowds); the VLM adds what
numbers cannot: what people are doing, unusual situations, and natural-language access to the data.
"""
import base64
import heapq
import itertools
import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import Future

import cv2

log = logging.getLogger("edgevision.vlm")

SYSTEM = """You are the on-device video analyst of a camera system. You see the latest camera frame and live \
measurements from a person detector + tracker running on every frame.

Rules:
- For counts (people in view, unique people, crossings) use the measurements: they are more accurate than \
counting in the image. Say "about" when you estimate from the image yourself.
- Describe what you can actually see; if the image does not show something, say so. Do not identify people.
- Be brief: 1-4 sentences unless the user asks for detail.
- To change what the system watches (alerts, counting line) or to read history, call the tools.

Measurement definitions: occupancy = people tracked in the current frame; unique_people = tracks that lasted \
>= 0.25 s since start (an upper bound on distinct people); crossings = people crossing the counting line, by \
direction; dwell = time a person stayed in view; coordinates are normalised image coordinates (0-1)."""

OBSERVE = """Write a short scene report for the operator log, about what you SEE in the image (the operator
already has the numbers: do not repeat counts or measurements). Reply with JSON only:
{"summary": "<one or two sentences: the place and what people are doing>",
 "activities": ["<short phrase>", ...],
 "anomalies": ["<visible unusual or unsafe situation, or leave empty>"],
 "risk": "none|low|medium|high"}
"risk" is about visible physical danger only: someone fallen, a fight, fire or smoke, a vehicle among
pedestrians, a crush or stampede. A busy or crowded but orderly scene is "none" or "low". Alert rules and
high counts alone are not a risk."""

VERIFY = """An automatic alert just fired: rule "{rule}" ({metric} {op} {threshold}; measured {value}).
Look at the image. Is the alert situation real? Reply with JSON only:
{{"confirmed": true|false, "reason": "<one sentence>"}}"""

# "blind" verification: the measured value and the live measurements are withheld, so the VLM has to judge
# from the image alone (the informed prompt invites copying the number, see results/vlm/e1_counting.json).
VERIFY_BLIND = """An automatic rule fired: "{rule}" (condition: {metric} {op} {threshold}).
Judge ONLY from the image, independently of the rule. Does the image actually show this situation?
Reply with JSON only:
{{"confirmed": true|false, "reason": "<one sentence describing what you see>"}}"""

METRIC_HELP = {
    "occupancy": "people in view right now",
    "unique_people": "distinct people seen since start",
    "crossings_total": "people who crossed the counting line since start",
    "crossings_last_min": "line crossings during the last minute",
    "max_dwell_s": "seconds the longest-present person has been in view",
    "loitering": "people present longer than the loitering limit",
}

TOOLS = [
    {"type": "function", "function": {
        "name": "get_history", "description": "Occupancy statistics and event counts for the last N minutes.",
        "parameters": {"type": "object", "properties": {"minutes": {"type": "number", "description": "1-1440"}},
                       "required": ["minutes"]}}},
    {"type": "function", "function": {
        "name": "get_recent_events", "description": "List recent analytics events (alerts, line crossings, loitering...).",
        "parameters": {"type": "object", "properties": {
            "minutes": {"type": "number"},
            "kind": {"type": "string", "enum": ["alert", "alert_verified", "line_cross", "loitering", "person_enter",
                                                "person_exit", "zone_enter", "zone_exit", "any"]}},
            "required": ["minutes"]}}},
    {"type": "function", "function": {
        "name": "set_alert",
        "description": "Create or replace an alert rule that is checked on every frame. Examples: 'more than 10 "
                       "people' -> occupancy > 10; 'someone stays longer than 2 minutes' -> max_dwell_s > 120; "
                       "'someone loiters' -> loitering > 0; 'over 5 people cross per minute' -> crossings_last_min > 5.",
        "parameters": {"type": "object", "properties": {
            "metric": {"type": "string", "enum": ["occupancy", "unique_people", "crossings_total", "crossings_last_min",
                                                  "max_dwell_s", "loitering"],
                       "description": "; ".join(f"{k} = {v}" for k, v in METRIC_HELP.items())},
            "op": {"type": "string", "enum": [">", ">=", "<", "<="],
                   "description": "'more than', 'over', 'exceeds' -> '>'; 'at least' -> '>='; 'fewer than', "
                                  "'below' -> '<'; 'at most' -> '<='"},
            "value": {"type": "number", "description": "threshold; durations in seconds (convert minutes)"},
            "for_s": {"type": "number", "description": "condition must hold this many seconds (default 3)"},
            "name": {"type": "string"}},
            "required": ["metric", "op", "value"]}}},
    {"type": "function", "function": {
        "name": "remove_alert", "description": "Remove an alert rule by name, or all rules with name 'all'.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "set_counting_line",
        "description": "Move the people-counting line. Coordinates are normalised 0-1 (x = left to right, "
                       "y = top to bottom). HORIZONTAL line across the middle: x1=0, y1=0.5, x2=1, y2=0.5 "
                       "(y stays the same). VERTICAL line through the centre: x1=0.5, y1=0, x2=0.5, y2=1 "
                       "(x stays the same). Vertical line at one third of the width: x1=0.33, y1=0, x2=0.33, y2=1.",
        "parameters": {"type": "object", "properties": {k: {"type": "number"} for k in ("x1", "y1", "x2", "y2")},
                       "required": ["x1", "y1", "x2", "y2"]}}},
]


def intent_error(question, name, args):
    """Deterministic guardrail: reject tool arguments that contradict an explicit word in the user's request
    (a 2B model sometimes copies the wrong example). -> error message for the model, or None."""
    q = question.lower()
    if name == "set_counting_line":
        try:
            x1, y1, x2, y2 = (float(args[k]) for k in ("x1", "y1", "x2", "y2"))
        except (KeyError, TypeError, ValueError):
            return None                                 # run_tool reports malformed arguments itself
        if "horizontal" in q and not (abs(y1 - y2) < 1e-6 and abs(x2 - x1) > 0.5):
            return ("the user asked for a HORIZONTAL line: y1 must equal y2 and x should run from 0 to 1, "
                    "e.g. x1=0, y1=0.5, x2=1, y2=0.5 for the middle. Call set_counting_line again.")
        if "vertical" in q and not (abs(x1 - x2) < 1e-6 and abs(y2 - y1) > 0.5):
            return ("the user asked for a VERTICAL line: x1 must equal x2 and y should run from 0 to 1, "
                    "e.g. x1=0.5, y1=0, x2=0.5, y2=1 for the centre. Call set_counting_line again.")
    return None


CHANGE_WORDS = re.compile(r"\b(alert|alarm|notify|warn|ping|rule|line|remove|delete|clear|turn off|set|change|move|"
                          r"make|put)\b", re.I)


def _position(q, default=0.5):
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", q)
    if m:
        return min(1.0, max(0.0, float(m.group(1)) / 100))
    for word, v in (("one third", 1 / 3), ("a third", 1 / 3), ("two thirds", 2 / 3), ("quarter", 0.25),
                    ("middle", 0.5), ("centre", 0.5), ("center", 0.5)):
        if word in q:
            return v
    return default


def fallback_intents(question):
    """Unambiguous control requests recognised without the model: -> [(tool, args)]. Used only when the model did
    not already perform that change successfully (2B models sometimes skip or garble these calls)."""
    q = question.lower()
    out = []
    if re.search(r"\b(remove|delete|clear|turn off|disable)\b.*\b(all|every)\b.*\b(alerts?|rules?|alarms?)\b", q):
        out.append(("remove_alert", {"name": "all"}))
    if "line" in q and re.search(r"\b(set|make|move|put|change|draw)\b", q):
        if "horizontal" in q:
            y = _position(q)
            out.append(("set_counting_line", {"x1": 0.0, "y1": y, "x2": 1.0, "y2": y}))
        elif "vertical" in q:
            x = _position(q)
            out.append(("set_counting_line", {"x1": x, "y1": 0.0, "x2": x, "y2": 1.0}))
    return out


def _ok(result):
    return not (isinstance(result, dict) and result.get("ok") is False)


def _describe_action(a):
    args = a["args"] if isinstance(a["args"], dict) else {}
    if a["tool"] == "set_alert" and _ok(a["result"]):
        r = a["result"]["rule"]
        return f"alert rule '{r['name']}' ({r['metric']} {r['op']} {r['value']:g} for {r['for_s']:g} s)"
    if a["tool"] == "remove_alert":
        return f"removed {a['result'].get('removed', 0)} alert rule(s)"
    if a["tool"] == "set_counting_line" and _ok(a["result"]):
        return f"counting line {a['result']['orientation']} {a['result']['line']}"
    return f"{a['tool']} {json.dumps(args)}"


UNTRUSTED_FIELDS = ("summary", "anomalies", "reason", "activities", "text")


def strip_untrusted(e):
    """Event without the free-text fields a VLM wrote from an image (they may carry words seen in the frame)."""
    return {k: v for k, v in e.items() if k not in UNTRUSTED_FIELDS}


def encode_image(img, max_side=896, quality=85):
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def parse_json(text):
    """First JSON object in a model reply (tolerates ``` fences and surrounding prose)."""
    if not text:
        return None
    t = re.sub(r"```(?:json)?", "", text)
    start = t.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(t)):
            depth += {"{": 1, "}": -1}.get(t[i], 0)
            if depth == 0:
                try:
                    return json.loads(t[start:i + 1])
                except json.JSONDecodeError:
                    break
        start = t.find("{", start + 1)
    return None


def text_tool_calls(text):
    """Tool calls written as text (<tool_call>{...}</tool_call>), for servers that do not parse them."""
    calls = []
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text or "", re.S):
        try:
            d = json.loads(m.group(1))
            calls.append({"id": f"t{len(calls)}", "function": {"name": d["name"],
                                                               "arguments": json.dumps(d.get("arguments", {}))}})
        except (json.JSONDecodeError, KeyError):
            continue
    return calls


class VLMClient:
    """Minimal OpenAI-compatible chat client (stdlib only)."""

    def __init__(self, cfg):
        self.base = cfg["endpoint"].rstrip("/")
        self.model, self.key, self.timeout = cfg["model"], cfg.get("api_key", ""), cfg["timeout_s"]

    def _post(self, path, payload):
        req = urllib.request.Request(self.base + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {self.key}"} if self.key else {})})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def healthy(self):
        try:
            with urllib.request.urlopen(self.base + "/models", timeout=3) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def chat(self, messages, tools=None, max_tokens=384, temperature=0.2):
        """-> (assistant message dict, info dict with latency and token usage)."""
        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                   "stream": False}
        if tools:
            payload["tools"] = tools
        t = time.time()
        r = self._post("/chat/completions", payload)
        msg = r["choices"][0]["message"]
        u = r.get("usage", {})
        info = {"latency_s": round(time.time() - t, 3), "prompt_tokens": u.get("prompt_tokens"),
                "completion_tokens": u.get("completion_tokens"), "timings": r.get("timings")}
        return msg, info


class SceneAgent:
    """Owns the VLM: one worker thread, jobs served by priority (chat > alert check > periodic report)."""

    def __init__(self, cfg, analytics, store, frames, policy=None):
        """frames: callable -> (raw BGR frame, annotated BGR frame, ts) or (None, None, None).
        policy: policy.Policy deciding every tool call (None = no policy layer, the pre-revision behaviour)."""
        self.cfg, self.an, self.store, self.frames, self.policy = cfg, analytics, store, frames, policy
        self.client = VLMClient(cfg)
        self.history = []                    # [(user, assistant)] for the chat
        self.footers = []                    # system-written (trusted) summary of each turn's changes
        self._jobs, self._seq = [], itertools.count()
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self.busy = False
        self.last_latency = None
        self.online = False
        self._thread = threading.Thread(target=self._worker, name="vlm", daemon=True)

    # ---------------------------------------------------------------- job queue
    def start(self):
        self.online = self.client.healthy()
        log.info("VLM endpoint %s: %s", self.client.base, "online" if self.online else "OFFLINE (will retry)")
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        with self._cv:
            for _, _, fut, _, _ in self._jobs:          # fail queued jobs fast instead of letting callers time out
                if fut.set_running_or_notify_cancel():
                    fut.set_exception(RuntimeError("VLM agent stopped"))
            self._jobs.clear()
            self._cv.notify_all()

    def submit(self, priority, fn, *args):
        fut = Future()
        with self._cv:
            heapq.heappush(self._jobs, (priority, next(self._seq), fut, fn, args))
            self._cv.notify()
        return fut

    def pending(self, priority=None):
        with self._cv:
            return sum(1 for j in self._jobs if priority is None or j[0] == priority)

    def _worker(self):
        while not self._stop.is_set():
            with self._cv:
                while not self._jobs and not self._stop.is_set():
                    self._cv.wait(timeout=1.0)
                if self._stop.is_set():
                    return
                _, _, fut, fn, args = heapq.heappop(self._jobs)
            if not fut.set_running_or_notify_cancel():
                continue
            self.busy = True
            try:
                fut.set_result(fn(*args))
                self.online = True
            except Exception as e:  # noqa: BLE001  (a failed VLM call must not kill the worker)
                self.online = self.client.healthy()
                log.warning("VLM job failed: %s: %s", type(e).__name__, e)
                fut.set_exception(e)
            finally:
                self.busy = False

    # ---------------------------------------------------------------- context
    @property
    def isolate(self):
        return self.policy is not None and self.policy.cfg.get("isolate_untrusted", True)

    def context(self, untrusted=True):
        """Measurements + history (+ the latest VLM scene reports unless untrusted=False)."""
        snap = self.an.snapshot(recent_events=8)
        if not untrusted:
            snap["recent_events"] = [strip_untrusted(e) for e in snap.get("recent_events", [])]
        hist = self.store.series_summary(10)
        obs = self.store.observations(2) if untrusted else []
        lines = [f"Measurements at t={snap['t']} s (JSON): {json.dumps(snap, separators=(',', ':'))}",
                 f"Last 10 minutes: {json.dumps(hist, separators=(',', ':'))}"]
        if obs:
            lines.append("Previous scene reports: " + " | ".join(o["text"] for o in obs))
        return "\n".join(lines)

    def _image_msg(self, text, frame):
        content = [{"type": "text", "text": text}]
        if frame is not None:
            content.insert(0, {"type": "image_url", "image_url": {"url": encode_image(frame, self.cfg["image_max_side"])}})
        return {"role": "user", "content": content}

    def _frame(self):
        raw, ann, ts = self.frames()
        return (ann if self.cfg.get("image", "annotated") == "annotated" else raw), ts

    # ---------------------------------------------------------------- tools
    def run_tool(self, name, args):
        """Execute one tool call; -> JSON-able result. Every change is validated and logged as an event."""
        if name == "get_history":
            m = max(1, min(1440, float(args.get("minutes", 10))))
            return {"series": self.store.series_summary(m), "event_counts": self.store.event_counts(m)}
        if name == "get_recent_events":
            m = max(1, min(1440, float(args.get("minutes", 10))))
            kind = args.get("kind", "any")
            ev = self.store.recent_events(m, None if kind in (None, "any") else [kind], limit=30)
            ev = [strip_untrusted(e) for e in ev] if self.isolate else ev
            return [{k: v for k, v in e.items() if k != "wall"} for e in ev]
        if name == "set_alert":
            rule = {"metric": args["metric"], "op": args["op"], "value": float(args["value"]),
                    "for_s": float(args.get("for_s", 3)), "cooldown_s": 60,
                    "name": args.get("name") or f"{args['metric']} {args['op']} {float(args['value']):g}"}
            r = self.an.add_rule(rule)
            self._config_event("set_alert", r)
            return {"ok": True, "rule": r}
        if name == "remove_alert":
            n = self.an.remove_rule(None if args.get("name") in ("all", "*") else args.get("name"))
            self._config_event("remove_alert", {"name": args.get("name"), "removed": n})
            return {"ok": n > 0, "removed": n}
        if name == "set_counting_line":
            pts = [[float(args["x1"]), float(args["y1"])], [float(args["x2"]), float(args["y2"])]]
            if not all(0 <= v <= 1 for p in pts for v in p) or pts[0] == pts[1]:
                return {"ok": False, "error": "coordinates must be two different points in 0-1"}
            self.an.set_line(pts)
            self._config_event("set_counting_line", {"line": pts})
            (x1, y1), (x2, y2) = pts
            shape = "horizontal" if abs(y2 - y1) < 1e-6 else "vertical" if abs(x2 - x1) < 1e-6 else "diagonal"
            return {"ok": True, "line": pts, "orientation": shape,
                    "counts": self.an.direction_labels}
        return {"ok": False, "error": f"unknown tool {name}"}

    def execute(self, name, args, origin, question="", who=None):
        """Run one tool call through the policy layer (if any). -> result dict for the model / caller."""
        if self.policy is None:
            err = intent_error(question, name, args) if origin == "model" else None
            return {"ok": False, "error": err} if err else self.run_tool(name, args)
        from .policy import principal
        who = who or principal()
        err = intent_error(question, name, args) if origin == "model" else None
        if err:
            self.policy.audit(who, origin, name, args, "deny", "intent check: " + err)
            return {"ok": False, "error": err}
        d = self.policy.check(who, name, args, origin, question, self.an.rules_public())
        if d["decision"] == "deny":
            return {"ok": False, "error": d["reason"], "policy": "denied"}
        if d["decision"] == "confirm":
            return {"ok": False, "error": f"needs confirmation by the operator (/confirm {d['id']})",
                    "policy": "confirm", "confirmation": d["id"]}
        return self.run_tool(name, d["args"])

    def _config_event(self, what, data):
        e = {"ts": self.an.ts, "wall": time.time(), "kind": "config", "change": what, **data}
        self.store.event(e)
        self.an.events.append(e)

    # ---------------------------------------------------------------- jobs
    def _complete(self, messages, tools=None, max_tokens=None):
        msg, info = self.client.chat(messages, tools=tools, max_tokens=max_tokens or self.cfg["max_tokens"],
                                     temperature=self.cfg["temperature"])
        self.last_latency = info["latency_s"]
        return msg, info

    def observe(self, trigger="periodic"):
        frame, ts = self._frame()
        if frame is None:
            return None
        t = time.time()
        msg, info = self._complete([{"role": "system", "content": SYSTEM + "\n\n" + self.context()},
                                    self._image_msg(OBSERVE, frame)], max_tokens=256)
        text = msg.get("content") or ""
        data = parse_json(text) or {"summary": text.strip()[:500]}
        summary = str(data.get("summary", "")).strip() or text.strip()[:300]
        self.store.observation(ts, trigger, summary, data, time.time() - t)
        if str(data.get("risk", "")).lower() in ("medium", "high"):
            e = {"ts": ts, "wall": time.time(), "kind": "vlm_risk", "risk": data["risk"], "summary": summary,
                 "anomalies": data.get("anomalies", [])}
            self.store.event(e); self.an.events.append(e)
        return {"summary": summary, **data, "latency_s": info["latency_s"]}

    def verify(self, alert):
        if self.cfg.get("verify_mode", "describe") == "describe":
            return self._describe_alert(alert)
        frame, ts = self._frame()
        if frame is None:
            return None
        fields = {k: alert.get(k) for k in ("rule", "metric", "op", "threshold", "value")}
        if self.cfg.get("verify_mode", "blind") == "blind":
            fields["metric"] = METRIC_HELP.get(fields["metric"], fields["metric"])
            msgs = [{"role": "system", "content": SYSTEM}, self._image_msg(VERIFY_BLIND.format(**fields), frame)]
        else:
            msgs = [{"role": "system", "content": SYSTEM + "\n\n" + self.context()},
                    self._image_msg(VERIFY.format(**fields), frame)]
        msg, info = self._complete(msgs, max_tokens=128)
        d = parse_json(msg.get("content") or "") or {}
        res = {"rule": alert.get("rule"), "confirmed": bool(d.get("confirmed", False)),
               "reason": str(d.get("reason", msg.get("content", "")))[:300], "latency_s": info["latency_s"]}
        e = {"ts": ts, "wall": time.time(), "kind": "alert_verified", **res}
        self.store.event(e); self.an.events.append(e)
        self.store.observation(ts, f"alert:{alert.get('rule')}",
                               f"{'Confirmed' if res['confirmed'] else 'Not confirmed'}: {res['reason']}", res,
                               info["latency_s"])
        return res

    def _describe_alert(self, alert):
        rep = self.observe(trigger=f"alert:{alert.get('rule')}")
        if rep is None:
            return None
        e = {"ts": self.an.ts, "wall": time.time(), "kind": "alert_context", "rule": alert.get("rule"),
             "summary": rep["summary"], "anomalies": rep.get("anomalies", []), "latency_s": rep["latency_s"]}
        self.store.event(e); self.an.events.append(e)
        return {"rule": alert.get("rule"), "summary": rep["summary"], "latency_s": rep["latency_s"]}

    def ask(self, question, who=None):
        """Chat turn. Stage 1: text-only tool routing (<= 3 tool rounds). Stage 2: answer with the live frame,
        no tools. who: policy principal of the requester. -> {"answer", "actions", "latency_s", "tokens"}."""
        t0, tokens, actions = time.time(), 0, []
        hist, hist_route = [], []
        n = self.cfg["history_turns"]
        for (q, a), f in zip(self.history[-n:], self.footers[-n:]):
            hist += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
            # isolated routing sees only the system-written footer of earlier answers, never the VLM's words
            hist_route += [{"role": "user", "content": q},
                           {"role": "assistant", "content": (f or "(answered)") if self.isolate else a}]
        hist_answer = hist[-4:]          # stage 2: last 2 turns only (a 2B model re-answers older questions)
        if self.cfg.get("use_tools", True):
            msgs = [{"role": "system", "content": SYSTEM + "\n\n" + self.context(untrusted=not self.isolate)},
                    *hist_route, {"role": "user", "content": question}]
            for _ in range(3):
                msg, info = self._complete(msgs, tools=TOOLS)
                tokens += info.get("completion_tokens") or 0
                calls = msg.get("tool_calls") or text_tool_calls(msg.get("content"))
                if not calls:
                    break
                msgs.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": [
                    {"id": c.get("id", f"c{i}"), "type": "function", "function": c["function"]}
                    for i, c in enumerate(calls)]})
                for i, c in enumerate(calls):
                    name = c["function"]["name"]
                    try:
                        args = json.loads(c["function"].get("arguments") or "{}")
                        result = self.execute(name, args, "model", question, who)
                    except Exception as e:  # noqa: BLE001  (bad arguments from the model -> tell it)
                        args, result = c["function"].get("arguments"), {"ok": False, "error": f"{type(e).__name__}: {e}"}
                    actions.append({"tool": name, "args": args, "result": result, "source": "model"})
                    msgs.append({"role": "tool", "tool_call_id": c.get("id", f"c{i}"), "name": name,
                                 "content": json.dumps(result, default=str)})
        if self.cfg.get("use_tools", True):                    # deterministic safety net for clear commands
            for name, args in fallback_intents(question):
                if not any(a["tool"] == name and (_ok(a["result"]) or a["result"].get("policy")) for a in actions):
                    actions.append({"tool": name, "args": args, "result": self.execute(name, args, "fallback",
                                                                                       question, who),
                                    "source": "fallback"})
        frame, ts = self._frame()
        q = question
        if actions:
            done = [a for a in actions if _ok(a["result"])]
            failed = [a for a in actions if a not in done]
            calls = json.dumps([{"tool": a["tool"], "args": a["args"], "result": a["result"]} for a in actions],
                               default=str)[:3000]
            q += (f"\n\n(Tool calls for this request: {calls}. "
                  f"Succeeded: {', '.join(a['tool'] for a in done) or 'none'}. "
                  f"Failed: {', '.join(a['tool'] for a in failed) or 'none'}. "
                  "Tell the user briefly and truthfully what was changed and what failed, then answer the question. "
                  "Do not claim any change that is not listed as succeeded."
                  + (" For a failed change, the user can type a command instead, e.g. /line 0 0.5 1 0.5, "
                     "/alert occupancy > 10, /unalert all." if failed else "") + ")")
        elif self.cfg.get("use_tools", True):
            q += ("\n\n(No tool calls were executed for this request, so no setting was changed. If the user asked "
                  "for a change, say it was NOT done and give the exact command, e.g. /unalert all, "
                  "/alert occupancy > 10, /line 0 0.5 1 0.5.)")
        msg, info = self._complete([{"role": "system", "content": SYSTEM + "\n\n" + self.context()}, *hist_answer,
                                    self._image_msg(q, frame)])
        tokens += info.get("completion_tokens") or 0
        answer = re.sub(r"<tool_call>.*?</tool_call>", "", msg.get("content") or "", flags=re.S).strip()
        if not answer and actions:
            answer = "Done: " + "; ".join(f"{a['tool']} -> {json.dumps(a['result'], default=str)[:120]}" for a in actions)
        # system-generated, always truthful summary of state changes (the model's own wording may not be)
        model_answer = answer                  # kept in the chat history without the system footer (else copied)
        changes = [a for a in actions if a["tool"] in ("set_alert", "remove_alert", "set_counting_line")]
        footer = ""
        if changes or CHANGE_WORDS.search(question):
            ok = [_describe_action(a) for a in changes if _ok(a["result"])]
            wait = [f"{a['tool']} {json.dumps(a['args'])} awaits /confirm {a['result']['confirmation']}"
                    for a in changes if a["result"].get("policy") == "confirm"]
            bad = [f"{a['tool']} failed ({a['result'].get('error', '')[:80]})" for a in changes
                   if not _ok(a["result"]) and a["result"].get("policy") != "confirm"]
            footer = ("Changed: " + "; ".join(ok) + "." if ok else "No settings were changed.") + \
                     (" Pending: " + "; ".join(wait) + "." if wait else "") + \
                     (" Failed: " + "; ".join(bad) + "." if bad else "")
            answer = f"{answer}\n\n[system] {footer}" if answer else f"[system] {footer}"
        self.history.append((question, model_answer or answer))
        self.footers.append(footer)
        self.store.chat("user", question)
        self.store.chat("assistant", answer, {"actions": actions})
        return {"answer": answer, "actions": actions, "latency_s": round(time.time() - t0, 2), "tokens": tokens, "t": ts}
