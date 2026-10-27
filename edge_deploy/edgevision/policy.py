"""Trusted policy layer between every requester (VLM tool calls, the rule-based fallback, "/" commands) and the
analytics state. The VLM proposes, the policy disposes: no tool runs unless this module allows it, and the
decision never depends on what the model says about itself.

Order of checks for one tool call (first failure wins; every decision is written to the audit log):
  1. known tool                    unknown names are denied
  2. role (RBAC)                   viewer: read tools; operator: changes; admin: also remove ALL alerts
  3. argument validation           types, enums, finite numbers, per-metric bounds, line geometry
  4. intent gate (model calls)     a state change proposed by the model must be asked for in the user's own
                                   words of this turn (change verb + object), so text that reached the model
                                   from a frame, a scene report or a tool result cannot trigger one
  5. rate limit                    token bucket per principal on state changes
  6. confirmation                  weakening or removing alerts proposed in chat (model or fallback) waits for
                                   "/confirm <id>" from the same principal; explicit "/" commands are already
                                   explicit and run directly
Requests themselves are rate-limited per principal (chat_per_min) and refused when the VLM queue is full.

Audit log: append-only SQLite table whose rows are hash-chained (hash = SHA-256(prev_hash || canonical JSON of
the row)); Store.verify_audit() recomputes the chain, so an edited or deleted row is detected.
"""
import itertools
import math
import re
import threading
import time

ROLES = {"viewer": 0, "operator": 1, "admin": 2}
READ_TOOLS = {"get_history", "get_recent_events"}
STATE_TOOLS = {"set_alert", "remove_alert", "set_counting_line"}
TOOL_ROLE = {**{t: "viewer" for t in READ_TOOLS}, **{t: "operator" for t in STATE_TOOLS}}
EVENT_KINDS = {"alert", "alert_verified", "alert_context", "line_cross", "loitering", "person_enter", "person_exit",
               "zone_enter", "zone_exit", "config", "vlm_risk", "any"}
# plausible ranges: a threshold outside them is a disabled rule in disguise ("alert above a million people")
BOUNDS = {"occupancy": (0, 500), "unique_people": (0, 100000), "crossings_total": (0, 1000000),
          "crossings_a_to_b": (0, 1000000), "crossings_b_to_a": (0, 1000000), "crossings_last_min": (0, 1000),
          "max_dwell_s": (1, 86400), "loitering": (0, 500)}
OPS = (">", ">=", "<", "<=")
MAX_RULES = 20
MIN_LINE_LEN = 0.1                     # normalised length; a dot-sized line counts nobody

CHANGE_VERB = re.compile(r"\b(alert|alarm|notify|warn|tell me|let me know|ping|set|add|create|change|move|make|put|"
                         r"draw|place|remove|delete|clear|drop|disable|turn off|stop|cancel|raise|lower|increase|"
                         r"decrease|replace|update|adjust)\b", re.I)
OBJECT_WORDS = {"set_alert": re.compile(r"\b(alerts?|alarms?|rules?|notif\w*|warn\w*|tell me|let me know|ping|"
                                        r"thresholds?)\b", re.I),
                "remove_alert": re.compile(r"\b(alerts?|alarms?|rules?|notif\w*|warn\w*|thresholds?)\b", re.I),
                "set_counting_line": re.compile(r"\b(line|tripwire|counter)\b", re.I)}


class Decision(dict):
    """{"decision": allow|deny|confirm, "reason": str, "args": normalised args, "id": confirmation id}"""

    @property
    def allowed(self):
        return self["decision"] == "allow"


def principal(name="local", role="operator"):
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}")
    return {"name": name, "role": role}


def _num(v, what):
    if isinstance(v, bool):
        raise ValueError(f"{what} must be a number")
    x = float(v)
    if not math.isfinite(x):
        raise ValueError(f"{what} must be finite")
    return x


def validate(tool, args, rules=()):
    """-> normalised args, or raises ValueError with a message meant for the requester."""
    if not isinstance(args, dict):
        raise ValueError("arguments must be a JSON object")
    if tool in ("get_history", "get_recent_events"):
        m = _num(args.get("minutes", 10), "minutes")
        out = {"minutes": min(1440.0, max(1.0, m))}
        if tool == "get_recent_events":
            kind = args.get("kind", "any") or "any"
            if kind not in EVENT_KINDS:
                raise ValueError(f"unknown event kind {kind!r}")
            out["kind"] = kind
        return out
    if tool == "set_alert":
        metric, op = args.get("metric"), args.get("op")
        if metric not in BOUNDS and not (isinstance(metric, str) and re.fullmatch(r"zone:\w{1,32}", metric)):
            raise ValueError(f"unknown metric {metric!r}")
        if op not in OPS:
            raise ValueError(f"operator must be one of {' '.join(OPS)}")
        value = _num(args.get("value"), "value")
        lo, hi = BOUNDS.get(metric, (0, 500))
        if not lo <= value <= hi:
            raise ValueError(f"{metric} threshold must be within [{lo:g}, {hi:g}]")
        for_s = _num(args.get("for_s", 3), "for_s")
        if not 0 <= for_s <= 600:
            raise ValueError("for_s must be within [0, 600] s")
        name = str(args.get("name") or f"{metric} {op} {value:g}")[:64]
        if len({r["name"] for r in rules} | {name}) > MAX_RULES:
            raise ValueError(f"at most {MAX_RULES} alert rules")
        return {"metric": metric, "op": op, "value": value, "for_s": for_s, "name": name}
    if tool == "remove_alert":
        name = args.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a rule name or 'all'")
        name = name.strip()
        if name not in ("all", "*") and name not in {r["name"] for r in rules}:
            raise ValueError(f"no alert rule named {name!r}")
        return {"name": "all" if name == "*" else name}
    if tool == "set_counting_line":
        try:
            x1, y1, x2, y2 = (_num(args[k], k) for k in ("x1", "y1", "x2", "y2"))
        except KeyError as e:
            raise ValueError(f"missing coordinate {e}") from None
        if not all(0 <= v <= 1 for v in (x1, y1, x2, y2)):
            raise ValueError("coordinates must be within 0-1")
        if math.hypot(x2 - x1, y2 - y1) < MIN_LINE_LEN:
            raise ValueError(f"the line must be at least {MIN_LINE_LEN} of the image long")
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    raise ValueError(f"unknown tool {tool!r}")


def requested_in(text, tool):
    """Intent gate: does the user's own text of this turn ask for this kind of change?"""
    t = text or ""
    return bool(CHANGE_VERB.search(t)) and bool(OBJECT_WORDS[tool].search(t))


def weakens(tool, args, rules):
    """Does this change remove or weaken an existing alert? (needs confirmation when proposed in chat)"""
    if tool == "remove_alert":
        return True
    if tool == "set_alert":
        old = next((r for r in rules if r["name"] == args["name"]), None)
        if old is None:
            return False
        if old["metric"] != args["metric"] or old["op"] != args["op"]:
            return True
        up = args["op"] in (">", ">=")              # for '>' a higher threshold fires less often
        looser = args["value"] > old["value"] if up else args["value"] < old["value"]
        return looser or args["for_s"] > old.get("for_s", 0)
    return False


class _Bucket:
    def __init__(self, per_min, burst, now):
        self.rate, self.cap = per_min / 60.0, float(burst)
        self.tokens, self.t = float(burst), now

    def take(self, now):
        self.tokens = min(self.cap, self.tokens + (now - self.t) * self.rate); self.t = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


class Policy:
    def __init__(self, cfg, store, clock=time.monotonic):
        """cfg = config["policy"]; store gets the audit rows."""
        self.cfg, self.store, self.clock = cfg, store, clock
        self._lock = threading.Lock()
        self._buckets = {}
        self._pending = {}
        self._ids = itertools.count(1)

    # ---------------------------------------------------------------- rate limits
    def _bucket(self, kind, who):
        per_min, burst = self.cfg[f"{kind}_per_min"], self.cfg[f"{kind}_burst"]
        if (kind, who) not in self._buckets:
            self._buckets[(kind, who)] = _Bucket(per_min, burst, self.clock())
        return self._buckets[(kind, who)]

    def allow_request(self, who, queue_len=0):
        """Chat request admission: -> (ok, reason)."""
        if queue_len >= self.cfg["max_queue"]:
            self.audit(who, "request", "chat", {}, "deny", "VLM queue full")
            return False, "the assistant is busy; try again shortly (commands still work: /help)"
        with self._lock:
            ok = self._bucket("chat", who["name"]).take(self.clock())
        if not ok:
            self.audit(who, "request", "chat", {}, "deny", "chat rate limit")
            return False, "too many requests; wait a minute (commands still work: /help)"
        return True, ""

    # ---------------------------------------------------------------- decisions
    def check(self, who, tool, args, origin, request_text="", rules=()):
        """origin: model | fallback | command. -> Decision (audited)."""
        d = self._decide(who, tool, args, origin, request_text, rules)
        self.audit(who, origin, tool, d.get("args", args), d["decision"], d["reason"], d.get("id"))
        return d

    def _decide(self, who, tool, args, origin, text, rules):
        if tool not in TOOL_ROLE:
            return Decision(decision="deny", reason=f"unknown tool {tool!r}")
        need = TOOL_ROLE[tool]
        try:
            a = validate(tool, args, rules)
        except (ValueError, TypeError) as e:
            return Decision(decision="deny", reason=f"invalid arguments: {e}")
        if tool == "remove_alert" and a["name"] == "all":
            need = "admin" if self.cfg.get("remove_all_admin_only", True) else "operator"
        if ROLES[who["role"]] < ROLES[need]:
            return Decision(decision="deny", reason=f"not authorised: {tool} needs role {need}, "
                                                    f"{who['name']} is {who['role']}", args=a)
        if tool in READ_TOOLS:
            return Decision(decision="allow", reason="read-only", args=a)
        if origin == "model" and self.cfg.get("intent_gate", True) and not requested_in(text, tool):
            return Decision(decision="deny", reason="change not requested in the user's message", args=a)
        with self._lock:
            ok = self._bucket("change", who["name"]).take(self.clock())
        if not ok:
            return Decision(decision="deny", reason="rate limit for configuration changes", args=a)
        if origin != "command" and self.cfg.get("confirm_weakening", True) and weakens(tool, a, rules):
            with self._lock:
                cid = f"c{next(self._ids)}"
                self._pending[cid] = {"who": who, "tool": tool, "args": a, "expires": self.clock() + self.cfg["confirm_ttl_s"]}
            return Decision(decision="confirm", reason=f"weakens or removes an alert: type /confirm {cid} to apply",
                            args=a, id=cid)
        return Decision(decision="allow", reason="ok", args=a)

    def confirm(self, who, cid):
        """-> (tool, args) to execute, or raises ValueError. Only the requesting principal may confirm."""
        with self._lock:
            p = self._pending.get(cid)
            if p is None or p["expires"] < self.clock():
                self._pending.pop(cid, None)
                err = "unknown or expired confirmation id"
            elif p["who"]["name"] != who["name"]:
                err = "only the operator who made the request can confirm it"
            else:
                del self._pending[cid]; err = None
        if err:
            self.audit(who, "command", "confirm", {"id": cid}, "deny", err)
            raise ValueError(err)
        self.audit(who, "command", p["tool"], p["args"], "allow", f"confirmed {cid}", cid)
        return p["tool"], p["args"]

    def cancel(self, who, cid):
        with self._lock:
            p = self._pending.get(cid)
            ok = p is not None and p["who"]["name"] == who["name"]
            if ok:
                del self._pending[cid]
        self.audit(who, "command", "cancel", {"id": cid}, "allow" if ok else "deny", "cancelled" if ok else "unknown id")
        return ok

    def pending(self, who=None):
        with self._lock:
            return {k: {"tool": v["tool"], "args": v["args"]} for k, v in self._pending.items()
                    if who is None or v["who"]["name"] == who["name"]}

    def audit(self, who, origin, tool, args, decision, reason, cid=None):
        if self.store is not None:
            self.store.audit({"principal": who["name"], "role": who["role"], "origin": origin, "tool": tool,
                              "args": args, "decision": decision, "reason": reason, "confirmation": cid})


DEFAULTS = {
    "enabled": True,
    "intent_gate": True,             # model-proposed changes must be asked for in the user's words
    "confirm_weakening": True,       # removing / weakening alerts from chat waits for /confirm
    "remove_all_admin_only": True,
    "isolate_untrusted": True,       # stage-1 routing never sees VLM-written text (scene reports, answers)
    "change_per_min": 6, "change_burst": 4,
    "chat_per_min": 20, "chat_burst": 6,
    "max_queue": 8,                  # refuse chat when this many VLM jobs are waiting
    "confirm_ttl_s": 120,
}
