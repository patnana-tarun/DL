"""Deterministic chat commands (start with "/"): they work even when the VLM is offline and never depend
on the model following instructions. Anything else goes to the VLM.

  /help                                   this list
  /stats                                  live measurements
  /history [minutes]                      occupancy summary + event counts
  /events [minutes]                       recent events
  /alerts                                 list alert rules
  /alert <metric> <op> <value> [for <s>]  add a rule, e.g. /alert occupancy > 10 for 5
  /unalert <name|all>                     remove rule(s)
  /line x1 y1 x2 y2                       move the counting line (normalised 0-1)
  /observe                                ask the VLM for a scene report now
  /pending                                changes from chat that wait for your confirmation
  /confirm <id> | /cancel <id>            apply or drop such a change
  /audit [n]                              last audit-log entries and whether the hash chain verifies
"""
import json

from .analytics import METRICS


def handle(text, pipeline, who=None):
    """-> reply dict, or None if `text` is not a command. who: policy principal (None = local operator)."""
    if not text.startswith("/"):
        return None
    parts = text[1:].split()
    cmd, args = (parts[0].lower(), parts[1:]) if parts else ("help", [])
    an, store, agent = pipeline.analytics, pipeline.store, pipeline.agent
    policy = getattr(pipeline, "policy", None)
    if policy is not None and who is None:
        from .policy import principal
        who = principal()
    try:
        if cmd == "help":
            return {"answer": __doc__.split("\n\n", 1)[1].strip()}
        if cmd == "stats":
            return {"answer": _fmt(an.snapshot(recent_events=5))}
        if cmd == "history":
            m = float(args[0]) if args else 10
            return {"answer": _fmt({"series": store.series_summary(m), "event_counts": store.event_counts(m)})}
        if cmd == "events":
            m = float(args[0]) if args else 10
            ev = [{k: v for k, v in e.items() if k != "wall"} for e in store.recent_events(m, limit=20)]
            return {"answer": _fmt(ev) if ev else "no events"}
        if cmd == "alerts":
            r = an.rules_public()
            return {"answer": _fmt(r) if r else "no alert rules"}
        if cmd == "alert":
            if len(args) < 3:
                return {"answer": f"usage: /alert <metric> <op> <value> [for <s>]; metrics: {', '.join(METRICS)}"}
            rule = {"metric": args[0], "op": args[1], "value": float(args[2]),
                    "for_s": float(args[4]) if len(args) >= 5 and args[3] == "for" else 3.0}
            return _change(pipeline, who, "set_alert", rule)
        if cmd == "unalert":
            if not args:
                return {"answer": "usage: /unalert <rule name> | all   (see /alerts for names)"}
            return _change(pipeline, who, "remove_alert", {"name": "all" if args[0] in ("all", "*") else " ".join(args)})
        if cmd == "line":
            x1, y1, x2, y2 = map(float, args[:4])
            return _change(pipeline, who, "set_counting_line", {"x1": x1, "y1": y1, "x2": x2, "y2": y2})
        if cmd == "observe":
            if agent is None:
                return {"answer": "VLM is disabled in the config"}
            r = agent.submit(0, agent.observe, "user").result(timeout=agent.cfg["timeout_s"] + 5)
            return {"answer": (r or {}).get("summary", "no frame yet"), "report": r}
        if cmd in ("pending", "confirm", "cancel", "audit") and policy is None:
            return {"answer": "the policy layer is disabled (policy.enabled: false)"}
        if cmd == "pending":
            p = policy.pending(who)
            return {"answer": _fmt(p) if p else "nothing waits for confirmation"}
        if cmd == "confirm":
            tool, a = policy.confirm(who, args[0])
            res = _run(pipeline, tool, a)
            return {"answer": f"confirmed: {tool} {json.dumps(a)} -> {json.dumps(res, default=str)[:200]}",
                    "actions": [{"tool": tool, "args": a, "result": res, "source": "confirm"}]}
        if cmd == "cancel":
            return {"answer": "cancelled" if policy.cancel(who, args[0]) else "unknown confirmation id"}
        if cmd == "audit":
            n = int(args[0]) if args else 10
            return {"answer": _fmt({"verify": store.verify_audit(), "last": store.audit_log(n)})}
    except (ValueError, IndexError) as e:
        return {"answer": f"error: {e}"}
    return {"answer": f"unknown command /{cmd}; try /help"}


def _run(pipeline, tool, args):
    """Apply a change that is already allowed (validated args) and log it as a config event."""
    an, store = pipeline.analytics, pipeline.store
    if tool == "set_alert":
        r = an.add_rule(args)
        store.event({"ts": an.ts, "kind": "config", "change": "set_alert", **r})
        return {"ok": True, "rule": r}
    if tool == "remove_alert":
        n = an.remove_rule(None if args["name"] in ("all", "*") else args["name"])
        store.event({"ts": an.ts, "kind": "config", "change": "remove_alert", "name": args["name"], "removed": n})
        return {"ok": n > 0, "removed": n}
    if tool == "set_counting_line":
        line = [[args["x1"], args["y1"]], [args["x2"], args["y2"]]]
        an.set_line(line)
        store.event({"ts": an.ts, "kind": "config", "change": "set_counting_line", "line": line})
        return {"ok": True, "line": line}
    raise ValueError(f"unknown tool {tool}")


def _change(pipeline, who, tool, args):
    policy = getattr(pipeline, "policy", None)
    if policy is not None:
        d = policy.check(who, tool, args, "command", rules=pipeline.analytics.rules_public())
        if d["decision"] != "allow":
            return {"answer": f"refused: {d['reason']}", "policy": d["decision"]}
        args = d["args"]
    res = _run(pipeline, tool, args)
    if tool == "set_alert":
        return {"answer": f"alert added: {res['rule']['name']}", "actions": [{"tool": tool, "result": res["rule"]}]}
    if tool == "remove_alert":
        return {"answer": f"removed {res['removed']} rule(s)"}
    (x1, y1), (x2, y2) = res["line"]
    return {"answer": f"counting line set to ({x1},{y1})-({x2},{y2})"}


def _fmt(x):
    return json.dumps(x, indent=1, default=str)
