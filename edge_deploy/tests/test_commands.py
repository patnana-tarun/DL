import copy
import json
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest

from edgevision import commands
from edgevision.analytics import Analytics
from edgevision.config import DEFAULTS
from edgevision.store import Store


class StubAgent:
    def __init__(self, report=None):
        self.cfg = {"timeout_s": 2}
        self.report = report
        self.submitted = []

    def observe(self, trigger="periodic"):
        self.trigger = trigger
        return self.report

    def submit(self, priority, fn, *args):
        self.submitted.append((priority, fn.__name__, args))
        f = Future()
        f.set_result(fn(*args))
        return f


@pytest.fixture
def p():
    an = Analytics(copy.deepcopy(DEFAULTS["analytics"]), DEFAULTS["alerts"], 1000, 500, 10)
    return SimpleNamespace(analytics=an, store=Store(":memory:"), agent=None)


def run(p, text):
    r = commands.handle(text, p)
    assert r is None or isinstance(r["answer"], str)
    return r


def test_non_command_returns_none(p):
    assert run(p, "how many people?") is None
    assert run(p, " /stats") is None                     # server.chat strips first; handle needs a leading /


@pytest.mark.parametrize("text", ["/help", "/", "/HELP"])
def test_help(p, text):
    a = run(p, text)["answer"]
    for c in ("/stats", "/history", "/events", "/alerts", "/alert <metric>", "/unalert", "/line", "/observe"):
        assert c in a
    assert not a.startswith("Deterministic")             # the module intro is not part of the help


def test_stats(p):
    for i in range(5):
        p.analytics.update(i / 10, np.array([[100, 100, 140, 200], [600, 100, 640, 200]], np.float32),
                           np.array([1, 2]))
    d = json.loads(run(p, "/stats")["answer"])
    assert d["occupancy"] == 2 and d["unique_people"] == 2 and len(d["recent_events"]) <= 5
    assert json.loads(run(p, "/STATS")["answer"])["occupancy"] == 2


def test_history(p):
    p.store.series({"t": 0, "occ_mean": 2.0, "occ_max": 3, "unique": 1, "crossings": 0})
    p.store.event({"ts": 0, "kind": "alert"})
    d = json.loads(run(p, "/history")["answer"])
    assert d["series"]["minutes"] == 10 and d["series"]["samples"] == 1 and d["event_counts"] == {"alert": 1}
    assert json.loads(run(p, "/history 30")["answer"])["series"]["minutes"] == 30.0
    assert run(p, "/history abc")["answer"].startswith("error:")


def test_events(p):
    assert run(p, "/events")["answer"] == "no events"
    for i in range(25):
        p.store.event({"ts": i, "kind": "line_cross", "i": i})
    ev = json.loads(run(p, "/events 5")["answer"])
    assert len(ev) == 20 and ev[-1]["i"] == 24 and all("wall" not in e for e in ev)


def test_alerts_list(p):
    rules = json.loads(run(p, "/alerts")["answer"])
    assert [r["name"] for r in rules] == ["crowding"]
    p.analytics.remove_rule()
    assert run(p, "/alerts")["answer"] == "no alert rules"


def test_alert_add(p):
    r = run(p, "/alert occupancy > 10 for 5")
    assert r["answer"] == "alert added: occupancy > 10"
    assert r["actions"][0]["tool"] == "set_alert" and r["actions"][0]["result"]["for_s"] == 5.0
    rule = p.analytics.rules_public()[-1]
    assert rule == {"name": "occupancy > 10", "metric": "occupancy", "op": ">", "value": 10.0, "for_s": 5.0,
                    "cooldown_s": 60.0, "active": False}
    ev = p.store.recent_events(kinds=["config"])
    assert ev[0]["change"] == "set_alert" and ev[0]["name"] == "occupancy > 10"
    assert run(p, "/alert unique_people >= 50")["actions"][0]["result"]["for_s"] == 3.0     # default for_s
    assert run(p, "/alert zone:door > 0")["answer"] == "alert added: zone:door > 0"


@pytest.mark.parametrize("text, expect", [
    ("/alert", "usage: /alert"), ("/alert occupancy >", "usage: /alert"),
    ("/alert temperature > 3", "error: unknown metric"), ("/alert occupancy != 3", "error: unknown operator"),
    ("/alert occupancy > lots", "error:"), ("/alert occupancy > 3 for", "alert added"),
    ("/alert occupancy > 3 for x", "error:"),
])
def test_alert_bad_input(p, text, expect):
    assert run(p, text)["answer"].startswith(expect)


def test_unalert(p):
    run(p, "/alert occupancy > 10")
    run(p, "/alert occupancy > 20")
    assert run(p, "/unalert occupancy > 10")["answer"] == "removed 1 rule(s)"     # names contain spaces
    assert run(p, "/unalert nope")["answer"] == "removed 0 rule(s)"
    assert [r["name"] for r in p.analytics.rules_public()] == ["crowding", "occupancy > 20"]
    assert run(p, "/unalert all")["answer"] == "removed 2 rule(s)"
    run(p, "/alert occupancy > 1")
    assert run(p, "/unalert")["answer"].startswith("usage:")                  # no silent clear-all
    assert [r["name"] for r in p.analytics.rules_public()] == ["occupancy > 1"]
    run(p, "/alert occupancy > 1")
    assert run(p, "/unalert *")["answer"] == "removed 1 rule(s)"


def test_line(p):
    r = run(p, "/line 0 0.5 1 0.5")
    assert r["answer"] == "counting line set to (0.0,0.5)-(1.0,0.5)"
    assert p.analytics.line_norm == [[0.0, 0.5], [1.0, 0.5]]
    assert p.analytics.direction_labels["a_to_b"] == "top to bottom"
    ev = p.store.recent_events(kinds=["config"])
    assert ev[0]["change"] == "set_counting_line" and ev[0]["line"] == [[0.0, 0.5], [1.0, 0.5]]


@pytest.mark.parametrize("text", ["/line", "/line 0 0.5 1", "/line a b c d", "/line 0.5 0.5 0.5 0.5"])
def test_line_bad_input(p, text):
    assert run(p, text)["answer"].startswith("error:")
    assert p.analytics.line_norm == [[0.5, 0.0], [0.5, 1.0]]
    assert p.store.recent_events(kinds=["config"]) == []


def test_observe_without_agent(p):
    assert run(p, "/observe")["answer"] == "VLM is disabled in the config"


def test_observe_with_agent(p):
    p.agent = StubAgent({"summary": "Quiet street.", "risk": "none"})
    r = run(p, "/observe")
    assert r["answer"] == "Quiet street." and r["report"]["risk"] == "none"
    assert p.agent.submitted == [(0, "observe", ("user",))]            # user request: top priority
    p.agent = StubAgent(None)
    assert run(p, "/observe")["answer"] == "no frame yet"


def test_unknown_command(p):
    assert run(p, "/dance now")["answer"] == "unknown command /dance; try /help"
