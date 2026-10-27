import base64
import copy
import json
import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest
from conftest import free_port

from edgevision import vlm
from edgevision.analytics import Analytics
from edgevision.config import DEFAULTS
from edgevision.store import Store
from edgevision.vlm import SceneAgent, encode_image, parse_json, text_tool_calls


# ---------------------------------------------------------------- pure helpers
@pytest.mark.parametrize("text, expected", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"summary": "two people", "risk": "low"}\n```', {"summary": "two people", "risk": "low"}),
    ('```\n{"confirmed": true}\n```', {"confirmed": True}),
    ('Sure! Here is the report: {"summary": "x", "activities": ["walking"]} Hope this helps.',
     {"summary": "x", "activities": ["walking"]}),
    ('{"a": {"b": {"c": 2}}, "d": [1, {"e": 3}]}', {"a": {"b": {"c": 2}}, "d": [1, {"e": 3}]}),
    ('noise {not json} then {"ok": true}', {"ok": True}),
    ('{"first": 1} {"second": 2}', {"first": 1}),
    ('{"sign": "reads {exit}"}', {"sign": "reads {exit}"}),
    ("no json here", None),
    ('{"unterminated": 1', None),
    ("", None),
    (None, None),
])
def test_parse_json(text, expected):
    assert parse_json(text) == expected


def test_text_tool_calls():
    t = ('I will set it. <tool_call>{"name": "set_alert", "arguments": {"metric": "occupancy", "op": ">", '
         '"value": 5, "extra": {"x": 1}}}</tool_call>\n<tool_call>\n{"name": "get_history", "arguments": '
         '{"minutes": 10}}\n</tool_call><tool_call>{broken}</tool_call><tool_call>{"arguments": {}}</tool_call>'
         '<tool_call>{"name": "remove_alert"}</tool_call>')
    calls = text_tool_calls(t)
    assert [c["function"]["name"] for c in calls] == ["set_alert", "get_history", "remove_alert"]
    assert [c["id"] for c in calls] == ["t0", "t1", "t2"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"metric": "occupancy", "op": ">", "value": 5,
                                                             "extra": {"x": 1}}
    assert json.loads(calls[2]["function"]["arguments"]) == {}
    assert text_tool_calls("plain answer") == [] and text_tool_calls(None) == []


def _decode(uri):
    assert uri.startswith("data:image/jpeg;base64,")
    buf = np.frombuffer(base64.b64decode(uri.split(",", 1)[1]), np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


@pytest.mark.parametrize("hw, max_side, out_hw", [((1000, 2000), 896, (448, 896)), ((2000, 1000), 896, (896, 448)),
                                                  ((200, 300), 896, (200, 300)), ((720, 1280), 640, (360, 640))])
def test_encode_image_size_limit(hw, max_side, out_hw):
    img = np.random.default_rng(0).integers(0, 255, (*hw, 3), dtype=np.uint8)
    out = _decode(encode_image(img, max_side=max_side))
    assert out.shape[:2] == out_hw                                   # never upscaled, aspect kept


# ---------------------------------------------------------------- mock OpenAI-compatible server
class MockLLM:
    """Scripted /v1/chat/completions: each request pops the next reply (dict -> message, int -> HTTP error)."""

    def __init__(self):
        self.replies, self.requests, self.models_hits = [], [], 0
        self.delay = 0.0
        owner = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802
                if self.path == "/v1/models":
                    owner.models_hits += 1
                    return self._json(200, {"data": [{"id": "mock"}]})
                return self._json(404, {})

            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append({"path": self.path, "body": body, "auth": self.headers.get("Authorization")})
                time.sleep(owner.delay)
                r = owner.replies.pop(0) if owner.replies else {"content": "default answer"}
                if isinstance(r, int):
                    return self._json(r, {"error": "boom"})
                return self._json(200, {"choices": [{"message": {"role": "assistant", **r}}],
                                        "usage": {"prompt_tokens": 100, "completion_tokens": 7}})

        self.port = free_port()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", self.port), H)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.endpoint = f"http://127.0.0.1:{self.port}/v1"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def tool_call(name, args, cid="call_1"):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


@pytest.fixture
def llm():
    m = MockLLM()
    yield m
    m.close()


@pytest.fixture
def frame():
    img = np.zeros((360, 640, 3), np.uint8)
    cv2.rectangle(img, (100, 100), (160, 300), (0, 255, 0), -1)
    return img


def make_agent(endpoint, frame, **over):
    cfg = copy.deepcopy(DEFAULTS["vlm"])
    cfg.update(endpoint=endpoint, timeout_s=10, **over)
    an = Analytics(copy.deepcopy(DEFAULTS["analytics"]), DEFAULTS["alerts"], 640, 360, 10)
    store = Store(":memory:")
    state = {"frame": frame}
    agent = SceneAgent(cfg, an, store, lambda: (state["frame"], state["frame"], 12.5 if state["frame"] is not None
                                                else None))
    agent.state = state
    return agent


@pytest.fixture
def agent(llm, frame):
    a = make_agent(llm.endpoint, frame).start()
    yield a
    a.stop()


@pytest.fixture
def blind_agent(llm, frame):
    a = make_agent(llm.endpoint, frame, verify_mode="blind").start()
    yield a
    a.stop()


def test_start_checks_health(agent, llm):
    assert agent.online and llm.models_hits == 1


def test_ask_executes_tool_then_answers(agent, llm):
    """Two stages: text-only routing with tools (tool call, then no more calls), then the answer with the frame."""
    llm.replies = [{"content": "", "tool_calls": [tool_call("set_alert", {"metric": "occupancy", "op": ">",
                                                                          "value": 5, "for_s": 2})]},
                   {"content": "routing done"},
                   {"content": "Done - I will alert you when more than 5 people are in view."}]
    r = agent.submit(0, agent.ask, "alert me when more than 5 people are here").result(timeout=20)
    assert r["answer"] == ("Done - I will alert you when more than 5 people are in view.\n\n[system] Changed: "
                           "alert rule 'occupancy > 5' (occupancy > 5 for 2 s).")
    assert len(r["actions"]) == 1
    a = r["actions"][0]
    assert a["tool"] == "set_alert" and a["args"]["value"] == 5 and a["result"]["ok"]
    rule = next(x for x in agent.an.rules_public() if x["name"] == "occupancy > 5")
    assert rule["value"] == 5.0 and rule["for_s"] == 2.0 and rule["op"] == ">"
    assert r["tokens"] == 21 and r["t"] == 12.5
    log = agent.store.chat_log()
    assert [(m["role"], m["text"]) for m in log] == [("user", "alert me when more than 5 people are here"),
                                                    ("assistant", r["answer"])]
    assert agent.store.recent_events(kinds=["config"])[0]["change"] == "set_alert"
    assert any(e.get("change") == "set_alert" for e in agent.an.events)
    first, second, third = (q["body"] for q in llm.requests)
    assert llm.requests[0]["path"] == "/v1/chat/completions"
    # stage 1: system context + the question as plain text (no image), tools offered
    assert first["tools"] == vlm.TOOLS and first["model"] == "qwen3-vl-2b" and first["stream"] is False
    assert first["messages"][0]["role"] == "system" and "Measurements at t=" in first["messages"][0]["content"]
    assert first["messages"][-1] == {"role": "user", "content": "alert me when more than 5 people are here"}
    tool_msg = second["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "call_1" and tool_msg["name"] == "set_alert"
    assert json.loads(tool_msg["content"])["ok"] is True
    assert second["messages"][-2]["tool_calls"][0]["function"]["name"] == "set_alert"
    # stage 2: the frame + the question + what was executed, no tools
    assert "tools" not in third
    user = third["messages"][-1]
    assert user["role"] == "user" and user["content"][0]["type"] == "image_url"
    assert _decode(user["content"][0]["image_url"]["url"]).shape == (360, 640, 3)
    text = user["content"][1]["text"]
    assert text.startswith("alert me when more than 5 people are here") and "set_alert" in text
    assert agent.last_latency is not None and agent.history[-1][0].startswith("alert me")


def test_ask_plain_question_routes_without_tools(agent, llm):
    llm.replies = [{"content": "no tool needed"}, {"content": "Three people are walking left."}]
    r = agent.submit(0, agent.ask, "what are people doing?").result(timeout=20)
    assert r["answer"] == "Three people are walking left." and r["actions"] == []
    assert len(llm.requests) == 2
    assert "tools" in llm.requests[0]["body"] and "tools" not in llm.requests[1]["body"]
    text = llm.requests[1]["body"]["messages"][-1]["content"][1]["text"]
    assert text.startswith("what are people doing?") and "No tool calls were executed" in text   # no fake changes


def test_ask_text_tool_calls_and_empty_answer(agent, llm):
    llm.replies = [{"content": '<tool_call>{"name": "set_counting_line", "arguments": {"x1": 0, "y1": 0.5, '
                               '"x2": 1, "y2": 0.5}}</tool_call>'},
                   {"content": ""}, {"content": ""}]
    r = agent.submit(0, agent.ask, "count people crossing horizontally").result(timeout=20)
    assert agent.an.line_norm == [[0.0, 0.5], [1.0, 0.5]]
    assert r["answer"].startswith("Done: set_counting_line")
    assert r["actions"][0]["result"] == {"ok": True, "line": [[0.0, 0.5], [1.0, 0.5]], "orientation": "horizontal",
                                         "counts": {"a_to_b": "top to bottom", "b_to_a": "bottom to top"}}


def test_ask_strips_text_tool_calls_from_final_answer(agent, llm):
    llm.replies = [{"content": "Two people. <tool_call>{\"name\": \"get_history\", \"arguments\": {\"minutes\": 5}}"
                               "</tool_call>"}, {"content": "routing done"},
                   {"content": "There were two people. <tool_call>bad</tool_call>"}]
    r = agent.submit(0, agent.ask, "history?").result(timeout=20)
    assert r["answer"] == "There were two people."
    assert r["actions"][0]["tool"] == "get_history" and "series" in r["actions"][0]["result"]


def test_ask_bad_tool_arguments_reported_to_model(agent, llm):
    llm.replies = [{"tool_calls": [tool_call("set_alert", {"metric": "temperature", "op": ">", "value": 1}),
                                   tool_call("set_alert", {"metric": "occupancy", "op": ">"}, "call_2"),
                                   {"id": "call_3", "function": {"name": "set_alert", "arguments": "{oops"}}]},
                   {"content": "cannot"}, {"content": "Sorry, I cannot do that."}]
    r = agent.submit(0, agent.ask, "alert on temperature").result(timeout=20)
    assert r["answer"].startswith("Sorry, I cannot do that.\n\n[system] No settings were changed. Failed: set_alert")
    assert [a["result"]["ok"] for a in r["actions"]] == [False, False, False]
    assert "unknown metric" in r["actions"][0]["result"]["error"]
    assert "KeyError" in r["actions"][1]["result"]["error"]
    assert "JSONDecodeError" in r["actions"][2]["result"]["error"]
    assert [x["name"] for x in agent.an.rules_public()] == ["crowding"]
    tool_msgs = [m for m in llm.requests[1]["body"]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["call_1", "call_2", "call_3"]


def test_ask_stops_after_tool_round_limit(agent, llm):
    """At most 3 tool rounds, then a final answer: tool calls the model can no longer see the results of
    must not be executed."""
    llm.replies = [{"tool_calls": [tool_call("get_history", {"minutes": 1})]}] * 3 + \
                  [{"content": "", "tool_calls": [tool_call("remove_alert", {"name": "all"})]}]
    r = agent.submit(0, agent.ask, "loop").result(timeout=20)
    assert len(llm.requests) == 4
    assert "tools" in llm.requests[2]["body"] and "tools" not in llm.requests[3]["body"]
    assert [a["tool"] for a in r["actions"]] == ["get_history"] * 3
    assert [x["name"] for x in agent.an.rules_public()] == ["crowding"]       # the 4th round's call was not run
    assert r["answer"].startswith("Done: get_history")


def test_ask_includes_history(agent, llm):
    llm.replies = [{"content": "r1"}, {"content": "first answer"}, {"content": "r2"}, {"content": "second answer"}]
    agent.submit(0, agent.ask, "q1").result(timeout=20)
    agent.submit(0, agent.ask, "q2").result(timeout=20)
    for req in (llm.requests[2], llm.requests[3]):          # both stages of the second turn carry the history
        msgs = req["body"]["messages"]
        assert msgs[1] == {"role": "user", "content": "q1"} and msgs[2] == {"role": "assistant", "content": "first answer"}


def test_ask_without_tools_and_without_frame(llm, frame):
    a = make_agent(llm.endpoint, frame, use_tools=False).start()
    a.state["frame"] = None
    try:
        r = a.submit(0, a.ask, "hello").result(timeout=20)
    finally:
        a.stop()
    assert r["answer"] == "default answer"
    body = llm.requests[0]["body"]
    assert "tools" not in body
    assert body["messages"][-1]["content"] == [{"type": "text", "text": "hello"}]


def test_run_tool_directly(agent):
    r = agent.run_tool("set_alert", {"metric": "unique_people", "op": ">=", "value": "10"})   # number as string
    assert r["ok"] and r["rule"]["value"] == 10.0 and r["rule"]["name"] == "unique_people >= 10"
    assert agent.run_tool("set_counting_line", {"x1": 0.5, "y1": 0.5, "x2": 0.5, "y2": 0.5})["ok"] is False
    assert agent.run_tool("set_counting_line", {"x1": -0.5, "y1": 0, "x2": 0.5, "y2": 1})["ok"] is False
    assert agent.run_tool("remove_alert", {"name": "unique_people >= 10"}) == {"ok": True, "removed": 1}
    assert agent.run_tool("remove_alert", {"name": "nope"}) == {"ok": False, "removed": 0}
    assert agent.run_tool("remove_alert", {"name": "all"})["removed"] == 1
    assert agent.run_tool("bogus", {}) == {"ok": False, "error": "unknown tool bogus"}
    agent.store.event({"ts": 1, "kind": "alert", "rule": "r"})
    agent.store.event({"ts": 2, "kind": "line_cross"})
    assert [e["kind"] for e in agent.run_tool("get_recent_events", {"minutes": 5, "kind": "alert"})] == ["alert"]
    assert len(agent.run_tool("get_recent_events", {"minutes": 5000})) >= 2                     # clamped, any kind
    h = agent.run_tool("get_history", {"minutes": 0})
    assert h["series"]["minutes"] == 1 and "event_counts" in h


def test_verify_default_describes_scene_without_verdict(agent, llm):
    llm.replies = [{"content": '{"summary": "A dense crowd near the door.", "activities": ["waiting"], '
                               '"anomalies": [], "risk": "low"}'}]
    alert = {"rule": "crowding", "metric": "occupancy", "op": ">", "threshold": 20, "value": 25}
    r = agent.submit(1, agent.verify, alert).result(timeout=20)
    assert r == {"rule": "crowding", "summary": "A dense crowd near the door.", "latency_s": r["latency_s"]}
    ev = agent.store.recent_events(kinds=["alert_context"])
    assert len(ev) == 1 and ev[0]["summary"] == "A dense crowd near the door." and ev[0]["rule"] == "crowding"
    assert agent.store.recent_events(kinds=["alert_verified"]) == []          # no yes/no verdict by default
    assert agent.store.observations()[0]["trigger"] == "alert:crowding"
    assert llm.requests[0]["body"]["max_tokens"] == 256                       # the scene-report prompt


def test_verify_stores_event_and_observation(blind_agent, llm):
    agent = blind_agent
    llm.replies = [{"content": '```json\n{"confirmed": true, "reason": "A dense crowd near the door."}\n```'}]
    alert = {"rule": "crowding", "metric": "occupancy", "op": ">", "threshold": 20, "value": 25}
    r = agent.submit(1, agent.verify, alert).result(timeout=20)
    assert r["confirmed"] is True and r["reason"] == "A dense crowd near the door." and r["rule"] == "crowding"
    ev = agent.store.recent_events(kinds=["alert_verified"])
    assert len(ev) == 1 and ev[0]["confirmed"] is True and ev[0]["ts"] == 12.5
    obs = agent.store.observations()
    assert obs[0]["trigger"] == "alert:crowding" and obs[0]["text"] == "Confirmed: A dense crowd near the door."
    assert any(e["kind"] == "alert_verified" for e in agent.an.events)
    body = llm.requests[0]["body"]
    prompt = body["messages"][-1]["content"][1]["text"]
    # blind (default): neither the measured value nor the live measurements reach the model
    assert '"crowding" (condition: people in view right now > 20)' in prompt and "25" not in prompt
    assert "Measurements at t=" not in body["messages"][0]["content"]
    assert body["max_tokens"] == 128


def test_verify_informed_mode_shows_measurement(llm, frame):
    a = make_agent(llm.endpoint, frame, verify_mode="informed").start()
    llm.replies = [{"content": '{"confirmed": false, "reason": "only a few people"}'}]
    try:
        r = a.submit(1, a.verify, {"rule": "crowding", "metric": "occupancy", "op": ">", "threshold": 20,
                                   "value": 25}).result(timeout=20)
    finally:
        a.stop()
    assert r["confirmed"] is False
    body = llm.requests[0]["body"]
    assert 'rule "crowding" (occupancy > 20; measured 25)' in body["messages"][-1]["content"][1]["text"]
    assert "Measurements at t=" in body["messages"][0]["content"]


def test_verify_unparseable_reply_is_not_confirmed(blind_agent, llm):
    agent = blind_agent
    llm.replies = [{"content": "I am not sure what I see."}]
    r = agent.submit(1, agent.verify, {"rule": "r", "metric": "occupancy", "op": ">", "threshold": 1,
                                       "value": 2}).result(timeout=20)
    assert r["confirmed"] is False and r["reason"] == "I am not sure what I see."
    assert agent.store.observations()[0]["text"].startswith("Not confirmed")


def test_observe_stores_report_and_risk_event(agent, llm):
    llm.replies = [{"content": '{"summary": "A crowd is pushing at the gate.", "activities": ["queueing"], '
                               '"anomalies": ["pushing"], "risk": "high"}'},
                   {"content": 'Report: {"summary": "Quiet corridor.", "activities": [], "anomalies": [], '
                               '"risk": "none"}'}]
    r = agent.submit(2, agent.observe, "periodic").result(timeout=20)
    assert r["summary"] == "A crowd is pushing at the gate." and r["risk"] == "high" and "latency_s" in r
    r2 = agent.submit(2, agent.observe).result(timeout=20)
    assert r2["summary"] == "Quiet corridor."
    obs = agent.store.observations()
    assert [(o["trigger"], o["text"]) for o in obs] == [("periodic", "A crowd is pushing at the gate."),
                                                        ("periodic", "Quiet corridor.")]
    risk = agent.store.recent_events(kinds=["vlm_risk"])
    assert len(risk) == 1 and risk[0]["risk"] == "high" and risk[0]["anomalies"] == ["pushing"]
    # the second request carries the first report as context
    assert "A crowd is pushing at the gate." in llm.requests[1]["body"]["messages"][0]["content"]
    assert llm.requests[0]["body"]["max_tokens"] == 256 and "tools" not in llm.requests[0]["body"]


def test_observe_plain_text_reply(agent, llm):
    llm.replies = [{"content": "Two people walk past the shop."}]
    r = agent.submit(2, agent.observe, "user").result(timeout=20)
    assert r["summary"] == "Two people walk past the shop."
    assert agent.store.observations()[0]["trigger"] == "user"


def test_observe_and_verify_without_frame(agent, llm):
    agent.state["frame"] = None
    assert agent.submit(2, agent.observe).result(timeout=20) is None
    assert agent.submit(1, agent.verify, {"rule": "r"}).result(timeout=20) is None
    assert llm.requests == [] and agent.store.observations() == []


def test_api_key_header(llm, frame):
    a = make_agent(llm.endpoint, frame, api_key="secret").start()
    try:
        a.submit(0, a.ask, "hi").result(timeout=20)
    finally:
        a.stop()
    assert llm.requests[0]["auth"] == "Bearer secret"


def test_endpoint_error_raises_and_worker_survives(agent, llm):
    llm.replies = [500, {"content": "routing"}, {"content": "back online"}]
    with pytest.raises(urllib.error.HTTPError):
        agent.submit(0, agent.ask, "first").result(timeout=20)
    assert agent.online is True                       # /models still answers
    r = agent.submit(0, agent.ask, "second").result(timeout=20)
    assert r["answer"] == "back online"
    assert agent._thread.is_alive() and not agent.busy
    assert [m["text"] for m in agent.store.chat_log()] == ["second", "back online"]


def test_unreachable_endpoint(frame):
    a = make_agent(f"http://127.0.0.1:{free_port()}/v1", frame).start()
    try:
        assert a.online is False
        with pytest.raises(urllib.error.URLError):
            a.submit(0, a.ask, "anyone?").result(timeout=20)
        with pytest.raises(urllib.error.URLError):
            a.submit(2, a.observe).result(timeout=20)
        assert a._thread.is_alive() and a.online is False
        assert a.submit(0, lambda: 42).result(timeout=5) == 42
    finally:
        a.stop()


def test_priority_queue_order(agent):
    gate, order = threading.Event(), []
    started = threading.Event()

    def blocker():
        started.set(); gate.wait(10)

    def job(tag):
        order.append(tag)
        return tag

    agent.submit(5, blocker)
    assert started.wait(5)
    futs = [agent.submit(2, job, "periodic-1"), agent.submit(1, job, "alert-1"), agent.submit(2, job, "periodic-2"),
            agent.submit(0, job, "chat-1"), agent.submit(1, job, "alert-2"), agent.submit(0, job, "chat-2")]
    assert agent.pending() == 6 and agent.pending(0) == 2 and agent.pending(2) == 2
    assert agent.busy
    gate.set()
    assert [f.result(timeout=5) for f in futs]
    assert order == ["chat-1", "chat-2", "alert-1", "alert-2", "periodic-1", "periodic-2"]
    assert agent.pending() == 0


def test_cancelled_job_is_skipped(agent):
    gate = threading.Event()
    agent.submit(0, gate.wait, 10)
    time.sleep(0.2)
    ran = []
    f = agent.submit(0, ran.append, 1)
    assert f.cancel()
    g = agent.submit(0, ran.append, 2)
    gate.set()
    g.result(timeout=5)
    assert ran == [2]


def test_stop_ends_worker(llm, frame):
    a = make_agent(llm.endpoint, frame).start()
    a.stop()
    a._thread.join(5)
    assert not a._thread.is_alive()


def test_context_contains_measurements_and_history(agent):
    agent.store.series({"t": 0, "occ_mean": 2.0, "occ_max": 3, "unique": 4, "crossings": 1})
    agent.store.observation(1.0, "periodic", "Earlier report.")
    ctx = agent.context()
    assert ctx.startswith("Measurements at t=0.0 s (JSON): {")
    snap = json.loads(ctx.split("(JSON): ", 1)[1].split("\n")[0])
    assert snap["occupancy"] == 0 and snap["rules"][0]["name"] == "crowding"
    assert '"samples":1' in ctx and "Previous scene reports: Earlier report." in ctx


def test_intent_guardrail_for_counting_line():
    err = vlm.intent_error
    assert err("make it a horizontal line", "set_counting_line", {"x1": 0.5, "y1": 0, "x2": 0.5, "y2": 1})
    assert err("make it a horizontal line", "set_counting_line", {"x1": 0.5, "y1": 0.5, "x2": 0.5, "y2": 0.5})
    assert err("make it a horizontal line", "set_counting_line", {"x1": 0, "y1": 0.5, "x2": 1, "y2": 0.5}) is None
    assert err("a vertical line please", "set_counting_line", {"x1": 0, "y1": 0.5, "x2": 1, "y2": 0.5})
    assert err("a vertical line please", "set_counting_line", {"x1": 0.3, "y1": 0, "x2": 0.3, "y2": 1}) is None
    assert err("from corner to corner", "set_counting_line", {"x1": 0, "y1": 0, "x2": 1, "y2": 1}) is None
    assert err("horizontal", "set_alert", {"metric": "occupancy"}) is None


def test_ask_guardrail_makes_model_retry(agent, llm):
    llm.replies = [{"tool_calls": [tool_call("set_counting_line", {"x1": 0.5, "y1": 0, "x2": 0.5, "y2": 1})]},
                   {"tool_calls": [tool_call("set_counting_line", {"x1": 0, "y1": 0.5, "x2": 1, "y2": 0.5}, "c2")]},
                   {"content": "done"}, {"content": "The line is now horizontal."}]
    before = agent.an.line_norm
    r = agent.submit(0, agent.ask, "make the counting line horizontal").result(timeout=20)
    assert [a["result"]["ok"] for a in r["actions"]] == [False, True]
    assert "HORIZONTAL" in r["actions"][0]["result"]["error"] and before != agent.an.line_norm
    assert agent.an.line_norm == [[0.0, 0.5], [1.0, 0.5]]
    note = llm.requests[-1]["body"]["messages"][-1]["content"][1]["text"]
    assert "Succeeded: set_counting_line" in note and "Failed: set_counting_line" in note


def test_fallback_intents_only_for_unambiguous_commands():
    f = vlm.fallback_intents
    assert f("Remove all alerts.") == [("remove_alert", {"name": "all"})]
    assert f("please clear every alert rule") == [("remove_alert", {"name": "all"})]
    assert f("Make the counting line horizontal across the middle") == [
        ("set_counting_line", {"x1": 0.0, "y1": 0.5, "x2": 1.0, "y2": 0.5})]
    assert f("set the line horizontally at 75% of the height")[0][1]["y1"] == 0.75
    assert f("move the counting line to a vertical line at one third of the width")[0][1]["x1"] == pytest.approx(1 / 3)
    for q in ("Delete the crowding alert.", "How many people crossed the line?", "alert me if more than 5 people",
              "what is happening?"):
        assert f(q) == []


def test_fallback_runs_when_model_skips_the_change_and_footer_is_truthful(agent, llm):
    llm.replies = [{"content": "sure"}, {"content": "The alerts were removed."}]       # model calls no tool
    r = agent.submit(0, agent.ask, "Remove all alerts.").result(timeout=20)
    assert agent.an.rules_public() == []
    assert r["actions"][0]["source"] == "fallback" and r["actions"][0]["result"]["removed"] == 1
    assert r["answer"].endswith("[system] Changed: removed 1 alert rule(s).")


def test_footer_reports_no_change_for_unrecognised_request(agent, llm):
    llm.replies = [{"content": "ok"}, {"content": "Done, the crowding alert is gone."}]   # a false claim
    r = agent.submit(0, agent.ask, "Delete the crowding alert.").result(timeout=20)
    assert [x["name"] for x in agent.an.rules_public()] == ["crowding"]
    assert r["answer"].endswith("[system] No settings were changed.")


def test_no_footer_for_plain_questions(agent, llm):
    llm.replies = [{"content": "x"}, {"content": "People are walking."}]
    assert agent.submit(0, agent.ask, "what are people doing?").result(timeout=20)["answer"] == "People are walking."


def test_history_keeps_model_words_without_system_footer(agent, llm):
    llm.replies = [{"content": "x"}, {"content": "Removed."}, {"content": "y"}, {"content": "OK."}]
    agent.submit(0, agent.ask, "Remove all alerts.").result(timeout=20)
    assert agent.history[-1] == ("Remove all alerts.", "Removed.")
    agent.submit(0, agent.ask, "thanks").result(timeout=20)
    sent = [m for m in llm.requests[-1]["body"]["messages"] if m["role"] == "assistant"]
    assert sent and all("[system]" not in m["content"] for m in sent)
