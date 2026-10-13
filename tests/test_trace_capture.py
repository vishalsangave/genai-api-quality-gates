from __future__ import annotations

import json

import httpx

from driftgate.agents import (
    AgentTrace,
    ToolCall,
    TraceCollector,
    TraceRecorder,
    attach,
    extract_trace,
)
from driftgate.config import DriftGateConfig


def _trace() -> AgentTrace:
    rec = TraceRecorder("t")
    rec.step("Router", handoff_to="Math")
    rec.step("Math", tool_calls=[ToolCall(name="calculator")])
    return rec.finish("4")


def _client(handler, collector, **kw) -> httpx.Client:
    client = httpx.Client(base_url="http://svc", transport=httpx.MockTransport(handler))
    attach(client, collector, **kw)
    return client


def test_captures_trace_from_json_body_key():
    body = {"response": "4", "trace": _trace().model_dump(mode="json")}
    collector = TraceCollector()
    with _client(lambda r: httpx.Response(200, json=body), collector) as c:
        data = c.post("/chat", json={}).json()
    assert data["response"] == "4"  # body still readable by the test
    assert collector.last.agents_visited == ["Router", "Math"]


def test_captures_trace_from_header_and_custom_names():
    header = json.dumps(_trace().model_dump(mode="json"))
    collector = TraceCollector()
    handler = lambda r: httpx.Response(200, json={}, headers={"X-Run": header})  # noqa: E731
    with _client(handler, collector, header="x-run") as c:
        c.get("/a")
        c.get("/b")
    assert len(collector) == 2 and collector.last.tool_names == ["calculator"]


def test_ignores_missing_malformed_and_non_json_traces():
    collector = TraceCollector()
    responses = iter([
        httpx.Response(200, json={"no": "trace"}),
        httpx.Response(200, json={"trace": {"steps": "nope"}}),
        httpx.Response(200, headers={"x-driftgate-trace": "{not json"}, json={}),
        httpx.Response(200, text="plain"),
    ])
    with _client(lambda r: next(responses), collector) as c:
        for _ in range(4):
            c.get("/x")
    assert len(collector) == 0


def test_empty_collector_explains_how_to_expose_a_trace():
    try:
        _ = TraceCollector().last
    except LookupError as e:
        assert "x-driftgate-trace" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected LookupError")


def test_streaming_responses_are_not_buffered():
    collector = TraceCollector()
    handler = lambda r: httpx.Response(  # noqa: E731
        200, headers={"content-type": "text/event-stream"}, content=b"data: hi\n\n"
    )
    with _client(handler, collector) as c, c.stream("GET", "/s") as resp:
        assert resp.read() == b"data: hi\n\n"
    assert len(collector) == 0


def test_extract_trace_direct_and_agents_config_defaults():
    resp = httpx.Response(200, json={"trace": _trace().model_dump(mode="json")})
    assert extract_trace(resp) is not None
    assert extract_trace(resp, body_key=None) is None
    cfg = DriftGateConfig().agents
    assert cfg.capture_traces and cfg.trace_header == "x-driftgate-trace" and cfg.trace_body_key == "trace"
