"""Automatic capture of agent traces from API responses.

A multi-agent service exposes the run it just performed either in a response
header (default ``x-driftgate-trace``, JSON) or in the JSON body under a key
(default ``trace``). ``attach`` hooks an httpx client so every response, live
or replayed from a cassette, is parsed into an ``AgentTrace`` and collected.
Unparsable trace data is ignored: capture must never break the request.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx

from driftgate.agents.trace import AgentTrace

DEFAULT_TRACE_HEADER = "x-driftgate-trace"
DEFAULT_TRACE_BODY_KEY = "trace"


class TraceCollector:
    """Traces captured during one test, in request order."""

    def __init__(self) -> None:
        self.traces: list[AgentTrace] = []

    def __iter__(self) -> Iterator[AgentTrace]:
        return iter(self.traces)

    def __len__(self) -> int:
        return len(self.traces)

    @property
    def last(self) -> AgentTrace:
        if not self.traces:
            raise LookupError(
                "no agent trace was captured; the service must return one in the "
                f"{DEFAULT_TRACE_HEADER!r} header or a {DEFAULT_TRACE_BODY_KEY!r} JSON body key"
            )
        return self.traces[-1]


def extract_trace(
    response: httpx.Response,
    *,
    header: str = DEFAULT_TRACE_HEADER,
    body_key: str | None = DEFAULT_TRACE_BODY_KEY,
) -> AgentTrace | None:
    raw = response.headers.get(header)
    try:
        if raw:
            return AgentTrace.model_validate(json.loads(raw))
        content_type = response.headers.get("content-type", "")
        if body_key and "json" in content_type:
            response.read()  # non-streaming JSON only; safe inside a response hook
            payload = response.json()
            if isinstance(payload, dict) and isinstance(payload.get(body_key), dict):
                return AgentTrace.model_validate(payload[body_key])
    except Exception:
        return None
    return None


def attach(
    client: httpx.Client,
    collector: TraceCollector,
    *,
    header: str = DEFAULT_TRACE_HEADER,
    body_key: str | None = DEFAULT_TRACE_BODY_KEY,
) -> None:
    def hook(response: httpx.Response) -> None:
        trace = extract_trace(response, header=header, body_key=body_key)
        if trace is not None:
            collector.traces.append(trace)

    client.event_hooks["response"] = [*client.event_hooks.get("response", []), hook]
