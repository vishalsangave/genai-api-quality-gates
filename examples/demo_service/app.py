"""Demo GenAI microservice used to exercise the full DriftGate pipeline.

Fault injection is deliberate: the ``X-Demo-Behavior`` header selects one
behavior per request so integration tests can deterministically reproduce
each failure class the framework is designed to catch.

Run directly for manual use::

    python examples/demo_service/app.py 8001
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from collections.abc import AsyncIterator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(title="DriftGate demo GenAI service")

TOKENS = ("DriftGate", " replays ", "recorded ", "traffic ", "deterministically.")
BEHAVIORS = frozenset({"normal", "slow", "invalid_schema", "truncated", "error_stream"})
TOKEN_DELAY_S = 0.015
SLOW_TOKEN_DELAY_S = 0.060


class ChatCompletionRequest(BaseModel):
    messages: list[dict[str, str]] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


class SummarizeRequest(BaseModel):
    text: str = Field(min_length=1)
    max_words: int | None = Field(default=None, ge=1)


def _sse(payload: str) -> bytes:
    return f"data: {payload}\n\n".encode()


def _event(name: str, payload: str) -> bytes:
    return f"event: {name}\ndata: {payload}\n\n".encode()


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(payload: ChatCompletionRequest, request: Request) -> StreamingResponse:
    behavior = request.headers.get("x-demo-behavior", "normal")
    if behavior not in BEHAVIORS:
        behavior = "normal"

    async def stream() -> AsyncIterator[bytes]:
        if behavior == "invalid_schema":
            # A number where the declared per-frame schema expects a string.
            yield _sse(json.dumps({"delta": 123}))
            yield _sse("[DONE]")
            return
        for token in TOKENS:
            await asyncio.sleep(SLOW_TOKEN_DELAY_S if behavior == "slow" else TOKEN_DELAY_S)
            yield _sse(json.dumps({"delta": token}))
        if behavior == "truncated":
            # A dangling partial frame: no blank line, no [DONE], no error
            # event — the silent-truncation failure mode.
            yield b'data: {"delta": "..."'
            return
        if behavior == "error_stream":
            yield _event(
                "error",
                json.dumps({"code": "upstream_timeout", "message": "synthetic mid-stream failure"}),
            )
            return
        yield _sse("[DONE]")

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/v1/summarize")
async def summarize(payload: SummarizeRequest) -> dict[str, str | int]:
    words = payload.text.split()
    if payload.max_words is not None:
        words = words[: payload.max_words]
    summary = " ".join(words)
    return {"summary": summary, "char_count": len(summary)}


def serve(port: int, *, ready: threading.Event | None = None) -> None:
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    serve(port)
