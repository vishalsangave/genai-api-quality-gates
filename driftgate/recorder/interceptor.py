"""httpx transport that records live traffic and replays cassettes offline."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import httpx

from driftgate.config import DriftGateConfig
from driftgate.recorder.cassette import (
    CassetteInteraction,
    CassetteMode,
    CassetteStore,
    RecordedRequest,
    RecordedResponse,
    RecordedSSEFrame,
)
from driftgate.recorder.fingerprint import FingerprintConfig, fingerprint_request
from driftgate.streaming.probe import SSEFrameParser


class _FrameStream(httpx.SyncByteStream):
    """Replay a semantic SSE event as one canonical wire frame at a time."""

    def __init__(self, frames: list[RecordedSSEFrame]) -> None:
        self._frames = frames

    def __iter__(self) -> Iterator[bytes]:
        yield from (frame.to_frame().to_bytes() for frame in self._frames)


class _BytesStream(httpx.SyncByteStream):
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __iter__(self) -> Iterator[bytes]:
        if self._body:
            yield self._body


def _path_and_query(url: httpx.URL) -> tuple[str, dict[str, list[str]]]:
    raw = str(url)
    parsed = urlsplit(raw)
    query: dict[str, list[str]] = {}
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        query.setdefault(key, []).append(value)
    return parsed.path, query


def _deserialize_body(body: bytes, content_type: str | None) -> Any:
    if not body:
        return None
    if content_type and "json" in content_type.lower():
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return body.decode("utf-8", errors="replace")
    return body.decode("utf-8", errors="replace")


class RecordingTransport(httpx.BaseTransport):
    """A synchronous httpx transport for RECORD / REPLAY / LIVE modes.

    Live recordings are intentionally buffered before being returned. This is
    a correctness-first choice for the MVP: it guarantees a finite response
    can be persisted and then replayed byte-for-byte. The streaming telemetry
    probe still measures real-time chunks from LIVE mode; record mode is for
    structural replay rather than latency measurement.
    """

    def __init__(
        self,
        real: httpx.BaseTransport,
        store: CassetteStore,
        *,
        fingerprint_config: FingerprintConfig | None = None,
        record_on_miss: bool = False,
    ) -> None:
        self.real = real
        self.store = store
        self.fingerprint_config = fingerprint_config or FingerprintConfig()
        self.record_on_miss = record_on_miss

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        started_ns = time.perf_counter_ns()
        request.extensions["driftgate_t0"] = started_ns
        body = request.read()
        fp = fingerprint_request(
            request.method,
            str(request.url),
            dict(request.headers),
            body,
            cfg=self.fingerprint_config,
        )

        should_lookup = self.store.mode in (CassetteMode.REPLAY_STRICT, CassetteMode.REPLAY_LENIENT)
        if should_lookup:
            interaction = self.store.lookup(fp)
            if interaction is not None:
                return self._build_replay_response(request, interaction)

        # LIVE never records; lenient replay records on miss (auto_record).
        response = self.real.handle_request(request)
        should_record = self.store.mode is CassetteMode.RECORD or (
            self.store.mode is CassetteMode.REPLAY_LENIENT and self.record_on_miss
        )
        if not should_record:
            return response
        return self._record_and_rebuffer(request, response, fp, body)

    def _build_replay_response(
        self, request: httpx.Request, interaction: CassetteInteraction
    ) -> httpx.Response:
        recorded = interaction.response
        stream: httpx.SyncByteStream
        if recorded.is_stream:
            stream = _FrameStream(recorded.sse_events)
        else:
            content = recorded.body
            if content is None:
                raw = b""
            elif isinstance(content, str):
                raw = content.encode("utf-8")
            else:
                raw = json.dumps(content, separators=(",", ":")).encode("utf-8")
            stream = _BytesStream(raw)
        return httpx.Response(
            recorded.status,
            headers=recorded.headers,
            stream=stream,
            request=request,
            extensions={
                "driftgate_replayed": True,
                "driftgate_t0": request.extensions["driftgate_t0"],
            },
        )

    def _record_and_rebuffer(
        self,
        request: httpx.Request,
        response: httpx.Response,
        fp: Any,
        request_body: bytes,
    ) -> httpx.Response:
        raw_body = response.read()
        headers = dict(response.headers)
        content_type = headers.get("content-type")
        is_stream = bool(content_type and "text/event-stream" in content_type.lower())
        if is_stream:
            parser = SSEFrameParser()
            frames = parser.feed(raw_body) + parser.finish()
            recorded_response = RecordedResponse(
                status=response.status_code,
                headers=headers,
                body=None,
                is_stream=True,
                sse_events=[RecordedSSEFrame.from_frame(frame) for frame in frames],
            )
        else:
            recorded_response = RecordedResponse(
                status=response.status_code,
                headers=headers,
                body=_deserialize_body(raw_body, content_type),
                is_stream=False,
            )
        path, query = _path_and_query(request.url)
        interaction = CassetteInteraction(
            fingerprint=fp.digest,
            canonical=fp.canonical,
            request=RecordedRequest(
                method=request.method,
                path=path,
                query=query,
                headers=dict(request.headers),
                body=_deserialize_body(request_body, request.headers.get("content-type")),
            ),
            response=recorded_response,
        )
        self.store.record(interaction)
        return httpx.Response(
            response.status_code,
            headers=response.headers,
            content=raw_body,
            request=request,
            extensions={"driftgate_t0": request.extensions["driftgate_t0"]},
        )

    def close(self) -> None:
        self.real.close()


def build_client(
    config: DriftGateConfig,
    store: CassetteStore,
    *,
    real: httpx.BaseTransport | None = None,
) -> httpx.Client:
    """Build a synchronous client wired to the DriftGate recording transport."""
    transport = RecordingTransport(
        real or httpx.HTTPTransport(),
        store,
        record_on_miss=config.traffic_mode == "auto_record",
    )
    return httpx.Client(base_url=config.service.base_url, transport=transport, timeout=30.0)
