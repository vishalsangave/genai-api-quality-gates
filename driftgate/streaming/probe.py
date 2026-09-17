"""SSE frame parsing and latency telemetry.

The parser intentionally operates on semantic SSE frames, **not raw TCP
chunks**. ``feed`` accepts arbitrary byte slices (including a one-byte slice)
and only emits an event after the SSE blank-line delimiter arrives. That keeps
TTFT/ITL measurements independent of packet fragmentation, OS buffering, and
proxy behavior.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import httpx
import numpy as np


class StreamingSLAViolation(AssertionError):
    """A streaming response broke a status, integrity, or latency invariant."""


@dataclass(frozen=True)
class SSEFrame:
    event: str | None
    id: str | None
    data: str

    def to_bytes(self) -> bytes:
        """Serialize the semantic frame into a canonical SSE wire frame."""
        lines: list[str] = []
        if self.event is not None:
            lines.append(f"event: {self.event}")
        if self.id is not None:
            lines.append(f"id: {self.id}")
        for line in self.data.split("\n"):
            lines.append(f"data: {line}")
        return ("\n".join(lines) + "\n\n").encode("utf-8")


@dataclass(frozen=True)
class StreamSample:
    index: int
    t_offset_ns: int
    frame: SSEFrame


@dataclass
class StreamTelemetry:
    ttft_ms: float | None
    itl_ms: list[float]
    p50_itl_ms: float
    p95_itl_ms: float
    max_itl_ms: float
    total_ms: float
    frames: list[StreamSample]
    token_frames: int
    terminated_cleanly: bool
    saw_done_marker: bool
    raw_bytes: int
    error_frames: list[SSEFrame] = field(default_factory=list)
    status_code: int | None = None
    content_type: str | None = None
    replayed: bool = False
    dangling_bytes: int = 0


class SSEFrameParser:
    """Incrementally split arbitrary byte chunks into semantic SSE frames."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    @property
    def pending_bytes(self) -> int:
        return len(self._buffer)

    def feed(self, chunk: bytes) -> list[SSEFrame]:
        if not chunk:
            return []
        # Normalizing at ingress means the rest of the parser only needs to
        # reason about LF delimiters. A CR can land at a chunk boundary, but
        # replacing every complete CRLF before frame scanning remains safe:
        # a dangling CR is retained until the next feed/finish call.
        self._buffer.extend(chunk)
        return self._extract_complete_frames()

    def finish(self) -> list[SSEFrame]:
        """Flush a final complete frame; retain an incomplete tail for diagnostics."""
        if self._buffer.endswith(b"\r"):
            self._buffer[-1:] = b"\n"
        self._buffer[:] = self._buffer.replace(b"\r\n", b"\n")
        frames = self._extract_complete_frames()
        # SSE permits an event to end at EOF only if it includes a final
        # newline. Treat that as a frame; a partial line remains dangling and
        # causes ``terminated_cleanly`` to be false in the probe.
        if self._buffer.endswith(b"\n") and self._buffer.strip(b"\n"):
            raw = bytes(self._buffer)
            self._buffer.clear()
            frame = self._parse_frame(raw.rstrip(b"\n"))
            if frame is not None:
                frames.append(frame)
        return frames

    def _extract_complete_frames(self) -> list[SSEFrame]:
        self._buffer[:] = self._buffer.replace(b"\r\n", b"\n")
        frames: list[SSEFrame] = []
        while True:
            try:
                delimiter = self._buffer.index(b"\n\n")
            except ValueError:
                break
            raw = bytes(self._buffer[:delimiter])
            del self._buffer[: delimiter + 2]
            frame = self._parse_frame(raw)
            if frame is not None:
                frames.append(frame)
        return frames

    @staticmethod
    def _parse_frame(raw: bytes) -> SSEFrame | None:
        if not raw:
            return None
        text = raw.decode("utf-8", errors="replace")
        event: str | None = None
        event_id: str | None = None
        data_lines: list[str] = []
        for line in text.split("\n"):
            if not line or line.startswith(":"):
                continue
            field, separator, value = line.partition(":")
            if not separator:
                continue
            if value.startswith(" "):
                value = value[1:]
            if field == "data":
                data_lines.append(value)
            elif field == "event":
                event = value
            elif field == "id":
                event_id = value
        if not data_lines and event is None and event_id is None:
            return None
        return SSEFrame(event=event, id=event_id, data="\n".join(data_lines))


class StreamingProbe:
    """Consume an SSE response and calculate TTFT / inter-token telemetry."""

    def __init__(
        self,
        *,
        done_marker: str = "[DONE]",
        token_predicate: Callable[[SSEFrame], bool] | None = None,
    ) -> None:
        self.done_marker = done_marker
        self.token_predicate = token_predicate or self._default_token_predicate

    def _default_token_predicate(self, frame: SSEFrame) -> bool:
        return (
            bool(frame.data.strip())
            and frame.data.strip() != self.done_marker
            and frame.event != "error"
        )

    def observe(self, response: httpx.Response, t0_ns: int | None = None) -> StreamTelemetry:
        """Consume ``response.iter_bytes`` and return frame-level telemetry."""
        t0 = t0_ns if t0_ns is not None else time.perf_counter_ns()
        content_type = response.headers.get("content-type")
        parser = SSEFrameParser()
        frames: list[StreamSample] = []
        token_times: list[int] = []
        error_frames: list[SSEFrame] = []
        raw_bytes = 0
        saw_done = False

        for chunk in response.iter_bytes():
            raw_bytes += len(chunk)
            now = time.perf_counter_ns()
            for frame in parser.feed(chunk):
                sample = StreamSample(len(frames), now - t0, frame)
                frames.append(sample)
                if frame.event == "error":
                    error_frames.append(frame)
                if frame.data.strip() == self.done_marker:
                    saw_done = True
                elif self.token_predicate(frame):
                    token_times.append(now)

        now = time.perf_counter_ns()
        for frame in parser.finish():
            sample = StreamSample(len(frames), now - t0, frame)
            frames.append(sample)
            if frame.event == "error":
                error_frames.append(frame)
            if frame.data.strip() == self.done_marker:
                saw_done = True
            elif self.token_predicate(frame):
                token_times.append(now)

        ttft = (token_times[0] - t0) / 1_000_000 if token_times else None
        itls = [
            (later - earlier) / 1_000_000
            for earlier, later in zip(token_times, token_times[1:], strict=False)
        ]
        return StreamTelemetry(
            ttft_ms=ttft,
            itl_ms=itls,
            p50_itl_ms=float(np.percentile(itls, 50)) if itls else 0.0,
            p95_itl_ms=float(np.percentile(itls, 95)) if itls else 0.0,
            max_itl_ms=max(itls, default=0.0),
            total_ms=(time.perf_counter_ns() - t0) / 1_000_000,
            frames=frames,
            token_frames=len(token_times),
            terminated_cleanly=saw_done and parser.pending_bytes == 0,
            saw_done_marker=saw_done,
            raw_bytes=raw_bytes,
            error_frames=error_frames,
            status_code=response.status_code,
            content_type=content_type,
            replayed=bool(response.extensions.get("driftgate_replayed", False)),
            dangling_bytes=parser.pending_bytes,
        )


def assert_streaming_sla(
    telemetry: StreamTelemetry,
    *,
    max_ttft_ms: float,
    max_itl_ms: float,
    require_done: bool = True,
    allow_replay: bool = False,
) -> StreamTelemetry:
    """Assert response structural and live-stream latency invariants.

    Replayed cassettes emit immediately by design; measuring an SLA against
    them would give a meaningless near-zero result. Structural invariants are
    still useful offline, but callers must opt in via ``allow_replay=True``
    to apply latency assertions to a replayed stream.
    """
    if telemetry.status_code != 200:
        raise StreamingSLAViolation(
            f"SSE endpoint returned HTTP {telemetry.status_code}, expected 200"
        )
    if not telemetry.content_type or "text/event-stream" not in telemetry.content_type.lower():
        raise StreamingSLAViolation(
            f"SSE endpoint content-type {telemetry.content_type!r} does not include text/event-stream"
        )
    if telemetry.error_frames:
        raise StreamingSLAViolation(
            "SSE stream emitted structured error event(s): "
            + "; ".join(frame.data for frame in telemetry.error_frames)
        )
    if require_done and not telemetry.terminated_cleanly:
        detail = (
            "missing [DONE] marker" if not telemetry.saw_done_marker else "dangling partial frame"
        )
        raise StreamingSLAViolation(f"SSE stream did not terminate cleanly: {detail}")
    if telemetry.replayed and not allow_replay:
        return telemetry
    if telemetry.ttft_ms is None:
        raise StreamingSLAViolation("SSE stream emitted no token frames; TTFT cannot be measured")
    if telemetry.ttft_ms > max_ttft_ms:
        raise StreamingSLAViolation(
            f"TTFT SLA breached: {telemetry.ttft_ms:.2f}ms > {max_ttft_ms:.2f}ms"
        )
    if telemetry.max_itl_ms > max_itl_ms:
        raise StreamingSLAViolation(
            f"Inter-token latency SLA breached: {telemetry.max_itl_ms:.2f}ms > {max_itl_ms:.2f}ms"
        )
    return telemetry


def frames_to_bytes(frames: Iterable[SSEFrame]) -> bytes:
    """Serialize frames to a canonical SSE byte sequence (mostly test support)."""
    return b"".join(frame.to_bytes() for frame in frames)
