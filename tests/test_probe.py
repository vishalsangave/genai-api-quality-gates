from __future__ import annotations

import time

import httpx
import pytest

from driftgate.streaming.probe import (
    SSEFrameParser,
    StreamingProbe,
    StreamingSLAViolation,
    assert_streaming_sla,
)


def _response(body: bytes, *, content_type: str = "text/event-stream") -> httpx.Response:
    return httpx.Response(200, headers={"content-type": content_type}, content=body)


def test_parser_handles_crlf_multiline_comments_and_chunk_boundaries() -> None:
    parser = SSEFrameParser()
    wire = b": keepalive\r\nevent: message\r\nid: 7\r\ndata: first\r\ndata: second\r\n\r\n"
    frames = []
    for byte in wire:
        frames.extend(parser.feed(bytes([byte])))
    frames.extend(parser.finish())

    assert len(frames) == 1
    assert frames[0].event == "message"
    assert frames[0].id == "7"
    assert frames[0].data == "first\nsecond"


def test_parser_treats_bare_cr_as_line_terminator_per_spec() -> None:
    """Mixed LF/CRLF producers must not lose frames (see docs/ARCHITECTURE.md)."""
    parser = SSEFrameParser()
    frames = parser.feed(b"data: x\r\n\rdata: y\n\n") + parser.finish()
    assert [frame.data for frame in frames] == ["x", "y"]

    # Bare-CR delimiter split across chunks.
    parser = SSEFrameParser()
    frames = parser.feed(b"data: a\r") + parser.feed(b"\rdata: b\n\n") + parser.finish()
    assert [frame.data for frame in frames] == ["a", "b"]

    # A CRLF split across chunks is one terminator, not two.
    parser = SSEFrameParser()
    frames = parser.feed(b"data: a\r") + parser.feed(b"\ndata: b\n\n") + parser.finish()
    assert [frame.data for frame in frames] == ["a\nb"]


def test_probe_calculates_frames_and_clean_termination() -> None:
    response = _response(b'data: {"delta":"one"}\n\ndata: {"delta":"two"}\n\ndata: [DONE]\n\n')
    telemetry = StreamingProbe().observe(response, time.perf_counter_ns())

    assert telemetry.token_frames == 2
    assert telemetry.ttft_ms is not None
    assert telemetry.terminated_cleanly
    assert telemetry.saw_done_marker
    assert_streaming_sla(telemetry, max_ttft_ms=1_000, max_itl_ms=1_000)


def test_probe_detects_error_event_and_missing_done() -> None:
    error_response = _response(b'event: error\ndata: {"message":"upstream failed"}\n\n')
    error_telemetry = StreamingProbe().observe(error_response)
    with pytest.raises(StreamingSLAViolation, match="structured error"):
        assert_streaming_sla(error_telemetry, max_ttft_ms=10, max_itl_ms=10)

    truncated_response = _response(b"data: token\n\n")
    truncated_telemetry = StreamingProbe().observe(truncated_response)
    with pytest.raises(StreamingSLAViolation, match=r"missing \[DONE\]"):
        assert_streaming_sla(truncated_telemetry, max_ttft_ms=10, max_itl_ms=10)


def test_probe_requires_event_stream_content_type() -> None:
    telemetry = StreamingProbe().observe(_response(b"{}", content_type="application/json"))
    with pytest.raises(StreamingSLAViolation, match="text/event-stream"):
        assert_streaming_sla(telemetry, max_ttft_ms=10, max_itl_ms=10)
