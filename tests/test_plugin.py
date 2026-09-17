from __future__ import annotations

import httpx

from driftgate.pytest_plugin import _safe_nodeid, assert_streaming_sla


def test_per_test_cassette_name_is_stable_and_safe() -> None:
    assert _safe_nodeid("tests/test_example.py::test_plugin_recorder[param value]") == (
        "tests_test_example.py_test_plugin_recorder_param_value"
    )


def test_streaming_sla_helper_accepts_live_semantic_frames() -> None:
    response = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=b'data: {"delta":"hello"}\n\ndata: {"delta":"world"}\n\ndata: [DONE]\n\n',
    )
    telemetry = assert_streaming_sla(response, max_ttft_ms=1_000, max_itl_ms=1_000)
    assert telemetry.terminated_cleanly
