"""Tests for driftgate.reporting: attribution, outcomes, report artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from driftgate import reporting as rpt
from driftgate.contracts.validator import ContractValidator
from driftgate.reporting import (
    PER_TEST_CALL_CAP,
    SESSION_NODEID,
    CallTracker,
    RunTotals,
    normalize_endpoint,
    render_markdown,
)


def _validator() -> ContractValidator:
    return ContractValidator(
        {
            "openapi": "3.1.0",
            "paths": {
                "/orders/{orderId}": {"get": {"responses": {"200": {"description": "ok"}}}},
                "/orders": {"post": {"responses": {"202": {"description": "ok"}}}},
            },
        }
    )


def test_normalize_path_uses_spec_template() -> None:
    validator = _validator()
    assert normalize_endpoint("GET", "/orders/ORD-12345", validator) == "/orders/{orderId}"
    assert normalize_endpoint("POST", "/orders", validator) == "/orders"
    assert normalize_endpoint("GET", "/orders/ORD-1?verbose=true", validator) == "/orders/{orderId}"


def test_normalize_path_undeclared_buckets_two_segments() -> None:
    assert normalize_endpoint("GET", "/v1/orders/ORD-1/history", None) == "/v1/orders"
    assert normalize_endpoint("GET", "/orders/ORD-12345", None) == "/orders/ORD-12345"


def test_heuristic_bucket_shallow_paths() -> None:
    assert rpt._heuristic_bucket("/orders") == "/orders"
    assert rpt._heuristic_bucket("/") == "/"


@pytest.fixture
def _httpx_wrapper_guard():
    was_installed = rpt.wrapper_installed()
    rpt.uninstall_httpx_wrapper()
    yield
    rpt.uninstall_httpx_wrapper()
    if was_installed:
        # Restore the session's wrapper so attribution survives these unit tests.
        from driftgate.pytest_plugin import _TRACKER

        rpt.install_httpx_wrapper(_TRACKER)


def test_tracker_caps_per_test_entries() -> None:
    tracker = CallTracker()
    tracker.start_test("test_a")
    for index in range(PER_TEST_CALL_CAP + 50):
        tracker.record("GET", f"/items/{index}", 200)
    assert len(tracker.calls_for("test_a")) == PER_TEST_CALL_CAP


def test_tracker_caps_total_tests() -> None:
    from driftgate.reporting import MAX_TRACKED_TESTS

    tracker = CallTracker()
    for index in range(MAX_TRACKED_TESTS + 10):
        tracker.start_test(f"test_{index}")
        tracker.record("GET", "/x", 200)
    assert tracker.calls_by_test_size == MAX_TRACKED_TESTS


def test_tracker_attributes_off_test_calls_to_session_bucket() -> None:
    tracker = CallTracker()
    tracker.record("GET", "/ping", 200)
    assert SESSION_NODEID in tracker.calls_by_test


def test_httpx_wrapper_captures_plain_and_subclassed_clients(_httpx_wrapper_guard) -> None:
    class _Transport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"ok": True})

    class SubClient(httpx.Client):
        def send(self, request, **kwargs):
            return super().send(request, **kwargs)

    tracker = CallTracker()
    assert rpt.install_httpx_wrapper(tracker) is True
    try:
        with httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(201))
        ) as client:
            client.get("http://test/v1/orders/ORD-9?include=items")
        with SubClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(404))
        ) as client:
            client.post("http://test/v1/orders", json={})
        assert ("GET", "/v1/orders/ORD-9", 201) in tracker.calls_for(SESSION_NODEID)
        assert ("POST", "/v1/orders", 404) in tracker.calls_for(SESSION_NODEID)
    finally:
        rpt.uninstall_httpx_wrapper()
    # Uninstalled: requests still work, and a new call is not recorded.
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200))) as client:
        client.get("http://test/after")
    assert not any(path == "/after" for _, path, _ in tracker.calls_for(SESSION_NODEID))


def test_wrapper_swallows_record_errors_and_keeps_transport_errors(_httpx_wrapper_guard) -> None:
    tracker = CallTracker()
    tracker.record = lambda *args: (_ for _ in ()).throw(RuntimeError("injected"))  # type: ignore[method-assign]
    assert rpt.install_httpx_wrapper(tracker) is True
    try:
        with httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200))
        ) as client:
            response = client.get("http://test/ok")
        assert response.status_code == 200

        def _boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down")

        with httpx.Client(transport=httpx.MockTransport(_boom)) as client:
            with pytest.raises(httpx.ConnectError):
                client.get("http://test/down")
    finally:
        rpt.uninstall_httpx_wrapper()


def test_outcome_precedence_error_beats_failed_beats_passed() -> None:
    raw = {
        "t1": {"setup_failed": True, "call": "passed"},
        "t2": {"call": "failed", "teardown_failed": True},
        "t3": {"call": "passed"},
        "t4": {"call": "skipped"},
        "t5": {},
    }
    outcomes = rpt.merge_outcomes(raw)
    assert outcomes["t1"] == "error"
    assert outcomes["t2"] == "error"
    assert outcomes["t3"] == "passed"
    assert outcomes["t4"] == "skipped"
    assert outcomes["t5"] == "skipped"


def test_build_endpoint_rows_buckets_templates_and_counts_failures() -> None:
    tracker = CallTracker()
    tracker.start_test("t1")
    tracker.record("GET", "/orders/ORD-1", 200)
    tracker.record("GET", "/orders/ORD-1", 404)
    tracker.start_test("t2")
    tracker.record("GET", "/orders/ORD-2", 200)
    tracker.start_test("t3")
    tracker.record("GET", "/orders/ORD-3", 500)
    outcomes = {"t1": "passed", "t2": "passed", "t3": "failed"}
    rows, touched = rpt.build_endpoint_rows(tracker, outcomes, _validator())
    assert len(rows) == 1
    row = rows[0]
    assert (row.method, row.template) == ("GET", "/orders/{orderId}")
    assert row.tests == 3
    assert row.failed == 1
    assert row.statuses_seen == (200, 404, 500)
    assert set(touched[("GET", "/orders/{orderId}")]) == {"t1", "t2", "t3"}


def test_build_coverage_counts_declared_and_uncovered() -> None:
    rows, touched = rpt.build_endpoint_rows(CallTracker(), {}, _validator())
    # Nothing tracked: both declared endpoints uncovered.
    coverage = rpt.build_coverage(_validator(), touched)
    assert coverage is not None
    assert coverage.declared == 2
    assert coverage.touched == 0
    assert coverage.percentage == 0.0
    assert set(coverage.uncovered) == {("GET", "/orders/{orderId}"), ("POST", "/orders")}

    tracker = CallTracker()
    tracker.start_test("t1")
    tracker.record("POST", "/orders", 202)
    _, touched = rpt.build_endpoint_rows(tracker, {"t": "passed"}, _validator())
    coverage = rpt.build_coverage(_validator(), touched)
    assert coverage.touched == 1
    assert coverage.uncovered == (("GET", "/orders/{orderId}"),)


def test_build_coverage_none_without_validator() -> None:
    assert rpt.build_coverage(None, {("GET", "/x"): {"t"}}) is None


def test_render_markdown_all_sections() -> None:
    report = rpt.ReportData(
        run_id="run-41",
        timestamp="2026-09-21T00:00:00+00:00",
        exitstatus=0,
        totals=RunTotals(passed=28, failed=1, errors=0, skipped=2),
        endpoints=(
            rpt.EndpointRow("GET", "/orders/{orderId}", 3, 1, (200, 404)),
            rpt.EndpointRow("POST", "/orders", 5, 0, (202, 400)),
        ),
        coverage=rpt.SpecCoverage(
            declared=9, touched=2, percentage=22.2, uncovered=(("GET", "/exports/{jobId}"),)
        ),
        governor=rpt.GovernorSummary(
            decision="PROMOTE", reasons=("all boundaries pass",), hard_failures=(), soft_signals=()
        ),
        spec_diff={
            "first_snapshot": False,
            "has_breaking": True,
            "added": [],
            "removed": [],
            "changed": [
                {
                    "method": "POST",
                    "template": "/orders",
                    "change": "changed",
                    "max_severity": "breaking",
                    "findings": [
                        {
                            "location": "POST /orders 202",
                            "kind": "required_added",
                            "severity": "breaking",
                            "detail": "response 'status' became required",
                        }
                    ],
                }
            ],
        },
    )
    output = render_markdown(report)
    for heading in (
        "# DriftGate Build Report",
        "## Test totals",
        "## Endpoints exercised",
        "## Spec coverage",
        "## Governor decision",
        "## Changed vs last build",
    ):
        assert heading in output
    assert "| 3 | 1 |" in output.replace("GET | /orders/{orderId} ", "")
    assert "PROMOTE" in output
    assert "[breaking] POST /orders" in output
    assert render_markdown(report) == output  # deterministic


def test_render_markdown_first_snapshot_and_no_data() -> None:
    report = rpt.ReportData(
        run_id="run-1",
        timestamp="t",
        exitstatus=0,
        totals=RunTotals(),
        endpoints=(),
        coverage=None,
        governor=None,
        spec_diff={
            "first_snapshot": True,
            "has_breaking": False,
            "added": [],
            "removed": [],
            "changed": [],
        },
    )
    output = render_markdown(report)
    assert "First snapshot taken" in output
    assert "No HTTP traffic" in output
    assert "No OpenAPI spec configured" in output


def test_report_json_roundtrip(tmp_path: Path) -> None:
    report = rpt.ReportData(
        run_id="r",
        timestamp="t",
        exitstatus=0,
        totals=RunTotals(passed=2),
        endpoints=(rpt.EndpointRow("GET", "/x", 1, 0, (200,)),),
        coverage=None,
        governor=None,
        spec_diff=None,
    )
    restored = rpt.ReportData.from_dict(json.loads(json.dumps(report.to_dict())))
    assert restored == report
