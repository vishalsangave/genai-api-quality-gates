"""Native pytest fixtures and assertions for DriftGate."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from driftgate.config import DriftGateConfig
from driftgate.contracts.validator import ContractValidator
from driftgate.recorder.cassette import CassetteMode, CassetteStore
from driftgate.recorder.interceptor import build_client
from driftgate.reporting import (
    SNAPSHOT_JSON,
    CallTracker,
    RunTotals,
    build_coverage,
    build_endpoint_rows,
    evaluate_governor,
    install_httpx_wrapper,
    merge_outcomes,
    uninstall_httpx_wrapper,
)
from driftgate.specdiff import (
    save_snapshot as save_spec_snapshot,
)
from driftgate.streaming.probe import StreamingProbe, StreamTelemetry
from driftgate.streaming.probe import assert_streaming_sla as _assert_sla


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("driftgate", "DriftGate API quality-gate options")
    group.addoption("--driftgate-config", metavar="PATH", help="Path to driftgate.config.yaml")
    group.addoption(
        "--cassette-mode",
        choices=["record", "replay_strict", "replay_lenient", "live", "auto_record"],
        help="Override configured traffic_mode for this test run",
    )
    group.addoption("--record", action="store_true", help="Shortcut for --cassette-mode=record")
    group.addoption("--cassette-dir", metavar="PATH", help="Override configured cassette_dir")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "driftgate_cassette(name): use a shared cassette filename")
    config.addinivalue_line("markers", "nightly: runs metered Layer 3 evaluation")
    config.addinivalue_line("markers", "slow: long-running stateful endpoint test")
    config.addinivalue_line("markers", "live: test requires a reachable target service")
    config.stash[_METRICS_KEY] = {}
    _OUTCOMES.clear()
    _TRACKER.calls_by_test.clear()
    install_httpx_wrapper(_TRACKER)


def pytest_unconfigure(config: pytest.Config) -> None:
    uninstall_httpx_wrapper()


def pytest_runtest_logstart(nodeid: str, location: object) -> None:
    _TRACKER.start_test(nodeid)


def pytest_runtest_logfinish(nodeid: str, location: object) -> None:
    _TRACKER.finish_test()


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    try:
        phases = _OUTCOMES.setdefault(report.nodeid, {})
        if report.when == "call":
            phases["call"] = report.outcome
        elif report.when == "setup":
            if report.outcome == "failed":
                phases["setup_failed"] = True
            elif report.outcome == "skipped":
                phases["call"] = "skipped"
        elif report.when == "teardown" and report.outcome == "failed":
            phases["teardown_failed"] = True
    except Exception:  # pragma: no cover - reporting must never break a run
        pass


_TRACKER = CallTracker()
_OUTCOMES: dict[str, dict[str, Any]] = {}

_METRICS_KEY: pytest.StashKey[dict[str, Any]] = pytest.StashKey()


def record_run_metrics(pytestconfig: pytest.Config, **metrics: Any) -> None:
    """Publish one test/session metric for ``driftgate run``.

    Test code can call this helper after collecting real service telemetry:

    ``record_run_metrics(pytestconfig, safety_pass_rate=1.0, ...)``.

    Multiple calls merge fields, which lets specialized fixtures contribute
    latency samples while a policy test contributes safety/task-success rates.
    Values must be JSON-serializable because the session-finish hook persists
    them for the separate CLI process.
    """
    pytestconfig.stash[_METRICS_KEY].update(metrics)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Persist metrics for the CLI governor, then write the build report."""
    try:
        metrics = session.config.stash[_METRICS_KEY]
        if metrics:
            payload = {"run_id": session.name, **metrics, "pytest_exitstatus": exitstatus}
            Path(".driftgate").mkdir(parents=True, exist_ok=True)
            Path(".driftgate/metrics.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )
    except Exception as error:
        print(f"driftgate: metrics persistence failed: {error}", file=sys.stderr)
    try:
        _write_build_report(session, exitstatus)
    except Exception as error:  # pragma: no cover - reporting must never break a run
        print(f"driftgate: report generation failed: {error}", file=sys.stderr)


def _write_build_report(session: pytest.Session, exitstatus: int) -> None:
    """Build report.json + report.md + spec-snapshot.json; never raises outward."""
    if hasattr(session.config, "workerinput"):  # xdist worker: skip artifact writes
        return
    from datetime import UTC, datetime

    from driftgate import reporting as rpt
    from driftgate import specdiff

    config = _effective_config(session.config)
    validator = None
    if config.contract.openapi_spec:
        try:
            validator = ContractValidator.from_file(config.contract.openapi_spec)
        except Exception as error:
            validator = None
            print(f"driftgate: contract validator skipped: {error}", file=sys.stderr)

    old_snapshot = specdiff.load_snapshot(SNAPSHOT_JSON)
    spec_diff = None
    if validator is not None:
        try:
            new_snapshot = specdiff.snapshot(validator.spec)
            spec_diff = specdiff.diff(old_snapshot, new_snapshot)
            save_spec_snapshot(SNAPSHOT_JSON, new_snapshot)
        except Exception as error:
            print(f"driftgate: spec diff skipped: {error}", file=sys.stderr)

    governor = evaluate_governor(config)
    outcomes = merge_outcomes(_OUTCOMES)
    rows, touched = build_endpoint_rows(_TRACKER, outcomes, validator)
    coverage = build_coverage(validator, touched)
    totals = RunTotals(
        passed=sum(1 for o in outcomes.values() if o == "passed"),
        failed=sum(1 for o in outcomes.values() if o == "failed"),
        errors=sum(1 for o in outcomes.values() if o == "error"),
        skipped=sum(1 for o in outcomes.values() if o == "skipped"),
    )
    report = rpt.ReportData(
        run_id=session.name,
        timestamp=datetime.now(UTC).isoformat(),
        exitstatus=exitstatus,
        totals=totals,
        endpoints=rows,
        coverage=coverage,
        governor=governor,
        spec_diff=spec_diff.to_dict() if spec_diff else None,
    )
    rpt.write_report(report)


def _effective_config(pytestconfig: pytest.Config) -> DriftGateConfig:
    configured_path = pytestconfig.getoption("driftgate_config")
    config = DriftGateConfig.load(configured_path)
    mode = "record" if pytestconfig.getoption("record") else pytestconfig.getoption("cassette_mode")
    cassette_dir = pytestconfig.getoption("cassette_dir")
    updates: dict[str, object] = {}
    if mode:
        updates["traffic_mode"] = mode
    if cassette_dir:
        updates["cassette_dir"] = Path(cassette_dir)
    return config.model_copy(update=updates) if updates else config


@pytest.fixture(scope="session")
def driftgate_config(pytestconfig: pytest.Config) -> DriftGateConfig:
    """Validated config with CLI mode/path overrides applied."""
    return _effective_config(pytestconfig)


def _safe_nodeid(nodeid: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", nodeid).strip("_")
    return sanitized or "unnamed_test"


@pytest.fixture
def driftgate_recorder(
    request: pytest.FixtureRequest, driftgate_config: DriftGateConfig
) -> Iterator[CassetteStore]:
    """A per-test cassette selected deterministically from pytest node ID."""
    marker = request.node.get_closest_marker("driftgate_cassette")
    if marker:
        if len(marker.args) != 1 or not isinstance(marker.args[0], str):
            raise pytest.UsageError("@pytest.mark.driftgate_cassette requires one string name")
        name = marker.args[0]
    else:
        name = _safe_nodeid(request.node.nodeid)
    path = driftgate_config.cassette_dir / f"{name}.yaml"
    mode = CassetteMode.from_traffic_mode(driftgate_config.traffic_mode)
    store = CassetteStore(
        path,
        mode,
        record_on_miss=driftgate_config.traffic_mode == "auto_record",
        service=driftgate_config.service.base_url,
    )
    yield store
    store.save()


@pytest.fixture
def driftgate_client(
    driftgate_config: DriftGateConfig, driftgate_recorder: CassetteStore
) -> Iterator[httpx.Client]:
    """Synchronous httpx client with AST-aware record/replay interception."""
    with build_client(driftgate_config, driftgate_recorder) as client:
        yield client


@pytest.fixture(scope="session")
def driftgate_contract(driftgate_config: DriftGateConfig) -> ContractValidator | None:
    """Precompiled Layer-1 validator, or None when no spec is configured."""
    path = driftgate_config.contract.openapi_spec
    return ContractValidator.from_file(path) if path and driftgate_config.contract.enforce else None


def assert_streaming_sla(
    response: httpx.Response,
    *,
    max_ttft_ms: float = 800.0,
    max_itl_ms: float = 35.0,
    require_done: bool = True,
) -> StreamTelemetry:
    """Consume a streaming response and enforce semantic-frame SLAs.

    The response's recording transport owns ``driftgate_t0``. A normal httpx
    response without that extension is still valid and is timed from assertion
    entry, which keeps this helper usable outside the fixture ecosystem.
    """
    probe = StreamingProbe()
    telemetry = probe.observe(response, response.extensions.get("driftgate_t0"))
    return _assert_sla(
        telemetry,
        max_ttft_ms=max_ttft_ms,
        max_itl_ms=max_itl_ms,
        require_done=require_done,
    )
