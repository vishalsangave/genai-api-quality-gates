"""Native pytest fixtures and assertions for DriftGate."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from driftgate.config import DriftGateConfig
from driftgate.contracts.validator import ContractValidator
from driftgate.recorder.cassette import CassetteMode, CassetteStore
from driftgate.recorder.interceptor import build_client
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
    """Persist voluntarily-recorded metrics for the CLI release governor."""
    metrics = session.config.stash[_METRICS_KEY]
    if not metrics:
        return
    payload = {"run_id": session.name, **metrics, "pytest_exitstatus": exitstatus}
    output = Path(".driftgate/metrics.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


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
