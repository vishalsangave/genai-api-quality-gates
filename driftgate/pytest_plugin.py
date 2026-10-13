"""Native pytest fixtures and assertions for DriftGate."""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NoReturn

import httpx
import pytest

from driftgate.agents.capture import TraceCollector
from driftgate.agents.capture import attach as attach_traces
from driftgate.agents.contracts import AgentContract
from driftgate.config import DriftGateConfig, JudgeConfig
from driftgate.contracts.validator import ContractValidator
from driftgate.datasets import CHECK_HINT, GENERATE_HINT, check_dataset
from driftgate.metamorphic.evaluator import (
    EmbeddingSimilarityJudge,
    SemanticComparer,
    SemanticSimilarityJudge,
)
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


def _resolve_dataset(raw: str, metafunc: pytest.Metafunc) -> Path | None:
    """Try the path as given, then relative to the test file, then the rootdir."""
    candidates = [Path(raw)]
    if not Path(raw).is_absolute():
        candidates += [Path(metafunc.definition.path).parent / raw, metafunc.config.rootpath / raw]
    return next((c for c in candidates if c.is_file()), None)


def _dataset_param(reason: str, *, skip: bool) -> Any:
    """One placeholder case that skips or fails, so a bad dataset never aborts collection."""
    mark = pytest.mark.skip(reason=reason) if skip else pytest.mark.driftgate_dataset_error(reason)
    return pytest.param(None, None, marks=mark, id="GoldenDataset")


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Generate tests from a golden dataset file or a red team attack library."""
    marker = metafunc.definition.get_closest_marker("golden_dataset")
    if marker is not None and {"prompt", "golden_response"} <= set(metafunc.fixturenames):
        if not marker.args or not isinstance(marker.args[0], str):
            raise pytest.UsageError(
                "@pytest.mark.golden_dataset requires one string argument for the file path."
            )
        dataset_path = _resolve_dataset(marker.args[0], metafunc)
        skip_problems = _effective_config(metafunc.config).datasets.on_problem == "skip"
        if dataset_path is None:
            cases = [_dataset_param(
                f"Golden dataset not found: {marker.args[0]}. "
                + GENERATE_HINT.format(path=marker.args[0]), skip=skip_problems,
            )]
        else:
            report = check_dataset(dataset_path)
            if report.errors:
                cases = [_dataset_param(
                    f"Golden dataset {dataset_path} is invalid ({report.summary()}). "
                    + CHECK_HINT.format(path=dataset_path), skip=skip_problems,
                )]
            elif not report.examples:
                cases = [_dataset_param(f"Golden dataset is empty: {dataset_path}", skip=True)]
            else:
                cases = [
                    pytest.param(e["prompt"], e["response"], id=f"GoldenExample-{i}")
                    for i, e in enumerate(report.examples)
                ]
        metafunc.parametrize(("prompt", "golden_response"), cases)

    red_team = metafunc.definition.get_closest_marker("red_team")
    if red_team is not None and "attack_prompt" in metafunc.fixturenames:
        from driftgate.security.prompt_attacks import attacks_for

        try:
            attacks = attacks_for(*red_team.args)
        except ValueError as e:
            raise pytest.UsageError(f"@pytest.mark.red_team: {e}") from e
        metafunc.parametrize(
            "attack_prompt",
            attacks,
            ids=[f"RedTeam-{i}" for i in range(len(attacks))],
        )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "driftgate_cassette(name): use a shared cassette filename")
    config.addinivalue_line("markers", "golden_dataset(path): parameterize a test with a golden dataset")
    config.addinivalue_line("markers", "driftgate_dataset_error(reason): internal; fails a test whose golden dataset is unusable")
    config.addinivalue_line("markers", "agent_contract(path): specify the agent contract for a test")
    config.addinivalue_line("markers", "red_team(*categories): parameterize attack_prompt with the attack library, optionally limited to categories (injection, tool_abuse, exfiltration, inter_agent, indirect_injection, ...)")
    config.addinivalue_line("markers", "nightly: runs metered Layer 3 evaluation")
    config.addinivalue_line("markers", "slow: long-running stateful endpoint test")
    config.addinivalue_line("markers", "live: test requires a reachable target service")
    config.stash[_METRICS_KEY] = {}
    _OUTCOMES.clear()
    _TRACKER.calls_by_test.clear()
    install_httpx_wrapper(_TRACKER)


def pytest_unconfigure(config: pytest.Config) -> None:
    uninstall_httpx_wrapper()


def pytest_runtest_setup(item: pytest.Item) -> None:
    error = item.get_closest_marker("driftgate_dataset_error")
    if error is not None:
        pytest.fail(error.args[0], pytrace=False)


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
    driftgate_config: DriftGateConfig,
    driftgate_recorder: CassetteStore,
    driftgate_traces: TraceCollector,
) -> Iterator[httpx.Client]:
    """Synchronous httpx client with AST-aware record/replay interception."""
    with build_client(driftgate_config, driftgate_recorder) as client:
        if driftgate_config.agents.capture_traces:
            attach_traces(
                client,
                driftgate_traces,
                header=driftgate_config.agents.trace_header,
                body_key=driftgate_config.agents.trace_body_key,
            )
        yield client


@pytest.fixture
def driftgate_traces() -> TraceCollector:
    """Agent traces captured from this test's responses (header or JSON body key)."""
    return TraceCollector()


def _judge_unavailable(judge_config: JudgeConfig, message: str) -> NoReturn:
    """Fail closed (default) or skip, per ``judge.on_error``; never pass."""
    if judge_config.on_error == "skip":
        pytest.skip(message)
    pytest.fail(message, pytrace=False)


def _require_api_key(judge_config: JudgeConfig, label: str) -> str:
    api_key = os.environ.get(judge_config.api_key_env)
    if not api_key:
        _judge_unavailable(judge_config, f"{label} judge requires {judge_config.api_key_env} to be set.")
    return api_key


def _post_json(
    client: httpx.Client, judge_config: JudgeConfig, label: str, path: str, payload: dict[str, Any]
) -> Any:
    try:
        response = client.post(path, json=payload)
        response.raise_for_status()
        return response.json()
    except (httpx.RequestError, httpx.HTTPStatusError, ValueError) as e:
        _judge_unavailable(judge_config, f"Could not get a verdict from {label} judge: {e}")


@pytest.fixture(scope="session")
def driftgate_evaluator(driftgate_config: DriftGateConfig) -> SemanticComparer:
    """Judge that decides whether a response is a valid substitute for a golden one.

    ``deterministic`` (default) uses the free offline embedding judge. Any LLM
    provider that is unreachable, unauthenticated or unknown fails the test
    instead of passing or silently skipping it.
    """
    judge_config = driftgate_config.judge
    provider = judge_config.provider

    if provider == "deterministic":
        return EmbeddingSimilarityJudge(threshold=judge_config.similarity_threshold)

    if provider == "ollama":
        client = httpx.Client(
            base_url=judge_config.ollama.base_url, timeout=judge_config.ollama.timeout
        )

        def transport(prompt: str) -> str:
            data = _post_json(client, judge_config, "Ollama", "/api/generate", {
                "model": judge_config.ollama.model, "prompt": prompt, "stream": False,
            })
            return data.get("response", "")

    elif provider == "anthropic":
        client = httpx.Client(
            base_url="https://api.anthropic.com/v1",
            timeout=judge_config.anthropic.timeout,
            headers={
                "x-api-key": _require_api_key(judge_config, "Anthropic"),
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )

        def transport(prompt: str) -> str:
            data = _post_json(client, judge_config, "Anthropic", "/messages", {
                "model": judge_config.anthropic.model,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
            })
            return " ".join(block.get("text", "") for block in data.get("content", []))

    elif provider == "openai":
        client = httpx.Client(
            base_url="https://api.openai.com/v1",
            timeout=judge_config.openai.timeout,
            headers={
                "Authorization": f"Bearer {_require_api_key(judge_config, 'OpenAI')}",
                "Content-Type": "application/json",
            },
        )

        def transport(prompt: str) -> str:
            data = _post_json(client, judge_config, "OpenAI", "/chat/completions", {
                "model": judge_config.openai.model,
                "messages": [{"role": "user", "content": prompt}],
            })
            choices = data.get("choices", [])
            return choices[0].get("message", {}).get("content", "") if choices else ""

    elif provider == "google":
        client = httpx.Client(
            base_url="https://generativelanguage.googleapis.com/v1beta",
            timeout=judge_config.google.timeout,
            headers={
                "x-goog-api-key": _require_api_key(judge_config, "Google"),
                "Content-Type": "application/json",
            },
        )

        def transport(prompt: str) -> str:
            data = _post_json(
                client, judge_config, "Google",
                f"/models/{judge_config.google.model}:generateContent",
                {"contents": [{"parts": [{"text": prompt}]}]},
            )
            candidates = data.get("candidates", [])
            parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
            return " ".join(part.get("text", "") for part in parts)

    else:
        raise pytest.UsageError(f"Unsupported judge provider {provider!r} for driftgate_evaluator.")

    return SemanticSimilarityJudge(transport=transport)


@pytest.fixture
def driftgate_agent_contract(request: pytest.FixtureRequest) -> AgentContract:
    """Loads and validates an agent contract from a YAML file specified by a marker."""
    marker = request.node.get_closest_marker("agent_contract")
    if not marker or not marker.args or not isinstance(marker.args[0], str):
        raise pytest.UsageError(
            "@pytest.mark.agent_contract requires one string argument for the file path."
        )
    contract_path = Path(marker.args[0])
    if not contract_path.exists():
        raise pytest.UsageError(f"Agent contract not found at: {contract_path}")

    try:
        return AgentContract.from_yaml(contract_path)
    except Exception as e:
        raise pytest.UsageError(f"Could not load or parse agent contract {contract_path}: {e}") from e


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