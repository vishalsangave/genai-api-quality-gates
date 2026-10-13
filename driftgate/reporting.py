"""Per-build QA report: endpoint attribution, outcomes, coverage, spec diff.

The pytest plugin installs a pass-through wrapper on ``httpx.Client.send``
for the duration of a session, attributing every request to the currently
running test and bucketing concrete paths into spec templates. At session
finish the tracker, per-test outcomes, spec coverage, the governor decision
(if metrics exist), and the spec diff (if a spec is configured) are rendered
into ``.driftgate/report.json`` and a QA-facing ``report.md``.

Failure contract: reporting never changes the pytest exit code. Every hook
body and the whole sessionfinish pipeline swallow exceptions into a stderr
warning. See docs/ARCHITECTURE.md "Build report".
"""

from __future__ import annotations

import json
import os
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

REPORT_DIR = Path(".driftgate")
REPORT_JSON = REPORT_DIR / "report.json"
REPORT_MD = REPORT_DIR / "report.md"
METRICS_JSON = REPORT_DIR / "metrics.json"
SNAPSHOT_JSON = REPORT_DIR / "spec-snapshot.json"
SESSION_NODEID = "(session)"
PER_TEST_CALL_CAP = 64
MAX_TRACKED_TESTS = 5_000


def _warn(message: str) -> None:
    print(f"driftgate: {message}", file=sys.stderr)


class CallTracker:
    """Bounded per-test map of unique (method, path, status) triples."""

    def __init__(self) -> None:
        self.calls_by_test: dict[str, set[tuple[str, str, int]]] = {}
        self._lock = threading.Lock()
        self._current: threading.local = threading.local()

    def start_test(self, nodeid: str) -> None:
        self._current.nodeid = nodeid

    def finish_test(self) -> None:
        self._current.nodeid = None

    def record(self, method: str, path: str, status: int) -> None:
        nodeid = getattr(self._current, "nodeid", None) or SESSION_NODEID
        with self._lock:
            if nodeid not in self.calls_by_test and len(self.calls_by_test) >= MAX_TRACKED_TESTS:
                return
            calls = self.calls_by_test.setdefault(nodeid, set())
            if len(calls) >= PER_TEST_CALL_CAP:
                return
            calls.add((method.upper(), path, status))

    def calls_for(self, nodeid: str) -> set[tuple[str, str, int]]:
        with self._lock:
            return set(self.calls_by_test.get(nodeid, ()))

    @property
    def calls_by_test_size(self) -> int:
        with self._lock:
            return len(self.calls_by_test)


_ORIGINAL_SEND: Callable[..., httpx.Response] | None = None


def wrapper_installed() -> bool:
    """True when the class-wide send wrapper is currently active."""
    return (
        _ORIGINAL_SEND is not None
        and getattr(httpx.Client.send, "__name__", "") == "_driftgate_send"
    )


def install_httpx_wrapper(tracker: CallTracker) -> bool:
    """Patch httpx.Client.send class-wide; returns False when already installed."""
    global _ORIGINAL_SEND
    if _ORIGINAL_SEND is not None:
        return False
    original = httpx.Client.send
    _ORIGINAL_SEND = original

    def _driftgate_send(
        client: httpx.Client, request: httpx.Request, *args: Any, **kwargs: Any
    ) -> httpx.Response:
        response = original(client, request, *args, **kwargs)
        try:
            tracker.record(request.method, request.url.path, response.status_code)
        except Exception:
            pass
        return response

    httpx.Client.send = _driftgate_send  # type: ignore[method-assign]
    return True


def uninstall_httpx_wrapper() -> None:
    global _ORIGINAL_SEND
    if (
        _ORIGINAL_SEND is not None
        and getattr(httpx.Client.send, "__name__", "") == "_driftgate_send"
    ):
        httpx.Client.send = _ORIGINAL_SEND  # type: ignore[method-assign]
    _ORIGINAL_SEND = None


def reinstall_session_wrapper(session_tracker: CallTracker) -> None:
    """Undo a test-level uninstall so session-wide attribution keeps working."""
    if not wrapper_installed():
        uninstall_httpx_wrapper()
        install_httpx_wrapper(session_tracker)


@dataclass(frozen=True)
class RunTotals:
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.errors + self.skipped


@dataclass(frozen=True)
class EndpointRow:
    method: str
    template: str
    tests: int
    failed: int
    statuses_seen: tuple[int, ...]


@dataclass(frozen=True)
class SpecCoverage:
    declared: int
    touched: int
    percentage: float
    uncovered: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class GovernorSummary:
    decision: str
    reasons: tuple[str, ...]
    hard_failures: tuple[str, ...]
    soft_signals: tuple[str, ...]


@dataclass(frozen=True)
class ReportData:
    run_id: str
    timestamp: str
    exitstatus: int
    totals: RunTotals
    endpoints: tuple[EndpointRow, ...]
    coverage: SpecCoverage | None
    governor: GovernorSummary | None
    spec_diff: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "exitstatus": self.exitstatus,
            "totals": {
                "passed": self.totals.passed,
                "failed": self.totals.failed,
                "errors": self.totals.errors,
                "skipped": self.totals.skipped,
                "total": self.totals.total,
            },
            "endpoints": [
                {
                    "method": row.method,
                    "template": row.template,
                    "tests": row.tests,
                    "failed": row.failed,
                    "statuses_seen": list(row.statuses_seen),
                }
                for row in self.endpoints
            ],
            "coverage": (
                {
                    "declared": self.coverage.declared,
                    "touched": self.coverage.touched,
                    "percentage": self.coverage.percentage,
                    "uncovered": [list(pair) for pair in self.coverage.uncovered],
                }
                if self.coverage
                else None
            ),
            "governor": (
                {
                    "decision": self.governor.decision,
                    "reasons": list(self.governor.reasons),
                    "hard_failures": list(self.governor.hard_failures),
                    "soft_signals": list(self.governor.soft_signals),
                }
                if self.governor
                else None
            ),
            "spec_diff": self.spec_diff,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReportData:
        totals = data.get("totals", {})
        endpoints = tuple(
            EndpointRow(
                method=row["method"],
                template=row["template"],
                tests=row["tests"],
                failed=row["failed"],
                statuses_seen=tuple(row.get("statuses_seen", ())),
            )
            for row in data.get("endpoints", [])
        )
        coverage_raw = data.get("coverage")
        coverage = (
            SpecCoverage(
                declared=coverage_raw["declared"],
                touched=coverage_raw["touched"],
                percentage=coverage_raw["percentage"],
                uncovered=tuple(tuple(pair) for pair in coverage_raw.get("uncovered", [])),
            )
            if isinstance(coverage_raw, dict)
            else None
        )
        governor_raw = data.get("governor")
        governor = (
            GovernorSummary(
                decision=governor_raw["decision"],
                reasons=tuple(governor_raw.get("reasons", ())),
                hard_failures=tuple(governor_raw.get("hard_failures", ())),
                soft_signals=tuple(governor_raw.get("soft_signals", ())),
            )
            if isinstance(governor_raw, dict)
            else None
        )
        return cls(
            run_id=data.get("run_id", "unnamed"),
            timestamp=data.get("timestamp", ""),
            exitstatus=data.get("exitstatus", 0),
            totals=RunTotals(
                passed=totals.get("passed", 0),
                failed=totals.get("failed", 0),
                errors=totals.get("errors", 0),
                skipped=totals.get("skipped", 0),
            ),
            endpoints=endpoints,
            coverage=coverage,
            governor=governor,
            spec_diff=data.get("spec_diff"),
        )


def normalize_endpoint(method: str, path: str, validator: Any) -> str:
    """Concrete path -> spec template when a spec matches; else a coarse bucket."""
    if validator is not None:
        try:
            template = validator.template_for(method, path)
        except Exception:
            template = None
        if template is not None:
            return template
    return _heuristic_bucket(path)


def _heuristic_bucket(path: str) -> str:
    parts = [segment for segment in path.split("/") if segment]
    return "/" + "/".join(parts[:2]) if parts else "/"


def merge_outcomes(raw: dict[str, dict[str, Any]]) -> dict[str, str]:
    """nodeid -> final outcome, precedence error > failed > passed > skipped."""
    final: dict[str, str] = {}
    for nodeid, phases in raw.items():
        if phases.get("setup_failed") or phases.get("teardown_failed"):
            final[nodeid] = "error"
        else:
            final[nodeid] = phases.get("call") or "skipped"
    return final


def build_endpoint_rows(
    tracker: CallTracker,
    outcomes: dict[str, str],
    validator: Any,
) -> tuple[tuple[EndpointRow, ...], dict[tuple[str, str], set[str]]]:
    """Aggregate tracked calls into endpoint rows; also return touched (method, template) -> nodeids."""
    normalized: dict[tuple[str, str], dict[str, set[int]]] = {}
    failed_by_endpoint: dict[tuple[str, str], set[str]] = {}
    cache: dict[tuple[str, str], str] = {}
    for nodeid, calls in tracker.calls_by_test.items():
        outcome = outcomes.get(nodeid, "passed")
        for method, path, status in calls:
            key = (method, path)
            if key not in cache:
                cache[key] = normalize_endpoint(method, path, validator)
            bucket = cache[key]
            endpoint = (method, bucket)
            normalized.setdefault(endpoint, {}).setdefault(nodeid, set()).add(status)
            if outcome in ("failed", "error"):
                failed_by_endpoint.setdefault(endpoint, set()).add(nodeid)
    rows = [
        EndpointRow(
            method=method,
            template=template,
            tests=len(nodeids),
            failed=len(failed_by_endpoint.get((method, template), ())),
            statuses_seen=tuple(
                sorted({status for statuses in nodeids.values() for status in statuses})
            ),
        )
        for (method, template), nodeids in sorted(normalized.items())
    ]
    touched = {key: set(nodeids) for key, nodeids in normalized.items()}
    return tuple(rows), touched


def build_coverage(validator: Any, touched: dict[tuple[str, str], set[str]]) -> SpecCoverage | None:
    if validator is None:
        return None
    declared: set[tuple[str, str]] = set()
    for template, path_item in validator.spec.get("paths", {}).items():
        if not isinstance(path_item, dict):
            continue
        for method, definition in path_item.items():
            if method.lower() in {
                "get",
                "post",
                "put",
                "patch",
                "delete",
                "head",
                "options",
            } and isinstance(definition, dict):
                declared.add((method.upper(), template))
    touched_keys = set(touched)
    uncovered = tuple(sorted(declared - touched_keys))
    count = len(declared)
    covered = count - len(uncovered)
    return SpecCoverage(
        declared=count,
        touched=covered,
        percentage=round(covered / count * 100, 1) if count else 0.0,
        uncovered=uncovered,
    )


def evaluate_governor(config: Any) -> GovernorSummary | None:
    from driftgate.governor.gate import ReleaseGovernor, run_metrics_from_dict

    if not METRICS_JSON.exists():
        return None
    try:
        loaded = json.loads(METRICS_JSON.read_text())
        if not isinstance(loaded, dict):
            return None
        metrics = run_metrics_from_dict(loaded)
        from driftgate.cli import _load_history, _thresholds

        report = ReleaseGovernor(
            _thresholds(config.governor),
            history=_load_history(config.governor.history_path),
        ).evaluate(metrics)
        return GovernorSummary(
            decision=str(report.decision),
            reasons=tuple(report.reasons),
            hard_failures=tuple(report.hard_failures),
            soft_signals=tuple(report.soft_signals),
        )
    except Exception as error:
        _warn(f"governor section skipped: {error}")
        return None


def write_report(report: ReportData) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = REPORT_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, REPORT_JSON)
    REPORT_MD.write_text(render_markdown(report))


def render_markdown(report: ReportData) -> str:
    lines: list[str] = [
        "# DriftGate Build Report",
        "",
        f"- Run: {report.run_id}",
        f"- Finished: {report.timestamp}",
        f"- Pytest exit status: {report.exitstatus}",
        "",
        "## Test totals",
        "",
        "| Outcome | Count |",
        "|---|---|",
        f"| Passed | {report.totals.passed} |",
        f"| Failed | {report.totals.failed} |",
        f"| Errors | {report.totals.errors} |",
        f"| Skipped | {report.totals.skipped} |",
        "",
    ]
    lines.append("## Endpoints exercised")
    lines.append("")
    if report.endpoints:
        lines.append("| Method | Template | Tests | Failed | Statuses seen |")
        lines.append("|---|---|---|---|---|")
        for row in report.endpoints:
            statuses = ", ".join(str(status) for status in row.statuses_seen)
            lines.append(
                f"| {row.method} | {row.template} | {row.tests} | {row.failed} | {statuses} |"
            )
    else:
        lines.append("_No HTTP traffic was attributed to tests._")
    lines.append("")
    lines.append("## Spec coverage")
    lines.append("")
    if report.coverage is None:
        lines.append("_No OpenAPI spec configured._")
    else:
        coverage = report.coverage
        lines.append(
            f"- {coverage.touched}/{coverage.declared} declared endpoints exercised ({coverage.percentage}%)"
        )
        if coverage.uncovered:
            uncovered = ", ".join(
                f"`{method} {template}`" for method, template in coverage.uncovered
            )
            lines.append(f"- Uncovered: {uncovered}")
    lines.append("")
    lines.append("## Governor decision")
    lines.append("")
    if report.governor is None:
        lines.append("_No metrics recorded; governor not evaluated._")
    else:
        lines.append(f"**{report.governor.decision}**")
        for reason in report.governor.reasons:
            lines.append(f"- {reason}")
        for failure in report.governor.hard_failures:
            lines.append(f"- hard failure: {failure}")
        for signal in report.governor.soft_signals:
            lines.append(f"- soft signal: {signal}")
    lines.append("")
    lines.append("## Changed vs last build")
    lines.append("")
    spec_diff = report.spec_diff
    if spec_diff is None:
        lines.append("_No spec configured; no diff._")
    elif spec_diff.get("first_snapshot"):
        lines.append("First snapshot taken; diff available next build.")
    elif not any(spec_diff.get(section) for section in ("added", "removed", "changed")):
        lines.append("No changes detected.")
    else:
        if spec_diff.get("has_breaking"):
            lines.append("**Breaking changes detected:**")
        for section in ("removed", "changed", "added"):
            for endpoint in spec_diff.get(section, []):
                severity = endpoint.get("max_severity", "info")
                header = f"{endpoint.get('method')} {endpoint.get('template')}"
                lines.append(f"- [{severity}] {header} ({section})")
                for finding in endpoint.get("findings", []):
                    lines.append(f"  - {finding.get('detail')}")
    lines.append("")
    return "\n".join(lines)
