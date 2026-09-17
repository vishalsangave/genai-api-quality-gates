from __future__ import annotations

from driftgate.governor.gate import GateDecision, GovernorThresholds, ReleaseGovernor, RunMetrics


def _healthy(**overrides) -> RunMetrics:
    values = {
        "run_id": "candidate",
        "safety_pass_rate": 1.0,
        "task_success_rate": 0.90,
        "p95_latency_ms": 900.0,
        "latency_samples": [800.0, 900.0, 950.0, 1000.0],
        "task_success_samples": [1.0] * 9 + [0.0],
    }
    values.update(overrides)
    return RunMetrics(**values)


def test_each_hard_boundary_rolls_back() -> None:
    for change in (
        {"safety_pass_rate": 0.99},
        {"task_success_rate": 0.79},
        {"p95_latency_ms": 15_001.0},
        {"safety_pass_rate": None},
    ):
        report = ReleaseGovernor().evaluate(_healthy(**change))
        assert report.decision is GateDecision.ROLLBACK
        assert report.hard_failures


def test_healthy_run_promotes_without_history_or_baseline() -> None:
    report = ReleaseGovernor().evaluate(_healthy())
    assert report.decision is GateDecision.PROMOTE


def test_confidence_interval_straddling_baseline_holds() -> None:
    baseline = _healthy(run_id="baseline", p95_latency_ms=900.0)
    candidate = _healthy(latency_samples=[850.0, 900.0, 950.0, 1000.0])
    report = ReleaseGovernor(baseline=baseline).evaluate(candidate)

    assert report.decision is GateDecision.HOLD
    assert any("overlaps baseline" in message for message in report.soft_signals)


def test_direction_aware_systemic_latency_regression_rolls_back() -> None:
    baseline = _healthy(run_id="baseline", p95_latency_ms=900.0)
    candidate = _healthy(p95_latency_ms=2000.0, latency_samples=[1900.0, 2000.0, 2100.0, 2200.0])
    report = ReleaseGovernor(baseline=baseline).evaluate(candidate)

    assert report.decision is GateDecision.ROLLBACK
    assert any("P95 latency CI" in message for message in report.soft_signals)


def test_worsening_latency_trend_holds() -> None:
    history = [
        _healthy(run_id=str(index), p95_latency_ms=latency)
        for index, latency in enumerate([100, 200, 300, 400, 500])
    ]
    candidate = _healthy(p95_latency_ms=600.0)
    report = ReleaseGovernor(history=history).evaluate(candidate)

    assert report.decision is GateDecision.HOLD
    assert report.trends["p95_latency_ms"].trend == "increasing"


def test_advisory_dimensions_can_be_promoted_to_hard() -> None:
    candidate = _healthy(context_preservation=0.5, evidence_coverage=0.5)
    advisory = ReleaseGovernor().evaluate(candidate)
    assert advisory.decision is GateDecision.PROMOTE

    thresholds = GovernorThresholds(context_preservation_hard=True, evidence_coverage_hard=True)
    enforced = ReleaseGovernor(thresholds).evaluate(candidate)
    assert enforced.decision is GateDecision.ROLLBACK
