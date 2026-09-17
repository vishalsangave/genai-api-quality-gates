"""Tri-state PROMOTE / HOLD / ROLLBACK release decision engine."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

import numpy as np

from driftgate.governor.statistics import CIResult, MKResult, bootstrap_ci, mann_kendall


class GateDecision(StrEnum):
    PROMOTE = "PROMOTE"
    HOLD = "HOLD"
    ROLLBACK = "ROLLBACK"


class MetricDirection(StrEnum):
    """How larger numeric values map to user-facing quality."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


@dataclass(frozen=True)
class GovernorThresholds:
    # Hard boundaries.
    safety_pass_rate_min: float = 1.0
    task_success_min: float = 0.80
    p95_latency_ms_max: float = 15_000.0

    # Five-dimensional matrix additions. They stay advisory unless explicitly
    # promoted to hard gates by a service's configuration.
    context_preservation_min: float = 0.90
    context_preservation_hard: bool = False
    evidence_coverage_min: float = 0.85
    evidence_coverage_hard: bool = False

    ci_level: float = 0.95
    ci_overlap_tolerance: float = 0.0
    trend_alpha: float = 0.05
    min_history_runs: int = 4
    rng_seed: int = 20260917


@dataclass(frozen=True)
class RunMetrics:
    """Metrics collected for one candidate or historical release run."""

    run_id: str
    safety_pass_rate: float | None
    task_success_rate: float | None
    p95_latency_ms: float | None
    latency_samples: Sequence[float] = ()
    task_success_samples: Sequence[float] = ()
    context_preservation: float | None = None
    evidence_coverage: float | None = None


@dataclass
class GateReport:
    decision: GateDecision
    reasons: list[str] = field(default_factory=list)
    hard_failures: list[str] = field(default_factory=list)
    soft_signals: list[str] = field(default_factory=list)
    latency_ci: CIResult | None = None
    success_ci: CIResult | None = None
    trends: dict[str, MKResult] = field(default_factory=dict)


@dataclass(frozen=True)
class _SoftMetric:
    key: Literal["p95_latency_ms", "task_success_rate", "context_preservation", "evidence_coverage"]
    label: str
    direction: MetricDirection
    current: float | None
    baseline: float | None
    history: list[float]


def _worse_than(left: float, right: float, direction: MetricDirection, tolerance: float) -> bool:
    return (
        left > right + tolerance
        if direction is MetricDirection.LOWER_IS_BETTER
        else left < right - tolerance
    )


def _better_or_equal(
    left: float, right: float, direction: MetricDirection, tolerance: float
) -> bool:
    return (
        left <= right + tolerance
        if direction is MetricDirection.LOWER_IS_BETTER
        else left >= right - tolerance
    )


class ReleaseGovernor:
    """Evaluate hard boundaries first, then distribution/trend soft signals.

    Decision precedence:

    1. a hard boundary breach → ROLLBACK;
    2. a confidence interval entirely worse than baseline → ROLLBACK;
    3. CI ambiguity or a significant trend toward worse → HOLD;
    4. no adverse evidence → PROMOTE.

    Critically, all comparison logic is direction-aware. A falling latency
    trend is good, whereas a falling task-success trend is not.
    """

    def __init__(
        self,
        thresholds: GovernorThresholds | None = None,
        *,
        history: Sequence[RunMetrics] = (),
        baseline: RunMetrics | None = None,
    ) -> None:
        self.thresholds = thresholds or GovernorThresholds()
        self.history = list(history)
        self.baseline = baseline

    def _hard_failures(self, current: RunMetrics) -> list[str]:
        t = self.thresholds
        failures: list[str] = []
        if current.safety_pass_rate is None or current.safety_pass_rate < t.safety_pass_rate_min:
            failures.append(f"Safety pass rate must be 100%; got {current.safety_pass_rate!r}")
        if current.task_success_rate is None or current.task_success_rate < t.task_success_min:
            failures.append(
                f"Task success rate must be >= {t.task_success_min:.0%}; got {current.task_success_rate!r}"
            )
        if current.p95_latency_ms is None or current.p95_latency_ms > t.p95_latency_ms_max:
            failures.append(
                f"P95 latency must be <= {t.p95_latency_ms_max:.0f}ms; got {current.p95_latency_ms!r}"
            )
        if t.context_preservation_hard and (
            current.context_preservation is None
            or current.context_preservation < t.context_preservation_min
        ):
            failures.append(
                "Context preservation hard gate must be "
                f">= {t.context_preservation_min:.0%}; got {current.context_preservation!r}"
            )
        if t.evidence_coverage_hard and (
            current.evidence_coverage is None or current.evidence_coverage < t.evidence_coverage_min
        ):
            failures.append(
                "Evidence coverage hard gate must be "
                f">= {t.evidence_coverage_min:.0%}; got {current.evidence_coverage!r}"
            )
        return failures

    def _build_soft_metrics(self, current: RunMetrics) -> list[_SoftMetric]:
        history = self.history
        baseline = self.baseline
        return [
            _SoftMetric(
                "p95_latency_ms",
                "P95 latency",
                MetricDirection.LOWER_IS_BETTER,
                current.p95_latency_ms,
                baseline.p95_latency_ms if baseline else None,
                [run.p95_latency_ms for run in history if run.p95_latency_ms is not None],
            ),
            _SoftMetric(
                "task_success_rate",
                "Task success rate",
                MetricDirection.HIGHER_IS_BETTER,
                current.task_success_rate,
                baseline.task_success_rate if baseline else None,
                [run.task_success_rate for run in history if run.task_success_rate is not None],
            ),
            _SoftMetric(
                "context_preservation",
                "Context preservation",
                MetricDirection.HIGHER_IS_BETTER,
                current.context_preservation,
                baseline.context_preservation if baseline else None,
                [
                    run.context_preservation
                    for run in history
                    if run.context_preservation is not None
                ],
            ),
            _SoftMetric(
                "evidence_coverage",
                "Evidence coverage",
                MetricDirection.HIGHER_IS_BETTER,
                current.evidence_coverage,
                baseline.evidence_coverage if baseline else None,
                [run.evidence_coverage for run in history if run.evidence_coverage is not None],
            ),
        ]

    def _metric_ci(
        self, current: RunMetrics, metric: _SoftMetric, rng: np.random.Generator
    ) -> CIResult | None:
        t = self.thresholds
        if metric.key == "p95_latency_ms" and current.latency_samples:
            return bootstrap_ci(current.latency_samples, level=t.ci_level, rng=rng)
        if metric.key == "task_success_rate" and current.task_success_samples:
            return bootstrap_ci(current.task_success_samples, level=t.ci_level, rng=rng)
        if metric.current is not None:
            # A scalar metric has no distribution; representing it as a
            # degenerate CI makes that lack of statistical evidence explicit.
            return CIResult(metric.current, metric.current, metric.current, t.ci_level, 1, True)
        return None

    def evaluate(self, current: RunMetrics) -> GateReport:
        hard_failures = self._hard_failures(current)
        if hard_failures:
            return GateReport(
                decision=GateDecision.ROLLBACK,
                reasons=list(hard_failures),
                hard_failures=hard_failures,
            )

        t = self.thresholds
        rng = np.random.default_rng(t.rng_seed)
        report = GateReport(decision=GateDecision.PROMOTE)
        hard_statistical_regression = False
        ambiguous_ci = False

        for metric in self._build_soft_metrics(current):
            metric_ci = self._metric_ci(current, metric, rng)
            if metric.key == "p95_latency_ms":
                report.latency_ci = metric_ci
            elif metric.key == "task_success_rate":
                report.success_ci = metric_ci

            if metric.current is None:
                continue
            if metric.baseline is not None and metric_ci is not None:
                # A scalar baseline is explicitly represented as a degenerate
                # interval. Baseline samples can be added in a future history
                # store without changing the decision API.
                base_ci = CIResult(
                    metric.baseline,
                    metric.baseline,
                    metric.baseline,
                    t.ci_level,
                    1,
                    True,
                )
                if metric.direction is MetricDirection.LOWER_IS_BETTER:
                    entirely_worse = metric_ci.low > base_ci.high + t.ci_overlap_tolerance
                    entirely_better = metric_ci.high <= base_ci.low + t.ci_overlap_tolerance
                else:
                    entirely_worse = metric_ci.high < base_ci.low - t.ci_overlap_tolerance
                    entirely_better = metric_ci.low >= base_ci.high - t.ci_overlap_tolerance
                if entirely_worse:
                    message = f"{metric.label} CI [{metric_ci.low:.4g}, {metric_ci.high:.4g}] is entirely worse than baseline {metric.baseline:.4g}"
                    report.soft_signals.append(message)
                    hard_statistical_regression = True
                elif not entirely_better:
                    report.soft_signals.append(
                        f"{metric.label} CI [{metric_ci.low:.4g}, {metric_ci.high:.4g}] overlaps baseline {metric.baseline:.4g}"
                    )
                    ambiguous_ci = True

            complete_history = metric.history + [metric.current]
            mk = mann_kendall(
                complete_history,
                alpha=t.trend_alpha,
                min_n=t.min_history_runs,
            )
            report.trends[metric.key] = mk
            trend_is_worse = (
                metric.direction is MetricDirection.LOWER_IS_BETTER and mk.trend == "increasing"
            ) or (metric.direction is MetricDirection.HIGHER_IS_BETTER and mk.trend == "decreasing")
            if trend_is_worse:
                report.soft_signals.append(
                    f"{metric.label} shows significant trend toward worse ({mk.trend}, p={mk.p_value:.4g})"
                )

        if hard_statistical_regression:
            report.decision = GateDecision.ROLLBACK
            report.reasons = ["Systemic statistical regression"] + report.soft_signals
        elif ambiguous_ci or report.soft_signals:
            report.decision = GateDecision.HOLD
            report.reasons = ["Statistical or trend evidence needs review"] + report.soft_signals
        else:
            report.decision = GateDecision.PROMOTE
            report.reasons = ["All hard boundaries pass; no adverse statistical or trend evidence"]
        return report
