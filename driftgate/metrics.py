"""DeepEval-style metrics: ``TestCase`` + ``Metric.measure`` + ``assert_test``.

Two families, one interface:

* Deterministic metrics (``ToolCorrectness``, ``HandoffCorrectness``,
  ``StepEfficiency``) read the agent trace; free, so they gate every PR.
* Rubric metrics (``AnswerRelevancy``, ``Faithfulness``, ``TaskCompletion``,
  ``GEval``) ask an injected ``Judge`` a set of atomic binary questions and
  score the fraction that pass. They keep this framework's rule of no numeric
  LLM scales, and they require an explicit judge: there is no silent default
  that could pass everything.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from driftgate.agents.trace import AgentTrace
from driftgate.metamorphic.evaluator import Judge


@dataclass
class TestCase:
    __test__ = False  # not a pytest test class

    input: str
    actual_output: str
    expected_output: str | None = None
    context: Sequence[str] = ()
    expected_tools: Sequence[str] | None = None
    expected_path: Sequence[str] | None = None
    expected_steps: int | None = None
    trace: AgentTrace | None = None


@dataclass(frozen=True)
class MetricResult:
    name: str
    score: float
    threshold: float
    reason: str

    @property
    def passed(self) -> bool:
        return self.score >= self.threshold


class Metric(Protocol):
    name: str
    threshold: float

    def measure(self, case: TestCase) -> MetricResult: ...


def _lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a, 1):
        for j, y in enumerate(b, 1):
            table[i][j] = table[i - 1][j - 1] + 1 if x == y else max(table[i - 1][j], table[i][j - 1])
    return table[-1][-1]


@dataclass
class ToolCorrectness:
    """F1 of the tools actually called vs ``expected_tools``."""

    threshold: float = 1.0
    name: str = "ToolCorrectness"

    def measure(self, case: TestCase) -> MetricResult:
        if case.trace is None or case.expected_tools is None:
            return MetricResult(self.name, 0.0, self.threshold, "needs trace and expected_tools")
        called, expected = set(case.trace.tool_names), set(case.expected_tools)
        if not called and not expected:
            return MetricResult(self.name, 1.0, self.threshold, "no tools expected or called")
        hit = len(called & expected)
        precision = hit / len(called) if called else 0.0
        recall = hit / len(expected) if expected else 1.0
        score = 0.0 if hit == 0 else 2 * precision * recall / (precision + recall)
        return MetricResult(
            self.name, score, self.threshold,
            f"called={sorted(called)} expected={sorted(expected)}",
        )


@dataclass
class HandoffCorrectness:
    """Order-aware similarity between the visited agent path and ``expected_path``."""

    threshold: float = 1.0
    name: str = "HandoffCorrectness"

    def measure(self, case: TestCase) -> MetricResult:
        if case.trace is None or case.expected_path is None:
            return MetricResult(self.name, 0.0, self.threshold, "needs trace and expected_path")
        actual, expected = case.trace.agents_visited, list(case.expected_path)
        longest = max(len(actual), len(expected))
        score = 1.0 if longest == 0 else _lcs_len(actual, expected) / longest
        return MetricResult(self.name, score, self.threshold, f"visited={actual} expected={expected}")


@dataclass
class StepEfficiency:
    """1.0 at or under ``expected_steps``; decays as the run takes extra steps."""

    threshold: float = 0.8
    name: str = "StepEfficiency"

    def measure(self, case: TestCase) -> MetricResult:
        if case.trace is None or not case.expected_steps:
            return MetricResult(self.name, 0.0, self.threshold, "needs trace and expected_steps")
        actual = len(case.trace.steps)
        score = min(1.0, case.expected_steps / actual) if actual else 0.0
        return MetricResult(self.name, score, self.threshold, f"{actual} steps, expected {case.expected_steps}")


@dataclass
class RubricMetric:
    """Score = fraction of atomic binary rubrics the judge passes."""

    judge: Judge
    rubrics: Sequence[str]
    name: str = "RubricMetric"
    threshold: float = 0.8
    uses_context: bool = False

    def _instruction(self, case: TestCase) -> str:
        parts = [f"User input:\n{case.input}"]
        if self.uses_context and case.context:
            parts.append("Context:\n" + "\n".join(f"- {c}" for c in case.context))
        if case.expected_output is not None:
            parts.append(f"Expected output:\n{case.expected_output}")
        return "\n\n".join(parts)

    def measure(self, case: TestCase) -> MetricResult:
        instruction = self._instruction(case)
        verdicts = [
            self.judge.score(instruction, case.actual_output, rubric, rubric_id=f"{self.name}[{i}]")
            for i, rubric in enumerate(self.rubrics)
        ]
        passed = sum(v.passed for v in verdicts)
        score = passed / len(verdicts) if verdicts else 0.0
        failed = [f"{self.rubrics[i]} -> {v.rationale.splitlines()[-1] if v.rationale else ''}"
                  for i, v in enumerate(verdicts) if not v.passed]
        reason = f"{passed}/{len(verdicts)} rubrics passed" + (f"; failed: {failed}" if failed else "")
        return MetricResult(self.name, score, self.threshold, reason)


def AnswerRelevancy(judge: Judge, threshold: float = 0.8) -> RubricMetric:  # noqa: N802
    return RubricMetric(
        judge,
        [
            "Does the output directly address what the user input asked for?",
            "Is the output free of significant content unrelated to the user input?",
        ],
        name="AnswerRelevancy",
        threshold=threshold,
    )


def Faithfulness(judge: Judge, threshold: float = 1.0) -> RubricMetric:  # noqa: N802
    return RubricMetric(
        judge,
        [
            "Is every factual claim in the output supported by the provided context?",
            "Does the output avoid contradicting the provided context?",
            "Does the output avoid inventing facts, names or numbers absent from the context?",
        ],
        name="Faithfulness",
        threshold=threshold,
        uses_context=True,
    )


def TaskCompletion(judge: Judge, threshold: float = 0.8) -> RubricMetric:  # noqa: N802
    return RubricMetric(
        judge,
        [
            "Does the output fully accomplish the task in the user input rather than only part of it?",
            "Does the output present a final answer rather than a plan or a deferral?",
        ],
        name="TaskCompletion",
        threshold=threshold,
    )


def GEval(name: str, criteria: Sequence[str], judge: Judge, threshold: float = 0.8) -> RubricMetric:  # noqa: N802
    """Custom metric: each criterion is one atomic yes/no question."""
    return RubricMetric(judge, list(criteria), name=name, threshold=threshold, uses_context=True)


@dataclass(frozen=True)
class EvaluationReport:
    results: list[tuple[TestCase, list[MetricResult]]]

    @property
    def passed(self) -> bool:
        return all(r.passed for _, rs in self.results for r in rs)

    def summary(self) -> str:
        lines: list[str] = []
        for i, (_, rs) in enumerate(self.results):
            for r in rs:
                mark = "PASS" if r.passed else "FAIL"
                lines.append(
                    f"[{mark}] case {i} {r.name}: {r.score:.2f} (>= {r.threshold:.2f}) {r.reason}"
                )
        return "\n".join(lines)


def evaluate(cases: Sequence[TestCase], metrics: Sequence[Metric]) -> EvaluationReport:
    return EvaluationReport([(c, [m.measure(c) for m in metrics]) for c in cases])


def assert_test(case: TestCase, metrics: Sequence[Metric]) -> None:
    """Raise ``AssertionError`` with per-metric reasons if any metric is below threshold."""
    report = evaluate([case], metrics)
    if not report.passed:
        raise AssertionError("metric(s) below threshold:\n" + report.summary())
