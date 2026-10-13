"""Behavioural assertions over a multi-agent trace: budgets, loops, routing."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from driftgate.agents.contracts import ContractRegistry, Violation
from driftgate.agents.trace import AgentTrace


@dataclass(frozen=True)
class TraceBudget:
    """Hard limits for one run. ``None`` means unchecked."""

    max_steps: int | None = None
    max_tool_calls: int | None = None
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_latency_ms: float | None = None
    max_repeats: int | None = 3  # same agent + identical input this many times = loop


def find_loops(trace: AgentTrace, max_repeats: int) -> list[Violation]:
    """Flag an agent re-run with identical input, and cyclic handoffs (A->B->A)."""
    violations: list[Violation] = []
    seen: Counter[tuple[str, str]] = Counter()
    for step in trace.steps:
        seen[(step.agent, repr(step.input))] += 1
    for (agent, _), count in seen.items():
        if count > max_repeats:
            violations.append(
                Violation(agent, "loop", f"ran {count} times with identical input (limit {max_repeats})")
            )
    edges = trace.handoffs
    for i, (src, dst) in enumerate(edges):
        if (dst, src) in edges[:i]:
            violations.append(Violation(src, "loop", f"handoff cycle {dst}->{src}->{dst}"))
            break
    return violations


def check_trace(
    trace: AgentTrace,
    *,
    budget: TraceBudget | None = None,
    registry: ContractRegistry | None = None,
    expected_path: Iterable[str] | None = None,
    required_tools: Iterable[str] = (),
    forbidden_tools: Iterable[str] = (),
    allowed_edges: Iterable[tuple[str, str]] | None = None,
) -> list[Violation]:
    violations: list[Violation] = []
    b = budget or TraceBudget()

    def over(rule: str, actual: float, limit: float | None) -> None:
        if limit is not None and actual > limit:
            violations.append(Violation("<run>", rule, f"{actual:g} exceeds limit {limit:g}"))

    over("max_steps", len(trace.steps), b.max_steps)
    over("max_tool_calls", len(trace.tool_calls), b.max_tool_calls)
    over("max_tokens", trace.total_tokens, b.max_tokens)
    over("max_cost_usd", trace.total_cost_usd, b.max_cost_usd)
    over("max_latency_ms", trace.total_latency_ms, b.max_latency_ms)
    if b.max_repeats is not None:
        violations += find_loops(trace, b.max_repeats)

    if registry is not None:
        violations += registry.check_trace(trace)

    if expected_path is not None and trace.agents_visited != list(expected_path):
        violations.append(
            Violation(
                "<run>", "expected_path", f"visited {trace.agents_visited}, expected {list(expected_path)}"
            )
        )

    called = set(trace.tool_names)
    for tool in required_tools:
        if tool not in called:
            violations.append(Violation("<run>", "required_tools", f"{tool!r} was never called"))
    banned = set(forbidden_tools)
    for step in trace.steps:
        for call in step.tool_calls:
            if call.name in banned:
                violations.append(Violation(step.agent, "forbidden_tools", f"called {call.name!r}"))

    if allowed_edges is not None:
        permitted = set(allowed_edges)
        for src, dst in trace.handoffs:
            if (src, dst) not in permitted:
                violations.append(Violation(src, "allowed_edges", f"handoff {src}->{dst} not permitted"))
    return violations


def assert_trace(trace: AgentTrace, **kwargs) -> None:
    """Raise ``AssertionError`` listing every violation (same kwargs as ``check_trace``)."""
    violations = check_trace(trace, **kwargs)
    if violations:
        raise AssertionError(
            f"{len(violations)} agent-trace violation(s):\n" + "\n".join(f"  - {v}" for v in violations)
        )
