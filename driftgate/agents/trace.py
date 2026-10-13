"""Trace model for multi-agent runs.

A trace is the ordered list of steps an orchestrator and its agents took to
answer one task: which agent ran, which tools it called, and who it handed off
to. Traces serialize to JSON so a whole multi-agent run can be recorded once
and replayed in CI, which a per-HTTP-call cassette cannot do when 20 agents
call each other in a non-deterministic order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    name: str
    input: Any = None
    output: Any = None
    error: str | None = None
    latency_ms: float = 0.0


class AgentStep(BaseModel):
    agent: str
    input: Any = None
    output: Any = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    handoff_to: str | None = None
    tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0


class AgentTrace(BaseModel):
    task: str = ""
    steps: list[AgentStep] = Field(default_factory=list)
    final_output: Any = None

    @property
    def agents_visited(self) -> list[str]:
        """Agents in execution order, with consecutive repeats collapsed."""
        path: list[str] = []
        for step in self.steps:
            if not path or path[-1] != step.agent:
                path.append(step.agent)
        return path

    @property
    def handoffs(self) -> list[tuple[str, str]]:
        """Every (from_agent, to_agent) edge: explicit handoffs, else step order."""
        edges: list[tuple[str, str]] = []
        for i, step in enumerate(self.steps):
            if step.handoff_to:
                edges.append((step.agent, step.handoff_to))
            elif i + 1 < len(self.steps) and self.steps[i + 1].agent != step.agent:
                edges.append((step.agent, self.steps[i + 1].agent))
        return edges

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [call for step in self.steps for call in step.tool_calls]

    @property
    def tool_names(self) -> list[str]:
        return [call.name for call in self.tool_calls]

    @property
    def total_tokens(self) -> int:
        return sum(step.tokens for step in self.steps)

    @property
    def total_cost_usd(self) -> float:
        return sum(step.cost_usd for step in self.steps)

    @property
    def total_latency_ms(self) -> float:
        return sum(step.latency_ms for step in self.steps)

    def steps_for(self, agent: str) -> list[AgentStep]:
        return [step for step in self.steps if step.agent == agent]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> AgentTrace:
        return cls.model_validate(json.loads(Path(path).read_text()))


class TraceRecorder:
    """Incrementally builds an ``AgentTrace`` from instrumented agent code."""

    def __init__(self, task: str = "") -> None:
        self.trace = AgentTrace(task=task)

    def step(
        self,
        agent: str,
        *,
        input: Any = None,
        output: Any = None,
        tool_calls: list[ToolCall] | None = None,
        handoff_to: str | None = None,
        tokens: int = 0,
        cost_usd: float = 0.0,
        latency_ms: float = 0.0,
    ) -> AgentStep:
        step = AgentStep(
            agent=agent,
            input=input,
            output=output,
            tool_calls=tool_calls or [],
            handoff_to=handoff_to,
            tokens=tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )
        self.trace.steps.append(step)
        return step

    def finish(self, final_output: Any = None) -> AgentTrace:
        self.trace.final_output = final_output
        return self.trace
