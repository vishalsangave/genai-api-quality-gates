from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
import yaml
from pydantic import BaseModel, Field

from driftgate.agents.trace import AgentTrace


@dataclass(frozen=True)
class Violation:
    """One broken rule, attributed to an agent so 20+ agents stay debuggable."""

    agent: str
    rule: str
    message: str

    def __str__(self) -> str:
        return f"[{self.agent}] {self.rule}: {self.message}"


class AgentContract(BaseModel):
    name: str
    version: str
    description: str
    allowed_tools: list[str] = Field(default_factory=list)
    allowed_handoffs: list[str] | None = None  # None = unrestricted
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AgentContract:
        """Load and validate an agent contract from a YAML file."""
        raw = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(raw)

    def validate_input(self, payload: Any) -> list[Violation]:
        return self._validate(payload, self.input_schema, "input_schema")

    def validate_output(self, payload: Any) -> list[Violation]:
        return self._validate(payload, self.output_schema, "output_schema")

    def _validate(self, payload: Any, schema: dict[str, Any], rule: str) -> list[Violation]:
        if not schema:
            return []
        validator = jsonschema.Draft202012Validator(schema)
        return [
            Violation(self.name, rule, f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}")
            for e in sorted(validator.iter_errors(payload), key=lambda e: list(e.absolute_path))
        ]

    def check_trace(self, trace: AgentTrace) -> list[Violation]:
        """Enforce this contract on every step the agent took in a trace."""
        violations: list[Violation] = []
        for step in trace.steps_for(self.name):
            if step.input is not None:
                violations += self.validate_input(step.input)
            if step.output is not None:
                violations += self.validate_output(step.output)
            for call in step.tool_calls:
                if call.name not in self.allowed_tools:
                    violations.append(
                        Violation(self.name, "allowed_tools", f"called disallowed tool {call.name!r}")
                    )
            if (
                self.allowed_handoffs is not None
                and step.handoff_to is not None
                and step.handoff_to not in self.allowed_handoffs
            ):
                violations.append(
                    Violation(
                        self.name, "allowed_handoffs", f"handed off to disallowed agent {step.handoff_to!r}"
                    )
                )
        return violations


def check_compatibility(producer: AgentContract, consumer: AgentContract) -> list[Violation]:
    """Can ``consumer`` safely be fed ``producer``'s output?

    Every property the consumer requires must be declared by the producer, and
    where both declare a ``type`` they must agree. Catches breaking contract
    changes between independently versioned agents before they hit runtime.
    """
    produced = producer.output_schema.get("properties", {})
    produced_required = set(producer.output_schema.get("required", []))
    violations: list[Violation] = []
    consumer_props = consumer.input_schema.get("properties", {})
    for key in consumer.input_schema.get("required", []):
        edge = f"{producer.name}->{consumer.name}"
        if key not in produced:
            violations.append(
                Violation(edge, "compatibility", f"consumer requires {key!r} but producer never declares it")
            )
            continue
        if key not in produced_required:
            violations.append(
                Violation(edge, "compatibility", f"consumer requires {key!r} but producer may omit it")
            )
        want, have = consumer_props.get(key, {}).get("type"), produced[key].get("type")
        if want and have and want != have:
            violations.append(
                Violation(edge, "compatibility", f"{key!r} is {have!r} from producer, {want!r} required")
            )
    return violations


class ContractRegistry:
    """All agent contracts for a system; the unit that scales to 20+ agents."""

    def __init__(self, contracts: list[AgentContract] | None = None) -> None:
        self._contracts: dict[str, AgentContract] = {}
        for contract in contracts or []:
            self.add(contract)

    @classmethod
    def from_dir(cls, directory: str | Path) -> ContractRegistry:
        root = Path(directory)
        paths = sorted([*root.glob("*.yaml"), *root.glob("*.yml")])
        return cls([AgentContract.from_yaml(p) for p in paths])

    def add(self, contract: AgentContract) -> None:
        if contract.name in self._contracts:
            raise ValueError(f"duplicate agent contract: {contract.name!r}")
        self._contracts[contract.name] = contract

    def __contains__(self, name: str) -> bool:
        return name in self._contracts

    def __getitem__(self, name: str) -> AgentContract:
        return self._contracts[name]

    @property
    def names(self) -> list[str]:
        return sorted(self._contracts)

    def check_trace(self, trace: AgentTrace, *, require_known_agents: bool = True) -> list[Violation]:
        """Validate a whole multi-agent trace against every agent's contract."""
        violations: list[Violation] = []
        for name in dict.fromkeys(step.agent for step in trace.steps):
            if name not in self._contracts:
                if require_known_agents:
                    violations.append(Violation(name, "unknown_agent", "no contract registered"))
                continue
            violations += self._contracts[name].check_trace(trace)
        for src, dst in trace.handoffs:
            if dst not in self._contracts and require_known_agents:
                violations.append(Violation(src, "unknown_agent", f"handed off to unregistered agent {dst!r}"))
        return violations

    def check_graph(self) -> list[Violation]:
        """Static check of every declared handoff edge for schema compatibility."""
        violations: list[Violation] = []
        for contract in self._contracts.values():
            for target in contract.allowed_handoffs or []:
                if target not in self._contracts:
                    violations.append(
                        Violation(contract.name, "unknown_agent", f"allowed_handoffs names unknown agent {target!r}")
                    )
                    continue
                violations += check_compatibility(contract, self._contracts[target])
        return violations
