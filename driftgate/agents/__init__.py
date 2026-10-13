"""Multi-agent testing: traces, contracts, and trace-level assertions."""

from driftgate.agents.assertions import TraceBudget, assert_trace, check_trace
from driftgate.agents.capture import TraceCollector, attach, extract_trace
from driftgate.agents.contracts import (
    AgentContract,
    ContractRegistry,
    Violation,
    check_compatibility,
)
from driftgate.agents.trace import AgentStep, AgentTrace, ToolCall, TraceRecorder

__all__ = [
    "AgentContract",
    "AgentStep",
    "AgentTrace",
    "ContractRegistry",
    "ToolCall",
    "TraceBudget",
    "TraceCollector",
    "TraceRecorder",
    "Violation",
    "assert_trace",
    "attach",
    "check_compatibility",
    "check_trace",
    "extract_trace",
]
