"""DriftGate — API quality gates for non-deterministic GenAI microservices.

Three decoupled layers:

1. Deterministic contract gate (``driftgate.contracts``) — sub-millisecond
   OpenAPI 3.1 / JSON Schema validation, zero token cost.
2. AST-aware cassette replay gate (``driftgate.recorder``) — offline
   record/replay of HTTP and Server-Sent Events traffic, fingerprinted on a
   normalized request shape rather than raw bytes.
3. Probabilistic & metamorphic gate (``driftgate.metamorphic``,
   ``driftgate.streaming``) — invariance checks, LLM-judge rubrics, and a
   streaming telemetry probe for TTFT/ITL SLAs.

A statistical release governor (``driftgate.governor``) turns run telemetry
into a PROMOTE / HOLD / ROLLBACK decision using bootstrap confidence
intervals and Mann-Kendall trend analysis, rather than brittle binary
thresholds.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
