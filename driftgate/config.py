"""Declarative configuration for DriftGate.

Parses ``driftgate.config.yaml`` into a strict Pydantic v2 model hierarchy.
Every section has a sensible default, so the framework is usable with zero
configuration; a config file only needs to override what a project cares
about.

Env-var overlay: any environment variable named ``DRIFT_GATE_<SECTION>__<KEY>``
(double underscore = one level of nesting) overrides the corresponding value
after the YAML file (or the defaults, if no file exists) has been loaded and
validated. This is the *only* place raw config strings exist — every other
module consumes the validated model, never `os.environ` directly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

DEFAULT_CONFIG_FILENAME = "driftgate.config.yaml"
ENV_PREFIX = "DRIFT_GATE_"

TrafficMode = Literal["record", "replay_strict", "replay_lenient", "live", "auto_record"]

_SUPPORTED_MUTATORS: frozenset[str] = frozenset({"local"})

_SUPPORTED_RELATIONS: frozenset[str] = frozenset(
    {
        "paraphrase_invariance",
        "entity_substitution",
        "adversarial_noise_invariance",
        "retrieval_context_order_invariance",
    }
)


class ConfigError(ValueError):
    """Raised for malformed configuration that Pydantic itself won't catch."""


class ServiceConfig(BaseModel):
    """The target service under test."""

    base_url: str = "http://localhost:8000"
    endpoint: str | None = None
    method: str | None = None


class ContractConfig(BaseModel):
    """Layer 1: deterministic OpenAPI / JSON Schema contract validation."""

    openapi_spec: Path | None = None
    strict_types: bool = True
    enforce: bool = True


class StreamingSLAConfig(BaseModel):
    """Streaming telemetry probe SLA thresholds (TTFT / ITL)."""

    enabled: bool = True
    max_ttft_ms: float = 800.0
    max_itl_ms: float = 35.0
    require_done_marker: bool = True
    done_marker: str = "[DONE]"


class MetamorphicAssertionConfig(BaseModel):
    """One configured metamorphic relation to run against the service."""

    relation: str
    similarity_threshold: float = 0.90
    max_permutations: int = 3
    mutator: str | None = None

    @field_validator("relation")
    @classmethod
    def _validate_relation(cls, v: str) -> str:
        if v not in _SUPPORTED_RELATIONS:
            raise ConfigError(
                f"Unknown metamorphic relation {v!r}; supported: {sorted(_SUPPORTED_RELATIONS)}"
            )
        return v

    @field_validator("mutator")
    @classmethod
    def _validate_mutator(cls, v: str | None) -> str | None:
        if v is not None and v not in _SUPPORTED_MUTATORS:
            raise ConfigError(
                f"Unsupported mutator {v!r}; MVP only implements local deterministic "
                f"mutators ({sorted(_SUPPORTED_MUTATORS)}). A non-local mutator such as "
                f"'back_translation' would require a network call, which contradicts "
                f"the zero-token-cost PR gate guarantee."
            )
        return v


class QualityGateConfig(BaseModel):
    """Statistical-engine knobs shared by the release governor."""

    confidence_level: float = 0.95
    max_allowable_drift: float = 0.05
    on_indeterminate: Literal["HOLD", "ROLLBACK", "PROMOTE"] = "HOLD"


class JudgeConfig(BaseModel):
    """LLM-as-a-judge provider selection.

    ``provider: "deterministic"`` (the default) never makes a network call —
    this is what keeps the PR-gate path at $0.00 token cost. A real judge is
    opt-in and only activates when a provider other than "deterministic" is
    configured *and* the referenced API key environment variable resolves.
    """

    provider: Literal["deterministic", "openai", "anthropic"] = "deterministic"
    model: str | None = None
    api_key_env: str = "DRIFT_GATE_JUDGE_API_KEY"


class GovernorConfig(BaseModel):
    """Statistical Release Governor thresholds.

    Hard boundaries (safety, task success, P95 latency) always block a
    release on breach. Context preservation and evidence coverage are
    advisory dimensions from the five-dimension gate matrix, `hard=False`
    by default — they influence HOLD/PROMOTE via the soft signal path but do
    not by themselves force a ROLLBACK unless explicitly promoted to hard.
    """

    safety_pass_rate_min: float = 1.0
    task_success_min: float = 0.80
    p95_latency_ms_max: float = 15_000.0

    context_preservation_min: float = 0.90
    context_preservation_hard: bool = False
    evidence_coverage_min: float = 0.85
    evidence_coverage_hard: bool = False

    ci_level: float = 0.95
    ci_overlap_tolerance: float = 0.0
    trend_alpha: float = 0.05
    min_history_runs: int = 4
    rng_seed: int = 20260917
    history_path: Path = Path(".driftgate/history.json")


class DriftGateConfig(BaseModel):
    """Root configuration document, mirroring ``driftgate.config.yaml``."""

    version: str = "1.0"
    service: ServiceConfig = Field(default_factory=ServiceConfig)
    traffic_mode: TrafficMode = "auto_record"
    cassette_dir: Path = Path("cassettes")
    contract: ContractConfig = Field(default_factory=ContractConfig)
    streaming_sla: StreamingSLAConfig = Field(default_factory=StreamingSLAConfig)
    metamorphic_assertions: list[MetamorphicAssertionConfig] = Field(default_factory=list)
    quality_gate: QualityGateConfig = Field(default_factory=QualityGateConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    governor: GovernorConfig = Field(default_factory=GovernorConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> DriftGateConfig:
        """Load config from YAML (if present) and apply the env-var overlay.

        Missing file → all defaults, not an error: the framework must be
        usable with zero configuration.
        """
        resolved = Path(path) if path is not None else Path(DEFAULT_CONFIG_FILENAME)
        data: dict[str, Any] = {}
        if resolved.is_file():
            loaded = yaml.safe_load(resolved.read_text()) or {}
            if not isinstance(loaded, dict):
                raise ConfigError(f"{resolved} must contain a YAML mapping at the top level")
            data = loaded
        data = _apply_env_overlay(data)
        return cls.model_validate(data)


def _coerce_env_value(raw: str) -> Any:
    """Best-effort scalar coercion for an environment-variable override."""
    low = raw.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _set_nested(data: dict[str, Any], path: list[str], value: Any) -> None:
    node = data
    for key in path[:-1]:
        existing = node.get(key)
        if existing is None:
            existing = {}
            node[key] = existing
        elif not isinstance(existing, dict):
            raise ConfigError(
                f"Env-var override path {'.'.join(path)!r} conflicts with a non-mapping "
                f"value already present at {key!r}"
            )
        node = existing
    node[path[-1]] = value


def _apply_env_overlay(data: dict[str, Any]) -> dict[str, Any]:
    """Overlay ``DRIFT_GATE_<SECTION>__<KEY>`` env vars onto a loaded config dict.

    This is the only place raw environment strings are read for configuration
    purposes; everything downstream consumes the validated model.
    """
    for env_key, raw_value in os.environ.items():
        if not env_key.startswith(ENV_PREFIX):
            continue
        suffix = env_key[len(ENV_PREFIX) :]
        if not suffix:
            continue
        path = [segment.lower() for segment in suffix.split("__") if segment]
        if not path:
            continue
        _set_nested(data, path, _coerce_env_value(raw_value))
    return data
