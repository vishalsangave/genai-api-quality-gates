"""Command-line interface for DriftGate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import pytest
import typer
import yaml

from driftgate import __version__
from driftgate.config import DriftGateConfig, GovernorConfig
from driftgate.contracts.inference import (
    infer_openapi_spec,
    load_cassette_interactions,
    promote_inferred_spec,
)
from driftgate.governor.gate import GateDecision, GovernorThresholds, ReleaseGovernor, RunMetrics

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="API quality gates for non-deterministic GenAI microservices.",
)

_EXIT_CODES = {GateDecision.PROMOTE: 0, GateDecision.ROLLBACK: 1, GateDecision.HOLD: 2}


def _thresholds(config: GovernorConfig) -> GovernorThresholds:
    return GovernorThresholds(
        safety_pass_rate_min=config.safety_pass_rate_min,
        task_success_min=config.task_success_min,
        p95_latency_ms_max=config.p95_latency_ms_max,
        context_preservation_min=config.context_preservation_min,
        context_preservation_hard=config.context_preservation_hard,
        evidence_coverage_min=config.evidence_coverage_min,
        evidence_coverage_hard=config.evidence_coverage_hard,
        ci_level=config.ci_level,
        ci_overlap_tolerance=config.ci_overlap_tolerance,
        trend_alpha=config.trend_alpha,
        min_history_runs=config.min_history_runs,
        rng_seed=config.rng_seed,
    )


def _metrics_from_dict(data: dict[str, Any]) -> RunMetrics:
    fields = {
        "run_id": data.get("run_id", "unnamed"),
        "safety_pass_rate": data.get("safety_pass_rate"),
        "task_success_rate": data.get("task_success_rate"),
        "p95_latency_ms": data.get("p95_latency_ms"),
        "latency_samples": data.get("latency_samples", ()),
        "task_success_samples": data.get("task_success_samples", ()),
        "context_preservation": data.get("context_preservation"),
        "evidence_coverage": data.get("evidence_coverage"),
    }
    return RunMetrics(**fields)


def _load_metrics(path: Path) -> RunMetrics:
    try:
        loaded = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise typer.BadParameter(f"Could not read metrics JSON {path}: {error}") from error
    if not isinstance(loaded, dict):
        raise typer.BadParameter(f"Metrics JSON {path} must be one object")
    return _metrics_from_dict(loaded)


def _load_history(path: Path | None) -> list[RunMetrics]:
    if path is None or not path.exists():
        return []
    try:
        loaded = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise typer.BadParameter(f"Could not read history JSON {path}: {error}") from error
    if not isinstance(loaded, list):
        raise typer.BadParameter(f"History JSON {path} must be an array")
    return [_metrics_from_dict(item) for item in loaded if isinstance(item, dict)]


def _print_report(report: Any) -> None:
    typer.echo(f"Decision: {report.decision}")
    for reason in report.reasons:
        typer.echo(f"- {reason}")
    if report.hard_failures:
        typer.echo("Hard failures:")
        for failure in report.hard_failures:
            typer.echo(f"  - {failure}")
    if report.soft_signals:
        typer.echo("Soft signals:")
        for signal in report.soft_signals:
            typer.echo(f"  - {signal}")


@app.command("gate")
def gate(
    metrics: Annotated[Path, typer.Option("--metrics", exists=True, readable=True)],
    history: Annotated[Path | None, typer.Option("--history", exists=True, readable=True)] = None,
    config: Annotated[Path | None, typer.Option("--config", exists=True, readable=True)] = None,
) -> None:
    """Evaluate metrics JSON and exit: PROMOTE=0, ROLLBACK=1, HOLD=2."""
    loaded_config = DriftGateConfig.load(config)
    report = ReleaseGovernor(
        _thresholds(loaded_config.governor),
        history=_load_history(history),
    ).evaluate(_load_metrics(metrics))
    _print_report(report)
    raise typer.Exit(_EXIT_CODES[report.decision])


@app.command("run")
def run(
    config: Annotated[Path | None, typer.Option("--config", exists=True, readable=True)] = None,
    marker: Annotated[str | None, typer.Option("--marker", "-m")] = None,
    record: Annotated[bool, typer.Option("--record")] = False,
    fail_on_hold: Annotated[bool, typer.Option("--fail-on-hold")] = False,
) -> None:
    """Run pytest with the plugin, then govern `.driftgate/metrics.json` if it exists."""
    args: list[str] = []
    if config:
        args.extend(["--driftgate-config", str(config)])
    if marker:
        args.extend(["-m", marker])
    if record:
        args.append("--record")
    pytest_exit = pytest.main(args)
    if pytest_exit != pytest.ExitCode.OK:
        raise typer.Exit(int(pytest_exit))

    loaded_config = DriftGateConfig.load(config)
    metrics_path = Path(".driftgate/metrics.json")
    if not metrics_path.exists():
        typer.echo("Pytest passed; no .driftgate/metrics.json was produced, skipping governor.")
        raise typer.Exit(0)
    report = ReleaseGovernor(
        _thresholds(loaded_config.governor),
        history=_load_history(loaded_config.governor.history_path),
    ).evaluate(_load_metrics(metrics_path))
    _print_report(report)
    if report.decision is GateDecision.HOLD and not fail_on_hold:
        typer.echo("HOLD returned as success; use --fail-on-hold for CI enforcement.")
        raise typer.Exit(0)
    raise typer.Exit(_EXIT_CODES[report.decision])


@app.command("record")
def record(
    config: Annotated[Path | None, typer.Option("--config", exists=True, readable=True)] = None,
) -> None:
    """Run pytest while forcing record mode."""
    args = ["--record"]
    if config:
        args.extend(["--driftgate-config", str(config)])
    raise typer.Exit(int(pytest.main(args)))


@app.command("infer-schema")
def infer_schema(
    cassette_dir: Annotated[Path, typer.Option("--from-cassettes", exists=True, file_okay=False)],
    output: Annotated[Path | None, typer.Option("--out")] = None,
    min_samples: Annotated[int, typer.Option("--min-samples", min=1)] = 3,
    promote: Annotated[bool, typer.Option("--promote")] = False,
) -> None:
    """Generate an advisory OpenAPI 3.1 starter contract from cassettes."""
    interactions = load_cassette_interactions(str(cassette_dir))
    result = infer_openapi_spec(interactions, min_samples=min_samples)
    spec = promote_inferred_spec(result.spec) if promote else result.spec
    rendered = yaml.safe_dump(spec, sort_keys=False, allow_unicode=True)
    for warning in result.warnings:
        typer.echo(f"Warning: {warning.message}", err=True)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
        typer.echo(f"Wrote {'strict' if promote else 'advisory'} contract to {output}")
    else:
        typer.echo(rendered, nl=False)


@app.command("version")
def version() -> None:
    """Print the installed DriftGate version."""
    typer.echo(__version__)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
