"""Command-line interface for DriftGate."""

from __future__ import annotations

import json
import os
import sys
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
from driftgate.governor.gate import (
    GateDecision,
    GovernorThresholds,
    ReleaseGovernor,
    RunMetrics,
    run_metrics_from_dict,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="API quality gates for non-deterministic GenAI microservices.",
)

_EXIT_CODES = {GateDecision.PROMOTE: 0, GateDecision.ROLLBACK: 1, GateDecision.HOLD: 2}

HISTORY_KEEP = 100


def _append_history(path: Path, metrics: RunMetrics) -> None:
    """Append the current run to the history ring buffer (keep last HISTORY_KEEP)."""
    try:
        history: list[dict[str, Any]] = []
        if path.exists():
            try:
                loaded = json.loads(path.read_text())
                if isinstance(loaded, list):
                    history = [item for item in loaded if isinstance(item, dict)]
            except (OSError, json.JSONDecodeError) as error:
                print(f"driftgate: corrupt history {path} restarted: {error}", file=sys.stderr)
                history = []
        history.append(
            {
                "run_id": metrics.run_id,
                "safety_pass_rate": metrics.safety_pass_rate,
                "task_success_rate": metrics.task_success_rate,
                "p95_latency_ms": metrics.p95_latency_ms,
                "latency_samples": list(metrics.latency_samples),
                "task_success_samples": list(metrics.task_success_samples),
                "context_preservation": metrics.context_preservation,
                "evidence_coverage": metrics.evidence_coverage,
            }
        )
        history = history[-HISTORY_KEEP:]
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(history, indent=2) + "\n")
        os.replace(tmp, path)
    except OSError as error:
        print(f"driftgate: could not append history to {path}: {error}", file=sys.stderr)


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
    return run_metrics_from_dict(data)


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
        print(f"driftgate: corrupt history {path} treated as empty: {error}", file=sys.stderr)
        return []
    if not isinstance(loaded, list):
        print(f"driftgate: history {path} is not an array; treated as empty", file=sys.stderr)
        return []
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
    history: Annotated[Path | None, typer.Option("--history")] = None,
    config: Annotated[Path | None, typer.Option("--config", exists=True, readable=True)] = None,
) -> None:
    """Evaluate metrics JSON and exit: PROMOTE=0, ROLLBACK=1, HOLD=2."""
    loaded_config = DriftGateConfig.load(config)
    current = _load_metrics(metrics)
    report = ReleaseGovernor(
        _thresholds(loaded_config.governor),
        history=_load_history(history),
    ).evaluate(current)
    _print_report(report)
    if history is not None:
        _append_history(history, current)
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
    current = _load_metrics(metrics_path)
    report = ReleaseGovernor(
        _thresholds(loaded_config.governor),
        history=_load_history(loaded_config.governor.history_path),
    ).evaluate(current)
    _print_report(report)
    _append_history(loaded_config.governor.history_path, current)
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


@app.command("init")
def init(
    base_url: Annotated[str, typer.Option("--base-url")] = "http://localhost:8000",
    spec: Annotated[Path | None, typer.Option("--spec")] = None,
    start_service_command: Annotated[str | None, typer.Option("--start-service-command")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Scaffold driftgate.config.yaml, a starter suite, and a CI workflow."""
    from driftgate import scaffold

    spec_source: str | None = str(spec) if spec else None
    spec_document = None
    if spec_source and Path(spec_source).exists():
        try:
            loaded = yaml.safe_load(Path(spec_source).read_text())
            if isinstance(loaded, dict):
                spec_document = loaded
        except yaml.YAMLError:
            typer.echo(
                f"Warning: could not parse spec {spec_source}; skipping test generation", err=True
            )

    files: dict[Path, str] = {
        Path(scaffold.CONFIG_FILENAME): scaffold.config_template(base_url, spec_source),
        scaffold.STARTER_TEST_PATH: scaffold.starter_test_source(base_url, spec_document),
        scaffold.WORKFLOW_PATH: scaffold.workflow_source(start_service_command),
    }
    existing = [str(path) for path in files if path.exists()]
    if existing and not force:
        typer.echo("Refusing to overwrite existing files (use --force):", err=True)
        for name in existing:
            typer.echo(f"  {name}", err=True)
        raise typer.Exit(1)
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        typer.echo(f"  wrote {path}")
    if scaffold.ensure_gitignore_entries(Path(".")):
        typer.echo("  updated .gitignore")
    typer.echo("\nNext steps:")
    typer.echo(f"  1. Start your service at {base_url}")
    if not spec_source:
        typer.echo(f"  2. uv run driftgate capture --base-url {base_url} --out openapi.yaml")
        typer.echo("  3. Point contract.openapi_spec at openapi.yaml, then: pytest -v")
    else:
        typer.echo("  2. pytest -v")


@app.command("capture")
def capture(
    base_url: Annotated[str, typer.Option("--base-url")] = "http://localhost:8000",
    requests: Annotated[Path | None, typer.Option("--requests", exists=True, readable=True)] = None,
    auth_header: Annotated[str | None, typer.Option("--auth-header")] = None,
    include_defaults: Annotated[bool, typer.Option("--include-defaults")] = False,
    save_to: Annotated[Path, typer.Option("--save-to")] = Path("cassettes"),
    out: Annotated[Path | None, typer.Option("--out")] = None,
    min_samples: Annotated[int, typer.Option("--min-samples", min=1)] = 1,
) -> None:
    """Record live traffic into cassettes and optionally infer an advisory spec."""
    from driftgate import capture as capture_module

    auth = capture_module.parse_auth_header(auth_header)
    if requests:
        calls = capture_module.load_request_list(requests)
    else:
        calls = capture_module.default_probes()
    if include_defaults:
        calls = capture_module.default_probes() + [
            call for call in calls if call not in capture_module.default_probes()
        ]
    for call in calls:
        if auth and "Authorization" not in call["headers"]:
            call["headers"] = {**auth, **call["headers"]}  # type: ignore[dict-item]
    cassette_path = str(save_to / "capture.yaml")
    try:
        count = capture_module.capture_traffic(
            base_url=base_url, calls=calls, cassette_path=cassette_path
        )
    except RuntimeError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Recorded {count} interaction(s) into {cassette_path}")
    if out:
        result = capture_module.infer_advisory_spec(str(save_to), min_samples=min_samples)
        for warning in result.warnings:
            typer.echo(f"Warning: {warning.message}", err=True)
        rendered = yaml.safe_dump(result.spec, sort_keys=False, allow_unicode=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered)
        typer.echo(
            f"Wrote advisory contract to {out} (review it, then consider --promote via infer-schema)"
        )
    typer.echo("Next: point contract.openapi_spec at the spec, then run pytest -v")


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


@app.command("report")
def report(
    as_json: Annotated[bool, typer.Option("--json")] = False,
    path: Annotated[Path, typer.Option("--path")] = Path(".driftgate/report.json"),
) -> None:
    """Print the last build's report as markdown (or raw JSON with --json)."""
    if not path.exists():
        typer.echo(f"No DriftGate report found at {path}; run pytest first.", err=True)
        raise typer.Exit(1)
    try:
        loaded = json.loads(path.read_text())
        if not isinstance(loaded, dict):
            raise ValueError("report must be one JSON object")
    except (OSError, json.JSONDecodeError, ValueError) as error:
        typer.echo(f"Could not read report {path}: {error}", err=True)
        raise typer.Exit(1) from error
    if as_json:
        typer.echo(json.dumps(loaded, indent=2, sort_keys=True))
        return
    from driftgate.reporting import ReportData, render_markdown

    typer.echo(render_markdown(ReportData.from_dict(loaded)), nl=False)


@app.command("version")
def version() -> None:
    """Print the installed DriftGate version."""
    typer.echo(__version__)


def main() -> None:
    app()
