from __future__ import annotations

import json

from typer.testing import CliRunner

from driftgate.cli import app

runner = CliRunner()


def _write_metrics(path, *, safety=1.0, success=0.9, p95=900.0) -> None:
    path.write_text(
        json.dumps(
            {
                "run_id": "cli-test",
                "safety_pass_rate": safety,
                "task_success_rate": success,
                "p95_latency_ms": p95,
                "latency_samples": [p95 - 10, p95, p95 + 10],
                "task_success_samples": [1, 1, 1, 1, 0],
            }
        )
    )


def test_gate_exit_codes(tmp_path) -> None:
    config = tmp_path / "driftgate.config.yaml"
    config.write_text("governor: {}\n")

    promote = tmp_path / "promote.json"
    _write_metrics(promote)
    promoted = runner.invoke(app, ["gate", "--config", str(config), "--metrics", str(promote)])
    assert promoted.exit_code == 0
    assert "PROMOTE" in promoted.stdout

    rollback = tmp_path / "rollback.json"
    _write_metrics(rollback, safety=0.9)
    rolled_back = runner.invoke(app, ["gate", "--config", str(config), "--metrics", str(rollback)])
    assert rolled_back.exit_code == 1
    assert "ROLLBACK" in rolled_back.stdout

    hold = tmp_path / "hold.json"
    _write_metrics(hold)
    baseline = tmp_path / "history.json"
    baseline.write_text(
        json.dumps(
            [
                {
                    "run_id": "baseline",
                    "safety_pass_rate": 1.0,
                    "task_success_rate": 0.9,
                    "p95_latency_ms": 900.0,
                }
            ]
        )
    )
    # The `gate` command treats history as trend input, so force an increasing
    # latency trend through enough historical observations to yield HOLD.
    baseline.write_text(
        json.dumps(
            [
                {
                    "run_id": str(index),
                    "safety_pass_rate": 1.0,
                    "task_success_rate": 0.9,
                    "p95_latency_ms": value,
                }
                for index, value in enumerate([100, 200, 300, 400, 500])
            ]
        )
    )
    held = runner.invoke(
        app,
        ["gate", "--config", str(config), "--metrics", str(hold), "--history", str(baseline)],
    )
    assert held.exit_code == 2
    assert "HOLD" in held.stdout


def test_infer_schema_cli_writes_advisory_contract(tmp_path) -> None:
    cassette_dir = tmp_path / "cassettes"
    cassette_dir.mkdir()
    # Two cassettes with structurally similar observations are enough when
    # the command's minimum is explicitly lowered for this focused CLI test.
    for index in (1, 2):
        (cassette_dir / f"order-{index}.yaml").write_text(
            f"""version: 1
recorded_at: '2026-09-17T00:00:00Z'
interactions:
  - fingerprint: fp-{index}
    canonical: '{{}}'
    request:
      method: GET
      path: /v1/orders/ORD-1000{index}
      query: {{}}
      headers: {{}}
      body: null
    response:
      status: 200
      headers:
        content-type: application/json
      body:
        orderId: ORD-1000{index}
        status: PENDING
      is_stream: false
      sse_events: []
"""
        )
    output = tmp_path / "inferred.yaml"
    result = runner.invoke(
        app,
        [
            "infer-schema",
            "--from-cassettes",
            str(cassette_dir),
            "--out",
            str(output),
            "--min-samples",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert output.exists()
    assert "x-driftgate-inferred: true" in output.read_text()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.0"


def test_gate_creates_history_on_cold_start(tmp_path) -> None:
    """--history must not require the file to pre-exist (bootstrap deadlock fix)."""
    metrics = tmp_path / "metrics.json"
    _write_metrics(metrics)
    history = tmp_path / "history.json"
    assert not history.exists()

    result = runner.invoke(app, ["gate", "--metrics", str(metrics), "--history", str(history)])

    assert result.exit_code == 0
    assert history.exists()
    entries = json.loads(history.read_text())
    assert len(entries) == 1
    assert entries[0]["run_id"] == "cli-test"


def test_gate_history_ring_buffer_trims_to_keep(tmp_path) -> None:
    from driftgate.cli import HISTORY_KEEP

    metrics = tmp_path / "metrics.json"
    _write_metrics(metrics)
    history = tmp_path / "history.json"
    preseed = [{"run_id": f"old-{i}", "p95_latency_ms": 100.0} for i in range(HISTORY_KEEP)]
    history.write_text(json.dumps(preseed))

    result = runner.invoke(app, ["gate", "--metrics", str(metrics), "--history", str(history)])

    assert result.exit_code == 0
    entries = json.loads(history.read_text())
    assert len(entries) == HISTORY_KEEP
    assert entries[-1]["run_id"] == "cli-test"
    assert entries[0]["run_id"] == "old-1"  # oldest trimmed


def test_gate_corrupt_history_restarts(tmp_path) -> None:
    metrics = tmp_path / "metrics.json"
    _write_metrics(metrics)
    history = tmp_path / "history.json"
    history.write_text("{not a list")

    result = runner.invoke(app, ["gate", "--metrics", str(metrics), "--history", str(history)])

    assert result.exit_code == 0
    entries = json.loads(history.read_text())
    assert len(entries) == 1
    assert entries[0]["run_id"] == "cli-test"


def test_gate_without_history_flag_writes_nothing(tmp_path) -> None:
    metrics = tmp_path / "metrics.json"
    _write_metrics(metrics)
    result = runner.invoke(app, ["gate", "--metrics", str(metrics)])
    assert result.exit_code == 0
    assert not (tmp_path / "history.json").exists()


def test_report_command_renders_markdown(tmp_path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "run_id": "r1",
                "timestamp": "t",
                "exitstatus": 0,
                "totals": {"passed": 3, "failed": 0, "errors": 0, "skipped": 0, "total": 3},
                "endpoints": [
                    {
                        "method": "GET",
                        "template": "/orders/{orderId}",
                        "tests": 2,
                        "failed": 0,
                        "statuses_seen": [200, 404],
                    }
                ],
                "coverage": None,
                "governor": {
                    "decision": "PROMOTE",
                    "reasons": [],
                    "hard_failures": [],
                    "soft_signals": [],
                },
                "spec_diff": None,
            }
        )
    )
    result = runner.invoke(app, ["report", "--path", str(report_path)])
    assert result.exit_code == 0
    assert "# DriftGate Build Report" in result.stdout
    assert "PROMOTE" in result.stdout
    assert "/orders/{orderId}" in result.stdout


def test_report_command_json_mode(tmp_path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"run_id": "r", "totals": {}, "endpoints": []}))
    result = runner.invoke(app, ["report", "--json", "--path", str(report_path)])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["run_id"] == "r"


def test_report_command_missing_file(tmp_path) -> None:
    result = runner.invoke(app, ["report", "--path", str(tmp_path / "absent.json")])
    assert result.exit_code == 1
    assert "No DriftGate report" in result.stderr
