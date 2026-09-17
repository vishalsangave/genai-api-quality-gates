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
