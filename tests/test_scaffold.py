"""Tests for driftgate.scaffold (init) and driftgate.capture."""

from __future__ import annotations

from pathlib import Path

import httpx
import yaml
from typer.testing import CliRunner

from driftgate.cli import app
from driftgate.config import DriftGateConfig
from driftgate.scaffold import (
    CONFIG_FILENAME,
    STARTER_TEST_PATH,
    WORKFLOW_PATH,
    config_template,
    load_spec_if_present,
    starter_test_source,
    workflow_source,
)

runner = CliRunner()


def _spec() -> dict:
    return {
        "openapi": "3.1.0",
        "paths": {
            "/orders": {
                "post": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["customerId"],
                                    "properties": {"customerId": {"type": "string"}},
                                }
                            }
                        }
                    },
                    "responses": {"202": {"description": "accepted"}},
                }
            },
            "/orders/{orderId}": {
                "get": {
                    "parameters": [
                        {
                            "name": "orderId",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {"200": {"description": "ok"}, "404": {"description": "missing"}},
                }
            },
            "/healthz": {"get": {"responses": {"200": {"description": "ok"}}}},
        },
    }


def test_config_template_loads_as_driftgate_config() -> None:
    rendered = config_template("http://localhost:9000/api", None)
    config = DriftGateConfig.model_validate(yaml.safe_load(rendered))
    assert config.service.base_url == "http://localhost:9000/api"
    assert config.contract.openapi_spec is None


def test_config_template_with_spec() -> None:
    rendered = config_template("http://localhost:9000", "openapi.yaml")
    config = DriftGateConfig.model_validate(yaml.safe_load(rendered))
    assert config.contract.openapi_spec == Path("openapi.yaml")


def test_starter_test_compiles_without_spec() -> None:
    source = starter_test_source("http://localhost:8000", None)
    compile(source, "test_api.py", "exec")


def test_starter_test_compiles_with_spec_and_covers_operations() -> None:
    source = starter_test_source("http://localhost:8000", _spec())
    compile(source, "test_api.py", "exec")
    assert "def test_service_reachable" in source
    assert "def test_post_orders_responds" in source
    assert "def test_get_orders_orderid_responds" in source
    # The path-parameterized probe uses a non-existent sentinel.
    assert "/orders/00000" in source
    # The POST test sends a schema-derived minimal body.
    assert "customerId:" in source
    # The readiness probe prefers a spec-declared param-free 200 GET.
    assert '"/healthz"' in source


def test_probe_path_prefers_spec_get() -> None:
    from driftgate.scaffold import probe_path

    assert probe_path(_spec()) == "/healthz"
    assert probe_path(None) == "/healthz"


def test_workflow_includes_report_step_without_service() -> None:
    source = workflow_source(None)
    assert "Run DriftGate gate" in source
    assert "GITHUB_STEP_SUMMARY" in source
    assert "retention-days: 90" in source
    assert "Start service under test" not in source


def test_workflow_with_start_service_command() -> None:
    source = workflow_source("npm install && node server.js")
    assert "Start service under test" in source
    assert "node server.js" in source


def test_init_command_writes_all_files(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "--base-url", "http://localhost:9000/v1"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert (tmp_path / CONFIG_FILENAME).exists()
    assert (tmp_path / STARTER_TEST_PATH).exists()
    assert (tmp_path / WORKFLOW_PATH).exists()
    gitignore = (tmp_path / ".gitignore").read_text()
    assert ".driftgate/" in gitignore
    assert "cassettes/" in gitignore
    generated = yaml.safe_load((tmp_path / CONFIG_FILENAME).read_text())
    assert generated["service"]["base_url"] == "http://localhost:9000/v1"
    assert "Next steps" in result.stdout


def test_init_refuses_clobber_without_force(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / CONFIG_FILENAME).write_text("existing: true\n")
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 1
    assert "Refusing to overwrite" in result.stderr
    assert (tmp_path / CONFIG_FILENAME).read_text() == "existing: true\n"

    forced = runner.invoke(app, ["init", "--force"])
    assert forced.exit_code == 0
    assert "openapi_spec" in (tmp_path / CONFIG_FILENAME).read_text()


def test_init_with_spec_generates_operation_tests(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    spec_path = tmp_path / "openapi.yaml"
    spec_path.write_text(yaml.safe_dump(_spec()))
    result = runner.invoke(app, ["init", "--spec", str(spec_path)])
    assert result.exit_code == 0
    starter = (tmp_path / STARTER_TEST_PATH).read_text()
    assert "def test_get_orders_orderid_responds" in starter


def test_load_spec_if_present(tmp_path) -> None:
    assert load_spec_if_present(None) is None
    assert load_spec_if_present(str(tmp_path / "absent.yaml")) is None
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(_spec()))
    loaded = load_spec_if_present(str(spec_path))
    assert loaded is not None and "paths" in loaded


def test_capture_replay_records_cassettes(tmp_path, monkeypatch) -> None:
    from driftgate.recorder.cassette import CassetteMode, CassetteStore
    from driftgate.recorder.fingerprint import FingerprintConfig
    from driftgate.recorder.interceptor import RecordingTransport

    calls = [
        {"method": "GET", "path": "/healthz", "headers": {}, "body": None},
        {
            "method": "POST",
            "path": "/v1/orders",
            "headers": {"Authorization": "Bearer x"},
            "body": {"customerId": "c1"},
        },
    ]
    cassette_path = tmp_path / "capture.yaml"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(202, json={"orderId": "ORD-00001", "status": "PENDING"})

    store = CassetteStore(str(cassette_path), CassetteMode.RECORD, service="http://mock")
    transport = RecordingTransport(
        httpx.MockTransport(handler), store, fingerprint_config=FingerprintConfig()
    )
    with httpx.Client(base_url="http://mock", transport=transport, timeout=5.0) as client:
        for call in calls:
            response = client.request(
                str(call["method"]), str(call["path"]), json=call["body"], headers=call["headers"]
            )
            response.read()
    store.save()
    assert cassette_path.exists()
    assert len(store.cassette.interactions) == 2


def test_capture_command_end_to_end(tmp_path, monkeypatch) -> None:
    """capture records cassettes + writes an advisory spec, driven through the CLI."""
    monkeypatch.chdir(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json={"service": "mock", "version": "1"})

    real_client = httpx.Client

    def patched_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        kwargs.pop("follow_redirects", None)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", patched_client)

    requests_file = tmp_path / "requests.yaml"
    requests_file.write_text(
        yaml.safe_dump(
            [
                {"method": "GET", "path": "/healthz"},
                {"method": "GET", "path": "/info"},
            ]
        )
    )
    result = runner.invoke(
        app,
        [
            "capture",
            "--base-url",
            "http://mock",
            "--requests",
            str(requests_file),
            "--save-to",
            str(tmp_path / "cassettes"),
            "--out",
            str(tmp_path / "openapi.yaml"),
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    cassettes = list((tmp_path / "cassettes").glob("*.yaml"))
    assert cassettes, "no cassette written"
    spec = yaml.safe_load((tmp_path / "openapi.yaml").read_text())
    assert spec["x-driftgate-inferred"] is True
    assert "paths" in spec


def test_capture_unreachable_service_fails_clean(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DRIFTGATE_TEST_UNREACHABLE", "1")

    def unreachable(*args, **kwargs):
        raise httpx.ConnectError("refused")

    # Reachability check happens before any recording; simulate with a bad host.
    result = runner.invoke(
        app, ["capture", "--base-url", "http://127.0.0.1:1", "--save-to", str(tmp_path / "c")]
    )
    assert result.exit_code == 1
    assert "No service answered" in result.stderr
