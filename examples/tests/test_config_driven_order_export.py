"""Config-driven order & export suite: proves the service template end-to-end.

``examples/driftgate.service.template.yaml`` is the manifest a adopting team
would copy. This file is the executable proof that the template is not just
documentation:

1. **Template validity** — :meth:`DriftGateConfig.load` accepts every key the
   template documents. If a config key is renamed in ``driftgate/config.py``
   and the template drifts, these tests fail immediately.

2. **Spec resolution** — the ``contract.openapi_spec`` the template points at
   exists, is OpenAPI 3.x, and covers the exact reference API endpoints.

3. **Config-driven behavior** — the suite then runs *through the plugin*
   against the running order & export API using the template's ``base_url``
   and ``openapi_spec``: responses are validated by a contract gate built
   purely from config, not from any test-local construction.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from driftgate.config import DriftGateConfig
from driftgate.contracts.validator import ContractValidator

TEMPLATE = Path("examples/driftgate.service.template.yaml")
EXPECTED_SPEC = Path("examples/express_mock/openapi.yaml")


def test_template_is_a_valid_driftgate_config() -> None:
    """Every key the template documents must load through the real config path."""
    config = DriftGateConfig.load(TEMPLATE)
    assert config.version == "1.0"
    assert config.service.base_url == "http://localhost:3000/v1"
    assert config.service.endpoint == "/orders"
    assert config.traffic_mode == "auto_record"
    assert config.cassette_dir == Path("cassettes")
    assert config.contract.openapi_spec == Path("examples/express_mock/openapi.yaml")
    assert config.contract.enforce is True
    assert config.streaming_sla.max_ttft_ms == 800.0
    assert config.judge.provider == "deterministic"
    assert config.judge.api_key_env == "DRIFT_GATE_JUDGE_API_KEY"
    assert [a.relation for a in config.metamorphic_assertions] == [
        "paraphrase_invariance",
        "adversarial_noise_invariance",
    ]
    assert config.governor.safety_pass_rate_min == 1.0
    assert config.governor.task_success_min == 0.80
    assert config.governor.p95_latency_ms_max == 15000.0
    assert config.governor.rng_seed == 20260917


def test_template_contract_spec_exists_and_covers_reference_api() -> None:
    config = DriftGateConfig.load(TEMPLATE)
    assert config.contract.openapi_spec is not None
    validator = ContractValidator.from_file(config.contract.openapi_spec)
    # The template's contract must describe the reference order & export API:
    # every endpoint from the reference spec appears as a declared operation.
    assert set(validator.spec["paths"]) >= {
        "/v1/auth/login",
        "/v1/orders",
        "/v1/orders/{orderId}",
        "/v1/exports",
        "/v1/exports/{jobId}",
        "/v1/exports/{jobId}/download",
    }
    declared_methods = {
        (method.upper(), template)
        for template, path_item in validator.spec["paths"].items()
        for method in path_item
        if method in {"get", "post", "delete"}
    }
    assert ("POST", "/v1/orders") in declared_methods
    assert ("DELETE", "/v1/orders/{orderId}") in declared_methods
    assert ("GET", "/v1/exports/{jobId}/download") in declared_methods


def test_template_contract_validates_reference_responses(
    express_client: pytest.FixtureRequest,
) -> None:
    """The template's spec must accept the real API's responses.

    Runs against the live order & export API (auto-started fixture or an
    external server via ``DRIFT_GATE_EXPRESS_URL``). Response bodies from
    the two endpoints whose success shapes the template's spec declares are
    validated through the exact config-driven path an adopting team uses.
    """
    config = DriftGateConfig.load(TEMPLATE)
    validator = ContractValidator.from_file(config.contract.openapi_spec)

    login_response = express_client.post("/auth/login", json={"username": "cfg", "apiKey": "gate"})
    assert login_response.status_code == 200
    login_violations = validator.validate_response(
        "POST", "/v1/auth/login", 200, body=login_response.json()
    )
    assert login_violations == []

    headers = {
        "Authorization": f"Bearer {login_response.json()['token']}",
        "X-Correlation-ID": "template-driven",
    }
    order_response = express_client.post(
        "/orders",
        json={
            "customerId": "cfg-cust",
            "items": [{"quantity": 3, "unitPrice": 11.5}],
            "shippingAddress": "2 Template St",
        },
        headers=headers,
    )
    assert order_response.status_code == 202
    body = order_response.json()
    assert body["totalAmount"] == 3 * 11.5
    order_violations = validator.validate_response("POST", "/v1/orders", 202, body=body)
    assert order_violations == [], (
        f"Contract gate rejected a valid order response: {order_violations}"
    )
