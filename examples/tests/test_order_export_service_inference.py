"""Schema-inference bootstrap proven end-to-end against the order/export API.

The express mock's own ``openapi.yaml`` is hand-written; this suite proves the
*inference* path instead: record real traffic into cassettes, run
``infer_openapi_spec`` over them, and assert:

1. a valid OpenAPI-shaped document comes back;
2. **self-consistency** — every interaction the contract was inferred from
   validates successfully against the contract that inference produced; and
3. an out-of-sample field on a brand-new interaction yields only advisory
   warnings, never errors (the ``x-driftgate-inferred`` contract).
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from driftgate.contracts.inference import (
    infer_openapi_spec,
    load_cassette_interactions,
)
from driftgate.contracts.validator import ContractValidator
from driftgate.recorder.cassette import CassetteInteraction, CassetteMode, CassetteStore
from driftgate.recorder.fingerprint import FingerprintConfig
from driftgate.recorder.interceptor import RecordingTransport

RECORD_DIR = Path(".driftgate/inference_cassettes")
BEARER = {"Authorization": "Bearer mock-jwt-bearer-token-12345"}
# Enough distinct samples per (method, path) for inference to build each
# operation, and the real Bearer header so the live mock answers orders
# with its 202 path instead of 401.
STATELESS_CALLS: list[tuple[str, str, dict[str, str], object]] = [
    ("POST", "/auth/login", {}, {"username": "drift", "apiKey": "gate"}),
    ("POST", "/auth/login", {}, {"username": "drift", "apiKey": "gate"}),
    ("POST", "/auth/login", {}, {"username": "drift", "apiKey": "gate"}),
    (
        "POST",
        "/orders",
        {**BEARER, "X-Correlation-ID": "infer-1"},
        {"customerId": "c-1", "items": [{"quantity": 1, "unitPrice": 5}], "shippingAddress": "a"},
    ),
    (
        "POST",
        "/orders",
        {**BEARER, "X-Correlation-ID": "infer-2"},
        {"customerId": "c-2", "items": [{"quantity": 2, "unitPrice": 7.5}], "shippingAddress": "b"},
    ),
    (
        "POST",
        "/orders",
        {**BEARER, "X-Correlation-ID": "infer-3"},
        {"customerId": "c-3", "items": [{"quantity": 3, "unitPrice": 9}], "shippingAddress": "c"},
    ),
]


@pytest.fixture(scope="module")
def recorded_interactions(express_base_url: str) -> Iterator[list[CassetteInteraction]]:
    """Record real traffic into a fresh cassette directory for inference."""
    shutil.rmtree(RECORD_DIR, ignore_errors=True)
    store = CassetteStore(
        RECORD_DIR / "bootstrap.yaml",
        CassetteMode.RECORD,
        service=express_base_url,
    )
    transport = RecordingTransport(
        httpx.HTTPTransport(), store, fingerprint_config=FingerprintConfig()
    )
    with httpx.Client(base_url=express_base_url, transport=transport, timeout=10.0) as client:
        for method, path, headers, body in STATELESS_CALLS:
            response = client.request(method, path, json=body, headers=headers)
            response.read()
    store.save()
    try:
        yield load_cassette_interactions(str(RECORD_DIR))
    finally:
        shutil.rmtree(RECORD_DIR, ignore_errors=True)


def test_inferred_spec_is_valid_openapi(recorded_interactions) -> None:
    result = infer_openapi_spec(recorded_interactions, min_samples=3)
    assert result.spec["openapi"].startswith("3.")
    assert result.spec["x-driftgate-inferred"] is True
    assert result.spec["paths"], "Inference produced no operations"


def test_inference_is_self_consistent(recorded_interactions) -> None:
    """Every source interaction validates against the contract inference built."""
    result = infer_openapi_spec(recorded_interactions, min_samples=3)
    validator = ContractValidator(result.spec)
    for interaction in recorded_interactions:
        contract_result = validator.check_interaction(interaction)
        assert contract_result.ok, (
            f"{contract_result.method} {contract_result.path} → {contract_result.status}: {contract_result.violations}"
        )


def test_inferred_contract_is_advisory(recorded_interactions) -> None:
    """A contract-breaking interaction must warn, never fail, against inference."""
    result = infer_openapi_spec(recorded_interactions, min_samples=3)
    validator = ContractValidator(result.spec)

    # Break the recorded response's observed shape outright: the inferred
    # spec declared every field seen at this status as required, so dropping
    # one violates it. Because the spec is marked advisory, the violation
    # downgrades from "error" to "warning" and the interaction stays ok.
    novel = recorded_interactions[1].model_copy(deep=True)
    assert isinstance(novel.response.body, dict)
    novel.response.body = {
        key: value for key, value in novel.response.body.items() if key == "orderId"
    }

    contract_result = validator.check_interaction(novel)
    assert contract_result.ok, "Advisory inferred contract must not fail on new fields"
    assert any(v.severity == "warning" for v in contract_result.violations)
