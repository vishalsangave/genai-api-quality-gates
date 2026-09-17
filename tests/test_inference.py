from __future__ import annotations

from driftgate.contracts.inference import (
    cluster_path_templates,
    infer_json_schema,
    infer_openapi_spec,
    promote_inferred_spec,
)
from driftgate.recorder.cassette import CassetteInteraction, RecordedRequest, RecordedResponse


def _interaction(path: str, amount: int, *, extra: bool = False) -> CassetteInteraction:
    return CassetteInteraction(
        fingerprint=f"fp-{path}-{amount}",
        canonical=f"canonical-{path}-{amount}",
        request=RecordedRequest(method="GET", path=path),
        response=RecordedResponse(
            status=200,
            body={
                "orderId": path.rsplit("/", 1)[-1],
                "amount": amount,
                **({"extra": True} if extra else {}),
            },
        ),
    )


def test_path_clustering_recognizes_identifier_segments() -> None:
    clusters = cluster_path_templates(
        [("GET", "/v1/orders/ORD-10001"), ("GET", "/v1/orders/ORD-10002")]
    )
    assert clusters[("GET", "/v1/orders/{param_3}")] == [
        "/v1/orders/ORD-10001",
        "/v1/orders/ORD-10002",
    ]


def test_schema_inference_widens_types_and_marks_required_by_presence() -> None:
    schema = infer_json_schema(
        [{"always": "x", "sometimes": 1}, {"always": "y", "sometimes": "two"}, {"always": "z"}]
    )

    assert schema["required"] == ["always"]
    assert schema["properties"]["sometimes"]["type"] == ["integer", "string"]
    assert "enum" not in schema["properties"]["always"]


def test_openapi_inference_is_advisory_and_promotable() -> None:
    interactions = [
        _interaction("/v1/orders/ORD-10001", 10),
        _interaction("/v1/orders/ORD-10002", 20, extra=True),
        _interaction("/v1/orders/ORD-10003", 30),
    ]
    result = infer_openapi_spec(interactions)

    assert result.spec["x-driftgate-inferred"] is True
    operation = result.spec["paths"]["/v1/orders/{param_3}"]["get"]
    assert operation["x-driftgate-inferred"] is True
    assert operation["responses"]["200"]["content"]["application/json"]["schema"]["required"] == [
        "amount",
        "orderId",
    ]

    promoted = promote_inferred_spec(result.spec)
    assert "x-driftgate-inferred" not in promoted
    assert "x-driftgate-inferred" not in promoted["paths"]["/v1/orders/{param_3}"]["get"]


def test_insufficient_samples_are_skipped_not_guessed() -> None:
    result = infer_openapi_spec([_interaction("/v1/orders/ORD-10001", 10)], min_samples=3)
    assert result.spec["paths"] == {}
    assert result.warnings
