from __future__ import annotations

from driftgate.contracts.validator import ContractValidator


def _spec(*, inferred: bool = False) -> dict:
    return {
        "openapi": "3.1.0",
        "x-driftgate-inferred": inferred,
        "info": {"title": "test", "version": "1"},
        "paths": {
            "/v1/orders/{orderId}": {
                "get": {
                    "responses": {
                        "200": {
                            "description": "ok",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "required": ["orderId", "status"],
                                        "properties": {
                                            "orderId": {"type": "string"},
                                            "status": {"type": "string"},
                                        },
                                        "additionalProperties": False,
                                    }
                                }
                            },
                        },
                        "404": {"description": "not found"},
                    }
                }
            },
            "/v1/orders": {
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
        },
    }


def test_validates_declared_status_and_response_shape() -> None:
    validator = ContractValidator(_spec())
    assert (
        validator.validate_response(
            "GET", "/v1/orders/ORD-1", 200, body={"orderId": "ORD-1", "status": "PENDING"}
        )
        == []
    )

    violations = validator.validate_response("GET", "/v1/orders/ORD-1", 200, body={"orderId": 1})
    assert {violation.kind for violation in violations} == {"response_body"}
    assert all(violation.severity == "error" for violation in violations)

    undeclared = validator.validate_response("GET", "/v1/orders/ORD-1", 500, body={})
    assert undeclared[0].kind == "undeclared_status"
    assert undeclared[0].severity == "error"


def test_request_validation_aggregates_errors() -> None:
    validator = ContractValidator(_spec())
    violations = validator.validate_request("POST", "/v1/orders", body={"customerId": 42})

    assert len(violations) == 1
    assert "42 is not of type 'string'" in violations[0].message


def test_inferred_contract_downgrades_regressions_to_warnings() -> None:
    validator = ContractValidator(_spec(inferred=True))
    violations = validator.validate_response(
        "GET", "/v1/orders/ORD-1", 500, body={"unexpected": True}
    )

    assert violations[0].severity == "warning"
