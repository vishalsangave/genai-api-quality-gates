from __future__ import annotations

import json
from pathlib import Path

import pytest

from driftgate.specdiff import (
    DiffSeverity,
    SpecSnapshot,
    diff,
    load_snapshot,
    save_snapshot,
    snapshot,
    snapshot_from_dict,
    snapshot_to_dict,
)


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
                                    "properties": {
                                        "customerId": {"type": "string"},
                                        "quantity": {"type": "integer"},
                                    },
                                }
                            }
                        }
                    },
                    "responses": {
                        "202": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "required": ["orderId"],
                                        "properties": {
                                            "orderId": {"type": "string"},
                                            "status": {"type": "string"},
                                        },
                                    }
                                }
                            }
                        },
                        "400": {"description": "bad request"},
                    },
                }
            },
            "/orders/{orderId}": {
                "get": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {"orderId": {"type": "string"}},
                                    }
                                }
                            }
                        },
                        "404": {"description": "missing"},
                    }
                }
            },
        },
    }


def _status_findings(diff_result, kind: str) -> list:
    return [
        finding
        for endpoint in diff_result.changed
        for finding in endpoint.findings
        if finding.kind == kind
    ]


def test_snapshot_extracts_statuses_digests_and_fields() -> None:
    snap = snapshot(_spec())
    create = next(op for op in snap.operations if op.method == "POST" and op.template == "/orders")
    assert create.request_required == ("customerId",)
    assert create.request_properties == ("customerId", "quantity")
    assert create.request_schema_digest
    status_202 = create.statuses["202"]
    assert status_202.properties == ("orderId", "status")
    assert status_202.required == ("orderId",)
    assert status_202.schema_digest
    # A response with no JSON content still snapshots, with an empty digest.
    assert create.statuses["400"].schema_digest == ""


def test_snapshot_is_deterministic() -> None:
    assert snapshot(_spec()).digest == snapshot(_spec()).digest


def test_snapshot_operations_sorted_by_method_and_template() -> None:
    keys = [(op.method, op.template) for op in snapshot(_spec()).operations]
    assert keys == sorted(keys)


def test_snapshot_roundtrips_through_dict() -> None:
    snap = snapshot(_spec())
    assert snapshot_from_dict(snapshot_to_dict(snap)) == snap


def test_diff_first_snapshot_has_no_findings() -> None:
    result = diff(None, snapshot(_spec()))
    assert result.first_snapshot is True
    assert result.has_breaking is False
    assert not (result.added or result.removed or result.changed)


def test_diff_identical_specs_is_empty() -> None:
    result = diff(snapshot(_spec()), snapshot(_spec()))
    assert not (result.added or result.removed or result.changed)
    assert result.has_breaking is False


def test_diff_added_endpoint_is_additive() -> None:
    new_spec = _spec()
    new_spec["paths"]["/exports/{jobId}"] = {"get": {"responses": {"200": {"description": "ok"}}}}
    result = diff(snapshot(_spec()), snapshot(new_spec))
    assert [(e.method, e.template) for e in result.added] == [("GET", "/exports/{jobId}")]
    assert result.added[0].max_severity is DiffSeverity.ADDITIVE
    assert result.has_breaking is False


def test_diff_removed_endpoint_is_breaking() -> None:
    old_spec = _spec()
    old_spec["paths"]["/exports/{jobId}"] = {"get": {"responses": {"200": {"description": "ok"}}}}
    result = diff(snapshot(old_spec), snapshot(_spec()))
    assert [(e.method, e.template) for e in result.removed] == [("GET", "/exports/{jobId}")]
    assert result.has_breaking is True


def test_diff_status_removed_is_breaking_status_added_is_additive() -> None:
    new_spec = _spec()
    del new_spec["paths"]["/orders/{orderId}"]["get"]["responses"]["404"]
    result = diff(snapshot(_spec()), snapshot(new_spec))
    assert "status_removed" in {f.kind for f in _status_findings(result, "status_removed")}
    assert result.has_breaking is True

    other = _spec()
    other["paths"]["/orders/{orderId}"]["get"]["responses"]["410"] = {"description": "gone"}
    result = diff(snapshot(_spec()), snapshot(other))
    assert _status_findings(result, "status_added")
    assert result.has_breaking is False


def test_diff_response_property_removed_is_breaking_added_is_additive() -> None:
    new_spec = _spec()
    schema = new_spec["paths"]["/orders"]["post"]["responses"]["202"]["content"][
        "application/json"
    ]["schema"]
    del schema["properties"]["status"]
    result = diff(snapshot(_spec()), snapshot(new_spec))
    removed = _status_findings(result, "property_removed")
    assert any("status" in f.detail for f in removed)
    assert result.has_breaking is True

    other = _spec()
    other["paths"]["/orders"]["post"]["responses"]["202"]["content"]["application/json"]["schema"][
        "properties"
    ]["discountPercent"] = {"type": "number"}
    result = diff(snapshot(_spec()), snapshot(other))
    added = _status_findings(result, "property_added")
    assert any("discountPercent" in f.detail for f in added)
    assert result.has_breaking is False


def test_diff_required_added_is_breaking() -> None:
    new_spec = _spec()
    schema = new_spec["paths"]["/orders"]["post"]["responses"]["202"]["content"][
        "application/json"
    ]["schema"]
    schema["required"] = ["orderId", "status"]
    result = diff(snapshot(_spec()), snapshot(new_spec))
    assert _status_findings(result, "required_added")
    assert result.has_breaking is True


def test_diff_type_changed_is_breaking() -> None:
    new_spec = _spec()
    new_spec["paths"]["/orders"]["post"]["responses"]["202"]["content"]["application/json"][
        "schema"
    ]["properties"]["orderId"] = {"type": "integer"}
    result = diff(snapshot(_spec()), snapshot(new_spec))
    changed = _status_findings(result, "type_changed")
    assert changed, "expected a type_changed finding"
    assert result.has_breaking is True


def test_diff_nested_only_change_is_info() -> None:
    new_spec = _spec()
    new_spec["paths"]["/orders"]["post"]["requestBody"]["content"]["application/json"]["schema"][
        "properties"
    ]["quantity"]["minimum"] = 1
    result = diff(snapshot(_spec()), snapshot(new_spec))
    info = [
        finding
        for endpoint in result.changed
        for finding in endpoint.findings
        if finding.severity is DiffSeverity.INFO
    ]
    assert info, "expected an info-level schema_changed finding"
    assert result.has_breaking is False


def test_snapshot_save_and_load_roundtrip(tmp_path: Path) -> None:
    snap = snapshot(_spec())
    target = tmp_path / "nested" / "spec-snapshot.json"
    save_snapshot(target, snap)
    loaded = load_snapshot(target)
    assert loaded == snap


def test_load_snapshot_missing_or_corrupt_returns_none(tmp_path: Path) -> None:
    assert load_snapshot(tmp_path / "absent.json") is None
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json")
    assert load_snapshot(corrupt) is None
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps([1, 2, 3]))
    assert load_snapshot(wrong) is None


def test_spec_snapshot_type_annotated_for_reuse() -> None:
    # SpecSnapshot is exported so reporting can type hints against it.
    assert isinstance(snapshot(_spec()), SpecSnapshot)


def test_diff_empty_old_digest_short_circuits_not_applied() -> None:
    # A snapshot whose digest was lost (e.g. hand-edited) must still diff structurally.
    old = snapshot(_spec())
    broken = SpecSnapshot(
        openapi_version=old.openapi_version,
        operations=old.operations,
        digest="",
    )
    result = diff(broken, snapshot(_spec()))
    assert not (result.added or result.removed or result.changed)


@pytest.mark.parametrize(
    ("mutate", "expected_kind", "expected_severity"),
    [
        (
            lambda spec: spec["paths"]["/orders"]["post"]["responses"]["202"]["content"][
                "application/json"
            ]["schema"]["properties"].update({"totalAmount": {"type": "number"}}),
            "property_added",
            DiffSeverity.ADDITIVE,
        ),
        (
            lambda spec: spec["paths"]["/orders"]["post"]["requestBody"]["content"][
                "application/json"
            ]["schema"]["properties"].pop("quantity"),
            "property_removed",
            DiffSeverity.BREAKING,
        ),
    ],
)
def test_diff_parametrized_request_field_changes(
    mutate, expected_kind: str, expected_severity: DiffSeverity
) -> None:
    new_spec = _spec()
    mutate(new_spec)
    result = diff(snapshot(_spec()), snapshot(new_spec))
    kinds = {f.kind for endpoint in result.changed for f in endpoint.findings}
    assert expected_kind in kinds
    severities = {
        f.severity
        for endpoint in result.changed
        for f in endpoint.findings
        if f.kind == expected_kind
    }
    assert expected_severity in severities
