"""Structural OpenAPI snapshot + diff between builds.

Pure functions over plain OpenAPI documents: ``snapshot`` normalizes a spec
into a digestible structure, ``diff`` classifies the delta between two
snapshots as breaking / additive / info. Both are conservative — only
certain, structural facts are flagged; no semantic analysis.

The artifact chain lives in the pytest sessionfinish pipeline
(``driftgate.reporting``): the previous build's snapshot is loaded from
``.driftgate/spec-snapshot.json``, diffed against the current spec, then the
new snapshot overwrites the file. See docs/ARCHITECTURE.md "Build report".
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

HTTP_METHODS = frozenset({"get", "post", "put", "patch", "delete", "head", "options"})


class DiffSeverity(StrEnum):
    BREAKING = "breaking"
    ADDITIVE = "additive"
    INFO = "info"


@dataclass(frozen=True)
class StatusSnapshot:
    schema_digest: str
    schema_type: str
    properties: tuple[str, ...]
    required: tuple[str, ...]
    property_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class OperationSnapshot:
    method: str
    template: str
    request_properties: tuple[str, ...]
    request_required: tuple[str, ...]
    request_schema_digest: str
    request_property_types: tuple[str, ...] = ()
    statuses: Mapping[str, StatusSnapshot] = field(default_factory=dict)


@dataclass(frozen=True)
class SpecSnapshot:
    openapi_version: str
    operations: tuple[OperationSnapshot, ...]
    digest: str


@dataclass(frozen=True)
class DiffFinding:
    location: str
    kind: str
    severity: DiffSeverity
    detail: str


@dataclass(frozen=True)
class EndpointDiff:
    method: str
    template: str
    change: str
    findings: tuple[DiffFinding, ...]
    max_severity: DiffSeverity


@dataclass(frozen=True)
class SpecDiff:
    added: tuple[EndpointDiff, ...]
    removed: tuple[EndpointDiff, ...]
    changed: tuple[EndpointDiff, ...]
    first_snapshot: bool
    has_breaking: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_snapshot": self.first_snapshot,
            "has_breaking": self.has_breaking,
            "added": [_endpoint_dict(e) for e in self.added],
            "removed": [_endpoint_dict(e) for e in self.removed],
            "changed": [_endpoint_dict(e) for e in self.changed],
        }


def _endpoint_dict(endpoint: EndpointDiff) -> dict[str, Any]:
    return {
        "method": endpoint.method,
        "template": endpoint.template,
        "change": endpoint.change,
        "max_severity": endpoint.max_severity.value,
        "findings": [
            {
                "location": f.location,
                "kind": f.kind,
                "severity": f.severity.value,
                "detail": f.detail,
            }
            for f in endpoint.findings
        ],
    }


def _canonical_digest(value: Any) -> str:
    if not isinstance(value, Mapping):
        return ""
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _json_schema(content: Any) -> Mapping[str, Any] | None:
    """Same media-type rule as ContractValidator: application/json, then +json."""
    if not isinstance(content, Mapping) or not content:
        return None
    json_media = content.get("application/json")
    if isinstance(json_media, Mapping) and isinstance(json_media.get("schema"), Mapping):
        return json_media["schema"]
    for media_type, media in content.items():
        if (
            isinstance(media_type, str)
            and media_type.endswith("+json")
            and isinstance(media, Mapping)
            and isinstance(media.get("schema"), Mapping)
        ):
            return media["schema"]
    return None


def _top_level_fields(
    schema: Mapping[str, Any] | None,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Returns (property names, required, per-name property types) — all sorted consistently."""
    if not isinstance(schema, Mapping):
        return (), (), ()
    names: tuple[str, ...] = ()
    properties = schema.get("properties")
    property_types: tuple[str, ...] = ()
    if isinstance(properties, Mapping):
        names = tuple(sorted(str(key) for key in properties))
        property_types = tuple(
            f"{name}:{_schema_type(properties[name] if isinstance(properties[name], Mapping) else None)}"
            for name in names
        )
    required: tuple[str, ...] = ()
    required_value = schema.get("required")
    if isinstance(required_value, list):
        required = tuple(sorted(str(item) for item in required_value))
    return names, required, property_types


def _schema_type(schema: Mapping[str, Any] | None) -> str:
    if not isinstance(schema, Mapping):
        return ""
    type_value = schema.get("type")
    if isinstance(type_value, str):
        return type_value
    if isinstance(type_value, list):
        return "|".join(sorted(str(item) for item in type_value))
    if "properties" in schema or "$ref" in schema:
        return "object"
    return "any"


def _status_snapshots(responses: Any) -> dict[str, StatusSnapshot]:
    statuses: dict[str, StatusSnapshot] = {}
    if not isinstance(responses, Mapping):
        return statuses
    for status, response in responses.items():
        if not isinstance(response, Mapping):
            continue
        schema = _json_schema(response.get("content"))
        properties, required = _top_level_fields(schema)
        statuses[str(status)] = StatusSnapshot(
            schema_digest=_canonical_digest(schema),
            schema_type=_schema_type(schema),
            properties=properties,
            required=required,
        )
    return statuses


def snapshot(spec: Mapping[str, Any]) -> SpecSnapshot:
    operations: list[OperationSnapshot] = []
    paths = spec.get("paths", {})
    if isinstance(paths, Mapping):
        for template, path_item in paths.items():
            if not isinstance(path_item, Mapping):
                continue
            for method, definition in path_item.items():
                if method.lower() not in HTTP_METHODS or not isinstance(definition, Mapping):
                    continue
                request_body = definition.get("requestBody")
                request_schema = _json_schema(
                    request_body.get("content") if isinstance(request_body, Mapping) else None
                )
                request_properties, request_required, request_property_types = _top_level_fields(
                    request_schema
                )
                statuses = _status_snapshot(definition.get("responses"))
                operations.append(
                    OperationSnapshot(
                        method=method.upper(),
                        template=str(template),
                        request_properties=request_properties,
                        request_required=request_required,
                        request_schema_digest=_canonical_digest(request_schema),
                        request_property_types=request_property_types,
                        statuses=statuses,
                    )
                )
    operations.sort(key=lambda op: (op.method, op.template))
    version = spec.get("openapi", "")
    version = version if isinstance(version, str) else ""
    return SpecSnapshot(
        openapi_version=version,
        operations=tuple(operations),
        digest=_canonical_digest(snapshot_to_dict(SpecSnapshot(version, tuple(operations), ""))),
    )


def _status_snapshot(responses: Any) -> dict[str, StatusSnapshot]:
    statuses: dict[str, StatusSnapshot] = {}
    if not isinstance(responses, Mapping):
        return statuses
    for status, response in responses.items():
        if not isinstance(response, Mapping):
            continue
        schema = _json_schema(response.get("content"))
        properties, required, property_types = _top_level_fields(schema)
        statuses[str(status)] = StatusSnapshot(
            schema_digest=_canonical_digest(schema),
            schema_type=_schema_type(schema),
            properties=properties,
            required=required,
            property_types=property_types,
        )
    return statuses


def snapshot_to_dict(snap: SpecSnapshot) -> dict[str, Any]:
    return {
        "openapi_version": snap.openapi_version,
        "digest": snap.digest,
        "operations": [_operation_dict(op) for op in snap.operations],
    }


def _operation_dict(op: OperationSnapshot) -> dict[str, Any]:
    return {
        "method": op.method,
        "template": op.template,
        "request_properties": list(op.request_properties),
        "request_required": list(op.request_required),
        "request_schema_digest": op.request_schema_digest,
        "request_property_types": list(op.request_property_types),
        "statuses": {
            status: {
                "schema_digest": status_snap.schema_digest,
                "schema_type": status_snap.schema_type,
                "properties": list(status_snap.properties),
                "required": list(status_snap.required),
                "property_types": list(status_snap.property_types),
            }
            for status, status_snap in sorted(op.statuses.items())
        },
    }


def snapshot_from_dict(data: Mapping[str, Any]) -> SpecSnapshot:
    operations = tuple(
        OperationSnapshot(
            method=op["method"],
            template=op["template"],
            request_properties=tuple(op.get("request_properties", ())),
            request_required=tuple(op.get("request_required", ())),
            request_schema_digest=op.get("request_schema_digest", ""),
            request_property_types=tuple(op.get("request_property_types", ())),
            statuses={
                status: StatusSnapshot(
                    schema_digest=status_snap.get("schema_digest", ""),
                    schema_type=status_snap.get("schema_type", ""),
                    properties=tuple(status_snap.get("properties", ())),
                    required=tuple(status_snap.get("required", ())),
                    property_types=tuple(status_snap.get("property_types", ())),
                )
                for status, status_snap in op.get("statuses", {}).items()
            },
        )
        for op in data.get("operations", [])
    )
    return SpecSnapshot(
        openapi_version=data.get("openapi_version", ""),
        operations=operations,
        digest=data.get("digest", ""),
    )


def load_snapshot(path: Path) -> SpecSnapshot | None:
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        print(f"driftgate: could not read spec snapshot {path}: {error}", file=sys.stderr)
        return None
    if not isinstance(loaded, dict):
        return None
    try:
        return snapshot_from_dict(loaded)
    except (KeyError, TypeError) as error:
        print(f"driftgate: malformed spec snapshot {path}: {error}", file=sys.stderr)
        return None


def save_snapshot(path: Path, snap: SpecSnapshot) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(snapshot_to_dict(snap), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _severity(findings: list[DiffFinding]) -> DiffSeverity:
    order = {DiffSeverity.BREAKING: 0, DiffSeverity.ADDITIVE: 1, DiffSeverity.INFO: 2}
    if not findings:
        return DiffSeverity.INFO
    return min(findings, key=lambda f: order[f.severity]).severity


def diff(old: SpecSnapshot | None, new: SpecSnapshot) -> SpecDiff:
    if old is None:
        return SpecDiff(added=(), removed=(), changed=(), first_snapshot=True, has_breaking=False)
    if old.digest and old.digest == new.digest:
        return SpecDiff(added=(), removed=(), changed=(), first_snapshot=False, has_breaking=False)

    old_ops = {(op.method, op.template): op for op in old.operations}
    new_ops = {(op.method, op.template): op for op in new.operations}
    added: list[EndpointDiff] = []
    removed: list[EndpointDiff] = []
    changed: list[EndpointDiff] = []

    for method, template in new_ops:
        if (method, template) not in old_ops:
            finding = DiffFinding(
                location=f"{method} {template}",
                kind="endpoint_added",
                severity=DiffSeverity.ADDITIVE,
                detail="Endpoint added",
            )
            added.append(EndpointDiff(method, template, "added", (finding,), DiffSeverity.ADDITIVE))

    for method, template in old_ops:
        if (method, template) not in new_ops:
            finding = DiffFinding(
                location=f"{method} {template}",
                kind="endpoint_removed",
                severity=DiffSeverity.BREAKING,
                detail="Endpoint removed",
            )
            removed.append(
                EndpointDiff(method, template, "removed", (finding,), DiffSeverity.BREAKING)
            )

    for key in sorted(set(old_ops) & set(new_ops)):
        old_op = old_ops[key]
        new_op = new_ops[key]
        findings = _request_findings(old_op, new_op) + _status_findings(old_op, new_op)
        if findings:
            changed.append(
                EndpointDiff(
                    method=new_op.method,
                    template=new_op.template,
                    change="changed",
                    findings=tuple(findings),
                    max_severity=_severity(findings),
                )
            )

    all_endpoints = (*added, *removed, *changed)
    return SpecDiff(
        added=tuple(sorted(added, key=lambda e: (e.method, e.template))),
        removed=tuple(sorted(removed, key=lambda e: (e.method, e.template))),
        changed=tuple(sorted(changed, key=lambda e: (e.method, e.template))),
        first_snapshot=False,
        has_breaking=any(e.max_severity is DiffSeverity.BREAKING for e in all_endpoints),
    )


def _request_findings(old: OperationSnapshot, new: OperationSnapshot) -> list[DiffFinding]:
    findings: list[DiffFinding] = []
    location = f"{new.method} {new.template} request"
    for name in sorted(set(old.request_properties) - set(new.request_properties)):
        findings.append(
            DiffFinding(
                location, "property_removed", DiffSeverity.BREAKING, f"property '{name}' removed"
            )
        )
    for name in sorted(set(new.request_properties) - set(old.request_properties)):
        findings.append(
            DiffFinding(
                location, "property_added", DiffSeverity.ADDITIVE, f"property '{name}' added"
            )
        )
    for name in sorted(set(new.request_required) - set(old.request_required)):
        findings.append(
            DiffFinding(
                location, "required_added", DiffSeverity.BREAKING, f"'{name}' became required"
            )
        )
    old_types = dict(entry.split(":", 1) for entry in old.request_property_types if ":" in entry)
    new_types = dict(entry.split(":", 1) for entry in new.request_property_types if ":" in entry)
    for name in sorted(set(old_types) & set(new_types)):
        if old_types[name] != new_types[name]:
            findings.append(
                DiffFinding(
                    location,
                    "type_changed",
                    DiffSeverity.BREAKING,
                    f"request property '{name}' type changed "
                    f"{old_types[name]!r} → {new_types[name]!r}",
                )
            )
    if (
        old.request_schema_digest
        and new.request_schema_digest
        and old.request_schema_digest != new.request_schema_digest
        and not findings
    ):
        findings.append(
            DiffFinding(
                location,
                "schema_changed",
                DiffSeverity.INFO,
                f"request schema changed ({old.request_schema_digest[:8]}… → {new.request_schema_digest[:8]}…)",
            )
        )
    return findings


def _status_findings(old: OperationSnapshot, new: OperationSnapshot) -> list[DiffFinding]:
    findings: list[DiffFinding] = []
    old_statuses = set(old.statuses)
    new_statuses = set(new.statuses)
    for status in sorted(old_statuses - new_statuses):
        findings.append(
            DiffFinding(
                f"{new.method} {new.template} {status}",
                "status_removed",
                DiffSeverity.BREAKING,
                f"status {status} removed",
            )
        )
    for status in sorted(new_statuses - old_statuses):
        findings.append(
            DiffFinding(
                f"{new.method} {new.template} {status}",
                "status_added",
                DiffSeverity.ADDITIVE,
                f"status {status} added",
            )
        )
    for status in sorted(old_statuses & new_statuses):
        old_snap = old.statuses[status]
        new_snap = new.statuses[status]
        location = f"{new.method} {new.template} {status}"
        for name in sorted(set(old_snap.properties) - set(new_snap.properties)):
            findings.append(
                DiffFinding(
                    location,
                    "property_removed",
                    DiffSeverity.BREAKING,
                    f"response property '{name}' removed",
                )
            )
        for name in sorted(set(new_snap.properties) - set(old_snap.properties)):
            findings.append(
                DiffFinding(
                    location,
                    "property_added",
                    DiffSeverity.ADDITIVE,
                    f"response property '{name}' added",
                )
            )
        for name in sorted(set(new_snap.required) - set(old_snap.required)):
            findings.append(
                DiffFinding(
                    location,
                    "required_added",
                    DiffSeverity.BREAKING,
                    f"response '{name}' became required",
                )
            )
        old_types = dict(entry.split(":", 1) for entry in old_snap.property_types if ":" in entry)
        new_types = dict(entry.split(":", 1) for entry in new_snap.property_types if ":" in entry)
        for name in sorted(set(old_types) & set(new_types)):
            if old_types[name] != new_types[name]:
                findings.append(
                    DiffFinding(
                        location,
                        "type_changed",
                        DiffSeverity.BREAKING,
                        f"response property '{name}' type changed "
                        f"{old_types[name]!r} → {new_types[name]!r}",
                    )
                )
        if not any(f.location == location for f in findings):
            if (
                old_snap.schema_digest
                and new_snap.schema_digest
                and old_snap.schema_digest != new_snap.schema_digest
            ):
                findings.append(
                    DiffFinding(
                        location,
                        "schema_changed",
                        DiffSeverity.INFO,
                        f"response schema changed ({old_snap.schema_digest[:8]}… → {new_snap.schema_digest[:8]}…)",
                    )
                )
    return findings
