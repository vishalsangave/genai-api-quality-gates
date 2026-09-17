"""Precompiled OpenAPI 3.1 / JSON Schema contract validation."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from driftgate.recorder.cassette import CassetteInteraction


class ContractViolationError(AssertionError):
    """Raised when :meth:`ContractResult.raise_if_failed` sees an error."""


@dataclass(frozen=True)
class ContractViolation:
    severity: Literal["error", "warning"]
    kind: Literal["undeclared_status", "request_body", "response_body", "missing_header", "path"]
    message: str
    schema_path: str = ""
    instance_path: str = ""


@dataclass
class ContractResult:
    method: str
    path: str
    status: int
    violations: list[ContractViolation]

    @property
    def ok(self) -> bool:
        return not any(v.severity == "error" for v in self.violations)

    def raise_if_failed(self) -> None:
        errors = [v for v in self.violations if v.severity == "error"]
        if not errors:
            return
        formatted = "\n".join(
            f"- [{v.kind}] {v.message} (instance={v.instance_path or '/'}, schema={v.schema_path or '/'})"
            for v in errors
        )
        raise ContractViolationError(
            f"Contract validation failed for {self.method} {self.path} → {self.status}:\n{formatted}"
        )


@dataclass(frozen=True)
class _Operation:
    template: str
    method: str
    definition: Mapping[str, Any]
    regex: re.Pattern[str]


class ContractValidator:
    """Validate request/response payloads against a precompiled OpenAPI 3.1 spec."""

    def __init__(self, spec: Mapping[str, Any], *, strict_status: bool = True) -> None:
        if str(spec.get("openapi", "")).split(".")[0] != "3":
            raise ValueError("ContractValidator requires an OpenAPI 3.x document")
        self.spec: Mapping[str, Any] = spec
        self.strict_status = strict_status
        self._inferred = bool(spec.get("x-driftgate-inferred", False))
        # An OpenAPI document itself is not a JSON Schema (it has no `$schema`),
        # so referencing cannot auto-detect a specification here. Its embedded
        # request/response schemas are Draft 2020-12 by OpenAPI 3.1 contract.
        resource = Resource(contents=spec, specification=DRAFT202012)
        self._registry = Registry().with_resource("urn:driftgate:openapi", resource)
        self._operations = self._compile_operations()
        self._request_validators: dict[tuple[str, str], Draft202012Validator] = {}
        self._response_validators: dict[tuple[str, str, str], Draft202012Validator] = {}
        self._precompile_validators()

    @classmethod
    def from_file(cls, path: str | Path, *, strict_status: bool = True) -> ContractValidator:
        path = Path(path)
        raw = path.read_text()
        if path.suffix.lower() == ".json":
            spec = json.loads(raw)
        else:
            spec = yaml.safe_load(raw)
        if not isinstance(spec, Mapping):
            raise ValueError(f"OpenAPI document {path} must be a mapping")
        return cls(spec, strict_status=strict_status)

    def _compile_operations(self) -> list[_Operation]:
        operations: list[_Operation] = []
        for template, path_item in self.spec.get("paths", {}).items():
            if not isinstance(path_item, Mapping):
                continue
            regex_source = "^" + re.sub(r"\{[^}]+\}", r"[^/]+", template) + "$"
            for method, definition in path_item.items():
                if method.lower() not in {
                    "get",
                    "post",
                    "put",
                    "patch",
                    "delete",
                    "head",
                    "options",
                }:
                    continue
                if isinstance(definition, Mapping):
                    operations.append(
                        _Operation(template, method.upper(), definition, re.compile(regex_source))
                    )
        return operations

    def _validator_for_schema(self, schema: Mapping[str, Any]) -> Draft202012Validator:
        return Draft202012Validator(schema, registry=self._registry, format_checker=FormatChecker())

    @staticmethod
    def _json_schema(content: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        if not content:
            return None
        json_media = content.get("application/json")
        if isinstance(json_media, Mapping) and isinstance(json_media.get("schema"), Mapping):
            return json_media["schema"]
        # OpenAPI content types can declare vendor +json media types.
        for media_type, media in content.items():
            if (
                media_type.endswith("+json")
                and isinstance(media, Mapping)
                and isinstance(media.get("schema"), Mapping)
            ):
                return media["schema"]
        return None

    def _precompile_validators(self) -> None:
        for op in self._operations:
            request_schema = self._json_schema(op.definition.get("requestBody", {}).get("content"))
            if request_schema is not None:
                self._request_validators[(op.method, op.template)] = self._validator_for_schema(
                    request_schema
                )
            for status, response in op.definition.get("responses", {}).items():
                if not isinstance(response, Mapping):
                    continue
                response_schema = self._json_schema(response.get("content"))
                if response_schema is not None:
                    self._response_validators[(op.method, op.template, str(status))] = (
                        self._validator_for_schema(response_schema)
                    )

    def _operation(self, method: str, path: str) -> _Operation | None:
        normalized = path.split("?", 1)[0]
        upper = method.upper()
        return next(
            (
                op
                for op in self._operations
                if op.method == upper and op.regex.fullmatch(normalized)
            ),
            None,
        )

    @staticmethod
    def _pointer(parts: Any) -> str:
        return "/" + "/".join(str(p).replace("~", "~0").replace("/", "~1") for p in parts)

    def _severity(self, op: _Operation | None) -> Literal["error", "warning"]:
        if self._inferred or (op is not None and bool(op.definition.get("x-driftgate-inferred"))):
            return "warning"
        return "error"

    def _schema_errors(
        self,
        validator: Draft202012Validator,
        instance: Any,
        *,
        kind: Literal["request_body", "response_body"],
        severity: Literal["error", "warning"],
    ) -> list[ContractViolation]:
        return [
            ContractViolation(
                severity=severity,
                kind=kind,
                message=error.message,
                schema_path=self._pointer(error.absolute_schema_path),
                instance_path=self._pointer(error.absolute_path),
            )
            for error in sorted(
                validator.iter_errors(instance), key=lambda e: list(e.absolute_path)
            )
        ]

    def validate_request(
        self,
        method: str,
        path: str,
        *,
        body: Any = None,
        query: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> list[ContractViolation]:
        del query, headers  # body contract is the current MVP boundary
        op = self._operation(method, path)
        if op is None:
            return [
                ContractViolation(
                    "error", "path", f"No OpenAPI operation matches {method.upper()} {path}"
                )
            ]
        validator = self._request_validators.get((op.method, op.template))
        if validator is None or body is None:
            return []
        return self._schema_errors(
            validator, body, kind="request_body", severity=self._severity(op)
        )

    def validate_response(
        self,
        method: str,
        path: str,
        status: int,
        *,
        body: Any = None,
        headers: Mapping[str, str] | None = None,
    ) -> list[ContractViolation]:
        del headers
        op = self._operation(method, path)
        if op is None:
            return [
                ContractViolation(
                    "error", "path", f"No OpenAPI operation matches {method.upper()} {path}"
                )
            ]
        responses = op.definition.get("responses", {})
        status_key = str(status)
        response_definition = responses.get(status_key) or responses.get("default")
        if response_definition is None:
            if not self.strict_status:
                return []
            return [
                ContractViolation(
                    self._severity(op),
                    "undeclared_status",
                    f"HTTP {status} is not declared for {op.method} {op.template}",
                )
            ]
        validator = self._response_validators.get((op.method, op.template, status_key))
        if validator is None or body is None:
            return []
        return self._schema_errors(
            validator, body, kind="response_body", severity=self._severity(op)
        )

    def check_interaction(self, interaction: CassetteInteraction) -> ContractResult:
        req = interaction.request
        res = interaction.response
        violations = self.validate_request(req.method, req.path, body=req.body)
        violations += self.validate_response(
            req.method, req.path, res.status, body=res.body, headers=res.headers
        )
        return ContractResult(req.method, req.path, res.status, violations)
