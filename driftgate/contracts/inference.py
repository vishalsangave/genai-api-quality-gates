"""Permissive OpenAPI 3.1 bootstrap from recorded DriftGate cassettes.

Schema inference is intentionally conservative. It observes traffic; it never
pretends observed traffic is a complete specification:

- a field is required only if it exists in every sample;
- incompatible sample types widen to a JSON Schema union;
- observed values are documented, never enforced as an ``enum``;
- ambiguous paths stay literal rather than being guessed into templates; and
- output is marked ``x-driftgate-inferred: true`` so ContractValidator keeps
  it advisory until a human promotes it to strict enforcement.
"""

from __future__ import annotations

import copy
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from driftgate.recorder.cassette import CassetteInteraction

JSONLike = Any

_IDENTIFIER_RE = re.compile(
    r"^(?:\d+|[A-Za-z][A-Za-z0-9_]*-\d+|[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12})$"
)
_ISO_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})$"
)
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}$")


@dataclass(frozen=True)
class InferenceWarning:
    message: str


@dataclass(frozen=True)
class InferenceResult:
    spec: dict[str, Any]
    warnings: tuple[InferenceWarning, ...]


def _segments(path: str) -> list[str]:
    return [segment for segment in path.split("?")[0].strip("/").split("/") if segment]


def cluster_path_templates(paths: Sequence[tuple[str, str]]) -> dict[tuple[str, str], list[str]]:
    """Cluster concrete paths into conservative inferred path templates.

    A varying segment becomes ``{param_n}`` only if every observed value at
    that position is identifier-like and at least two distinct values exist.
    This avoids converting a meaningful varying word (for example, a route
    action) into a guessed path parameter.
    """
    grouped: dict[tuple[str, int], list[str]] = defaultdict(list)
    for method, path in paths:
        grouped[(method.upper(), len(_segments(path)))].append(path.split("?", 1)[0])

    result: dict[tuple[str, str], list[str]] = {}
    for (method, _count), members in grouped.items():
        segment_columns = (
            list(zip(*(_segments(path) for path in members), strict=True)) if members else []
        )
        template_parts: list[str] = []
        for index, values in enumerate(segment_columns):
            unique = set(values)
            if len(unique) >= 2 and all(_IDENTIFIER_RE.fullmatch(v) for v in unique):
                template_parts.append(f"{{param_{index + 1}}}")
            else:
                # Ambiguous cluster -> keep literal paths, never guess.
                template_parts.append(values[0])
        template = "/" + "/".join(template_parts)
        ambiguous = any(
            len(set(values)) > 1 and not all(_IDENTIFIER_RE.fullmatch(v) for v in set(values))
            for values in segment_columns
        )
        if ambiguous:
            for member in members:
                result[(method, member)] = [member]
        else:
            result[(method, template)] = members
    return result


def _type_of(value: JSONLike) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    return "string"


def _format_hint(values: Sequence[str]) -> str | None:
    nonempty = [v for v in values if v]
    if not nonempty:
        return None
    if all(_UUID_RE.fullmatch(v) for v in nonempty):
        return "uuid"
    if all(_ISO_DATETIME_RE.fullmatch(v) for v in nonempty):
        return "date-time"
    if all(_EMAIL_RE.fullmatch(v) for v in nonempty):
        return "email"
    return None


def infer_json_schema(
    samples: Sequence[JSONLike], *, min_samples_for_enum: int = 8
) -> dict[str, Any]:
    """Infer a permissive JSON Schema from observed values.

    ``min_samples_for_enum`` exists for future policy tuning but is deliberately
    not used to emit enums in this MVP: values are never made restrictive from
    traffic alone. Small observed sets become an informational description.
    """
    del min_samples_for_enum
    if not samples:
        return {}
    types = {_type_of(sample) for sample in samples}
    schema: dict[str, Any] = {"type": next(iter(types)) if len(types) == 1 else sorted(types)}

    if types == {"object"}:
        mappings = [sample for sample in samples if isinstance(sample, Mapping)]
        keys = sorted({str(key) for item in mappings for key in item})
        properties: dict[str, Any] = {}
        required: list[str] = []
        for key in keys:
            values = [item[key] for item in mappings if key in item]
            properties[key] = infer_json_schema(values)
            if len(values) == len(mappings):
                required.append(key)
        schema["properties"] = properties
        if required:
            schema["required"] = required
        # Open by default: an observation, not a design contract.
        schema["additionalProperties"] = True
    elif types == {"array"}:
        arrays = [sample for sample in samples if isinstance(sample, list)]
        elements = [element for array in arrays for element in array]
        if elements:
            schema["items"] = infer_json_schema(elements)
    elif types == {"string"}:
        strings = [str(sample) for sample in samples]
        format_hint = _format_hint(strings)
        if format_hint:
            schema["format"] = format_hint
        distinct = sorted(set(strings))
        if 1 < len(distinct) <= 8:
            schema["description"] = "Observed values (informational only): " + ", ".join(
                repr(value) for value in distinct
            )
    return schema


def _request_schema(interactions: Sequence[CassetteInteraction]) -> dict[str, Any] | None:
    samples = [i.request.body for i in interactions if i.request.body is not None]
    return infer_json_schema(samples) if samples else None


def _response_schema(interactions: Sequence[CassetteInteraction]) -> dict[str, Any] | None:
    samples = [i.response.body for i in interactions if i.response.body is not None]
    return infer_json_schema(samples) if samples else None


def infer_openapi_spec(
    interactions: Sequence[CassetteInteraction], *, min_samples: int = 3
) -> InferenceResult:
    """Build an advisory OpenAPI 3.1 document from recorded interactions."""
    warnings: list[InferenceWarning] = []
    paths = cluster_path_templates([(i.request.method, i.request.path) for i in interactions])
    inferred_paths: dict[str, Any] = {}

    for (method, template), concrete_paths in paths.items():
        source = [
            interaction
            for interaction in interactions
            if interaction.request.method.upper() == method
            and interaction.request.path in concrete_paths
        ]
        if len(source) < min_samples:
            warnings.append(
                InferenceWarning(
                    f"Skipped {method} {template}: only {len(source)} interaction(s), "
                    f"need at least {min_samples}"
                )
            )
            continue
        operation: dict[str, Any] = {
            "x-driftgate-inferred": True,
            "responses": {},
        }
        request_schema = _request_schema(source)
        if request_schema is not None:
            operation["requestBody"] = {
                "required": False,
                "content": {"application/json": {"schema": request_schema}},
            }
        by_status: dict[int, list[CassetteInteraction]] = defaultdict(list)
        for interaction in source:
            by_status[interaction.response.status].append(interaction)
        for status, status_interactions in sorted(by_status.items()):
            content: dict[str, Any] = {}
            schema = _response_schema(status_interactions)
            if schema is not None:
                content["application/json"] = {"schema": schema}
            operation["responses"][str(status)] = {
                "description": f"Observed HTTP {status} response (inferred)",
                **({"content": content} if content else {}),
            }
        inferred_paths.setdefault(template, {})[method.lower()] = operation

    spec = {
        "openapi": "3.1.0",
        "info": {
            "title": "DriftGate inferred contract",
            "version": datetime.now(UTC).strftime("%Y%m%d%H%M%S"),
            "description": (
                "Permissive contract inferred from recorded traffic. Review and run "
                "`driftgate infer-schema --promote` before using as a strict CI gate."
            ),
        },
        "x-driftgate-inferred": True,
        "paths": inferred_paths,
    }
    return InferenceResult(spec=spec, warnings=tuple(warnings))


def promote_inferred_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Remove advisory markers, making an inferred contract strict by intent."""
    promoted = copy.deepcopy(dict(spec))
    promoted.pop("x-driftgate-inferred", None)
    for path_item in promoted.get("paths", {}).values():
        if not isinstance(path_item, Mapping):
            continue
        for operation in path_item.values():
            if isinstance(operation, dict):
                operation.pop("x-driftgate-inferred", None)
    return promoted


def load_cassette_interactions(directory: str) -> list[CassetteInteraction]:
    """Load every YAML cassette below a directory without modifying it."""
    from pathlib import Path

    import yaml

    from driftgate.recorder.cassette import Cassette

    interactions: list[CassetteInteraction] = []
    for path in sorted(Path(directory).rglob("*.y*ml")):
        raw = yaml.safe_load(path.read_text())
        if raw:
            interactions.extend(Cassette.model_validate(raw).interactions)
    return interactions
