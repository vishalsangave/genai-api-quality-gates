"""Golden datasets: check an existing file, or generate one from the application.

A golden dataset is JSONL, one ``{"prompt": <request body>, "response": <approved
response body>}`` object per line. ``check_dataset`` reports every problem with
its line number; ``generate`` builds a dataset from what the application under
test actually accepts and returns (recorded cassettes, its OpenAPI spec, or a
list of prompts sent to the running service).
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

CHECK_HINT = "Run `driftgate dataset check {path}` to diagnose it."
GENERATE_HINT = (
    "Run `driftgate dataset generate --out {path}` (from cassettes, an OpenAPI spec, "
    "or a prompts file) to create one for your application."
)


@dataclass(frozen=True)
class DatasetIssue:
    line: int
    severity: str  # "error" breaks the dataset; "warning" is cosmetic (e.g. duplicate)
    kind: str
    message: str

    def __str__(self) -> str:
        return f"line {self.line}: {self.kind}: {self.message}"


@dataclass
class DatasetReport:
    path: Path
    examples: list[dict[str, Any]] = field(default_factory=list)  # valid, de-duplicated
    issues: list[DatasetIssue] = field(default_factory=list)
    lines_seen: int = 0

    @property
    def errors(self) -> list[DatasetIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors and bool(self.examples)

    def summary(self, limit: int = 5) -> str:
        parts = [str(i) for i in self.errors[:limit]]
        if len(self.errors) > limit:
            parts.append(f"... and {len(self.errors) - limit} more")
        if not self.examples and not self.errors:
            parts.append("dataset has no examples")
        return "; ".join(parts)


def _canonical(example: Mapping[str, Any]) -> str:
    return json.dumps(example, sort_keys=True, ensure_ascii=False)


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, (str, list, dict)) and len(value) == 0)


def check_dataset(path: str | Path) -> DatasetReport:
    """Validate a golden dataset file; never raises for bad content."""
    target = Path(path)
    report = DatasetReport(path=target)
    try:
        text = target.read_text()
    except OSError as error:
        report.issues.append(DatasetIssue(0, "error", "unreadable", str(error)))
        return report

    seen: set[str] = set()
    for number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        report.lines_seen += 1
        try:
            item = json.loads(raw)
        except json.JSONDecodeError as error:
            report.issues.append(DatasetIssue(number, "error", "invalid_json", error.msg))
            continue
        if not isinstance(item, dict):
            report.issues.append(
                DatasetIssue(number, "error", "not_an_object", f"got {type(item).__name__}")
            )
            continue
        missing = [key for key in ("prompt", "response") if key not in item]
        if missing:
            report.issues.append(
                DatasetIssue(number, "error", "missing_key", f"missing {', '.join(missing)}")
            )
            continue
        blank = [key for key in ("prompt", "response") if _is_blank(item[key])]
        if blank:
            report.issues.append(
                DatasetIssue(number, "error", "empty_value", f"{', '.join(blank)} is empty")
            )
            continue
        key = _canonical({"prompt": item["prompt"], "response": item["response"]})
        if key in seen:
            report.issues.append(
                DatasetIssue(number, "warning", "duplicate", "same prompt and response as an earlier line")
            )
            continue
        seen.add(key)
        report.examples.append({"prompt": item["prompt"], "response": item["response"]})
    return report


def fix_dataset(path: str | Path) -> tuple[DatasetReport, Path | None]:
    """Rewrite the file keeping only valid, unique examples; the original is saved as ``.bak``."""
    target = Path(path)
    report = check_dataset(target)
    if not report.issues or not target.exists():
        return report, None
    backup = target.with_suffix(target.suffix + ".bak")
    backup.write_text(target.read_text())
    target.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in report.examples))
    return report, backup


def append_unique(path: str | Path, examples: Iterable[Mapping[str, Any]]) -> tuple[int, int]:
    """Append examples not already present. Returns ``(added, duplicates_skipped)``."""
    target = Path(path)
    existing = {_canonical(e) for e in check_dataset(target).examples} if target.exists() else set()
    fresh: list[Mapping[str, Any]] = []
    skipped = 0
    for example in examples:
        key = _canonical({"prompt": example["prompt"], "response": example["response"]})
        if key in existing:
            skipped += 1
            continue
        existing.add(key)
        fresh.append(example)
    if fresh:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a") as handle:
            for example in fresh:
                handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    return len(fresh), skipped


# --- Generation: what the application accepts -------------------------------------------


def _resolve(schema: Mapping[str, Any], spec: Mapping[str, Any], depth: int = 0) -> Mapping[str, Any]:
    ref = schema.get("$ref")
    if not ref or depth > 8:
        return schema
    if not ref.startswith("#/"):
        raise ValueError(f"only local $ref is supported, got {ref!r}")
    node: Any = spec
    for part in ref[2:].split("/"):
        node = node[part.replace("~1", "/").replace("~0", "~")]
    return _resolve(node, spec, depth + 1)


def sample_from_schema(schema: Mapping[str, Any], spec: Mapping[str, Any], depth: int = 0) -> Any:
    """A minimal valid value: example > default > const > enum[0] > type-based placeholder."""
    schema = _resolve(schema, spec)
    for key in ("example", "default", "const"):
        if key in schema:
            return copy.deepcopy(schema[key])
    if schema.get("examples"):
        return copy.deepcopy(schema["examples"][0])
    if schema.get("enum"):
        return copy.deepcopy(schema["enum"][0])
    for combiner in ("oneOf", "anyOf"):
        if schema.get(combiner):
            return sample_from_schema(schema[combiner][0], spec, depth + 1)
    if schema.get("allOf"):
        merged: dict[str, Any] = {}
        for part in schema["allOf"]:
            value = sample_from_schema(part, spec, depth + 1)
            if isinstance(value, dict):
                merged.update(value)
        return merged
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((t for t in kind if t != "null"), None)
    if kind == "object" or "properties" in schema:
        if depth > 6:
            return {}
        props = schema.get("properties", {})
        required = schema.get("required", list(props))
        return {k: sample_from_schema(props[k], spec, depth + 1) for k in required if k in props}
    if kind == "array":
        count = max(1, int(schema.get("minItems", 1)))
        item = schema.get("items", {})
        return [sample_from_schema(item, spec, depth + 1) for _ in range(count)]
    if kind == "integer":
        return int(schema.get("minimum", 1))
    if kind == "number":
        return float(schema.get("minimum", 1))
    if kind == "boolean":
        return True
    fmt = schema.get("format")
    return {
        "date-time": "2026-01-01T00:00:00Z",
        "date": "2026-01-01",
        "email": "user@example.com",
        "uuid": "123e4567-e89b-12d3-a456-426614174000",
        "uri": "https://example.com",
    }.get(fmt, "example")


def body_operations(spec: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Every (METHOD, path) in the spec that takes a JSON request body."""
    found: list[tuple[str, str]] = []
    for path, item in (spec.get("paths") or {}).items():
        for method, op in item.items():
            if method.lower() in {"post", "put", "patch"} and isinstance(op, dict):
                content = (op.get("requestBody") or {}).get("content", {})
                if any("json" in media for media in content):
                    found.append((method.upper(), path))
    return found


def request_bodies_from_spec(
    spec: Mapping[str, Any], method: str, path: str
) -> list[Any]:
    """Explicit spec examples first, then one schema-derived sample, de-duplicated."""
    try:
        operation = spec["paths"][path][method.lower()]
        content = operation["requestBody"]["content"]
    except KeyError as error:
        raise ValueError(f"{method} {path} has no JSON request body in the spec") from error
    media = next((m for name, m in content.items() if "json" in name), None)
    if media is None:
        raise ValueError(f"{method} {path} has no JSON request body in the spec")
    bodies: list[Any] = []
    if "example" in media:
        bodies.append(copy.deepcopy(media["example"]))
    for example in (media.get("examples") or {}).values():
        if isinstance(example, dict) and "value" in example:
            bodies.append(copy.deepcopy(example["value"]))
    if media.get("schema"):
        bodies.append(sample_from_schema(media["schema"], spec))
    unique: dict[str, Any] = {}
    for body in bodies:
        unique.setdefault(_canonical({"b": body}), body)
    return list(unique.values())


def render_template(template: Any, prompt: str) -> Any:
    """Replace ``{prompt}`` in every string of a JSON template (any nesting)."""
    if isinstance(template, str):
        return template.replace("{prompt}", prompt)
    if isinstance(template, list):
        return [render_template(item, prompt) for item in template]
    if isinstance(template, dict):
        return {k: render_template(v, prompt) for k, v in template.items()}
    return template


def examples_from_cassettes(
    location: str | Path, *, method: str | None = None, endpoint: str | None = None
) -> list[dict[str, Any]]:
    """Successful recorded exchanges that have a request and a response body."""
    from driftgate.contracts.inference import load_cassette_interactions
    from driftgate.recorder.cassette import Cassette

    path = Path(location)
    if path.is_dir():
        interactions = load_cassette_interactions(str(path))
    else:
        raw = yaml.safe_load(path.read_text())
        interactions = Cassette.model_validate(raw).interactions if raw else []
    examples: list[dict[str, Any]] = []
    for item in interactions:
        if method and item.request.method.upper() != method.upper():
            continue
        if endpoint and item.request.path != endpoint:
            continue
        if not 200 <= item.response.status < 300:
            continue
        if item.request.body is None or item.response.body is None:
            continue
        examples.append({"prompt": item.request.body, "response": item.response.body})
    return examples


@dataclass
class GenerationResult:
    examples: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def _response_body(response: httpx.Response) -> Any:
    if "json" in response.headers.get("content-type", ""):
        return response.json()
    return response.text


def generate_from_service(
    bodies: Sequence[Any],
    *,
    method: str,
    endpoint: str,
    client: httpx.Client,
    headers: Mapping[str, str] | None = None,
    on_request: Callable[[int, int], None] | None = None,
) -> GenerationResult:
    """Send each request body to the running service; keep only 2xx exchanges."""
    result = GenerationResult()
    for index, body in enumerate(bodies, start=1):
        if on_request:
            on_request(index, len(bodies))
        try:
            response = client.request(method, endpoint, json=body, headers=dict(headers or {}))
        except httpx.HTTPError as error:
            result.skipped.append(f"request {index}: {error}")
            continue
        if not response.is_success:
            result.skipped.append(f"request {index}: HTTP {response.status_code}")
            continue
        try:
            payload = _response_body(response)
        except ValueError as error:
            result.skipped.append(f"request {index}: unreadable response ({error})")
            continue
        if _is_blank(payload):
            result.skipped.append(f"request {index}: empty response body")
            continue
        result.examples.append({"prompt": body, "response": payload})
    return result
