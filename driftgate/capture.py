"""Live-traffic capture for ``driftgate capture``.

Replays a user-declared (or default-probe) request list against a running
service through the recording transport, persisting cassettes that
``driftgate infer-schema`` turns into an advisory OpenAPI contract.
Fail-fast on an unreachable service. See docs/ARCHITECTURE.md "Adoption tooling".
"""

from __future__ import annotations

import httpx
import yaml

from driftgate.contracts.inference import (
    InferenceResult,
    infer_openapi_spec,
    load_cassette_interactions,
)
from driftgate.recorder.cassette import CassetteMode, CassetteStore
from driftgate.recorder.fingerprint import FingerprintConfig
from driftgate.recorder.interceptor import RecordingTransport

DEFAULT_PROBE_PATHS = (
    "/",
    "/healthz",
    "/health",
    "/openapi.json",
    "/openapi.yaml",
    "/docs",
    "/api-docs",
    "/status",
)


def parse_auth_header(value: str | None) -> dict[str, str]:
    if not value:
        return {}
    name, _, header_value = value.partition(":")
    return {name.strip(): header_value.strip()}


def load_request_list(path: str) -> list[dict[str, object]]:
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, list) or not loaded:
        raise ValueError(
            "The requests file must be a non-empty YAML list of {method, path, headers, body} entries"
        )
    calls: list[dict[str, object]] = []
    for index, entry in enumerate(loaded):
        if not isinstance(entry, dict) or "path" not in entry:
            raise ValueError(f"Request entry #{index} must be a mapping with at least 'path'")
        calls.append(
            {
                "method": str(entry.get("method", "GET")).upper(),
                "path": str(entry["path"]),
                "headers": {str(k): str(v) for k, v in (entry.get("headers") or {}).items()},
                "body": entry.get("body"),
            }
        )
    return calls


def default_probes() -> list[dict[str, object]]:
    return [
        {"method": "GET", "path": path, "headers": {}, "body": None} for path in DEFAULT_PROBE_PATHS
    ]


def assert_service_reachable(base_url: str, timeout: float = 3.0) -> None:
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            client.get(base_url)
    except httpx.TransportError as error:
        raise RuntimeError(
            f"No service answered at {base_url}: {error}. Start the service first; "
            "DriftGate never starts or stops services itself."
        ) from error


def capture_traffic(
    *,
    base_url: str,
    calls: list[dict[str, object]],
    cassette_path: str,
) -> int:
    """Record every call through the recording transport; returns interaction count."""
    assert_service_reachable(base_url)
    store = CassetteStore(
        cassette_path,
        CassetteMode.RECORD,
        service=base_url,
    )
    transport = RecordingTransport(
        httpx.HTTPTransport(), store, fingerprint_config=FingerprintConfig()
    )
    with httpx.Client(base_url=base_url, transport=transport, timeout=10.0) as client:
        for call in calls:
            response = client.request(
                str(call["method"]),
                str(call["path"]),
                json=call["body"],
                headers=dict(call["headers"]),  # type: ignore[arg-type]
            )
            response.read()
    store.save()
    return len(store.cassette.interactions)


def infer_advisory_spec(cassette_dir: str, *, min_samples: int = 1) -> InferenceResult:
    interactions = load_cassette_interactions(cassette_dir)
    return infer_openapi_spec(interactions, min_samples=min_samples)
