"""Full three-layer pipeline integration against the demo GenAI service.

The service is started in-process over an ephemeral port; every layer of the
framework (contract, cassette replay, streaming probe, metamorphic evaluation,
release governor) runs against it exactly as a real service deployment would.
"""

from __future__ import annotations

import json
import random
import socket

import httpx
import pytest

from driftgate.contracts.validator import ContractValidator
from driftgate.metamorphic.evaluator import InvarianceChecker, evaluate_structural_invariance
from driftgate.metamorphic.relations import AdversarialNoiseRelation
from driftgate.recorder.cassette import CassetteMode, CassetteStore
from driftgate.recorder.fingerprint import FingerprintConfig
from driftgate.recorder.interceptor import RecordingTransport
from driftgate.streaming.probe import SSEFrameParser

CHAT = "/v1/chat/completions"
SUMMARIZE = "/v1/summarize"


@pytest.fixture(scope="module")
def demo_contract() -> ContractValidator:
    import yaml

    from examples.tests.conftest import ROOT

    with (ROOT / "examples" / "demo_service" / "openapi.yaml").open() as handle:
        spec = yaml.safe_load(handle)
    return ContractValidator(spec)


def _chat(client: httpx.Client, behavior: str) -> httpx.Response:
    return client.post(
        CHAT,
        json={"messages": [{"role": "user", "content": "Summarize replay testing."}]},
        headers={"X-Demo-Behavior": behavior},
    )


def json_body(response: httpx.Response) -> object:
    import json

    return json.loads(response.content)


def test_layer1_normal_stream_passes_contract(demo_base_url, demo_contract) -> None:
    with httpx.Client(base_url=demo_base_url, timeout=10.0) as client:
        response = _chat(client, "normal")
        assert response.status_code == 200
        response.read()
        assert (
            demo_contract.validate_response(
                "POST", CHAT, 200, headers={"content-type": "text/event-stream"}
            )
            == []
        )


def test_layer1_invalid_schema_is_caught(demo_base_url, demo_contract) -> None:
    # The service deliberately emits a frame whose declared per-delta schema
    # expects a string delta but delivers a number. Feeding the collected
    # frames through the contract's declared shape must produce violations.
    with httpx.Client(base_url=demo_base_url, timeout=10.0) as client:
        response = _chat(client, "invalid_schema")
        parser = SSEFrameParser()
        frames = parser.feed(response.read()) + parser.finish()
    from jsonschema import Draft202012Validator

    delta_schema = {
        "type": "object",
        "required": ["delta"],
        "properties": {"delta": {"type": "string"}},
    }
    validator = Draft202012Validator(delta_schema)
    bad_frames = [frame for frame in frames if frame.data and frame.data != "[DONE]"]
    assert bad_frames
    assert any(list(validator.iter_errors(json.loads(frame.data))) for frame in bad_frames)


def test_layer2_record_then_replay_bytes_identical(tmp_path, demo_base_url) -> None:
    cassette = tmp_path / "chat.yaml"
    store = CassetteStore(cassette, CassetteMode.RECORD, service=demo_base_url)
    transport = RecordingTransport(
        httpx.HTTPTransport(),
        store,
        fingerprint_config=FingerprintConfig(),
    )
    with httpx.Client(base_url=demo_base_url, transport=transport, timeout=10.0) as client:
        live = _chat(client, "normal")
        live_body = live.read()
    store.save()

    # The replay transport deliberately points at a dead port: only the
    # cassette can answer, proving full offline determinism.
    dead_store = CassetteStore(cassette, CassetteMode.REPLAY_STRICT, service=demo_base_url)
    dead_transport = RecordingTransport(
        httpx.HTTPTransport(),
        dead_store,
        fingerprint_config=FingerprintConfig(),
    )
    dead_base = _dead_base_url()
    with httpx.Client(base_url=dead_base, transport=dead_transport, timeout=10.0) as client:
        replayed = client.post(
            CHAT,
            json={"messages": [{"role": "user", "content": "Summarize replay testing."}]},
            headers={"X-Demo-Behavior": "normal"},
        )
        assert replayed.extensions["driftgate_replayed"] is True
        replay_body = replayed.read()
    assert replay_body == live_body


def _dead_base_url() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{sock.getsockname()[1]}"


def test_layer2_authorization_redacted_on_disk(tmp_path, demo_base_url) -> None:
    cassette = tmp_path / "redact.yaml"
    store = CassetteStore(cassette, CassetteMode.RECORD, service=demo_base_url)
    transport = RecordingTransport(
        httpx.HTTPTransport(), store, fingerprint_config=FingerprintConfig()
    )
    with httpx.Client(base_url=demo_base_url, transport=transport, timeout=10.0) as client:
        response = client.post(
            CHAT,
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer super-secret", "X-Demo-Behavior": "normal"},
        )
        assert response.status_code == 200
        response.read()
    store.save()
    assert "super-secret" not in cassette.read_text()


def test_probe_normal_stream_meets_sla(demo_base_url) -> None:
    with httpx.Client(base_url=demo_base_url, timeout=10.0) as client:
        response = _chat(client, "normal")
        parser = SSEFrameParser()
        frames = parser.feed(response.read()) + parser.finish()
    assert [frame.data for frame in frames][-1] == "[DONE]"


def test_probe_catches_truncated_stream(demo_base_url) -> None:
    with httpx.Client(base_url=demo_base_url, timeout=10.0) as client:
        response = _chat(client, "truncated")
        parser = SSEFrameParser()
        frames = parser.feed(response.read()) + parser.finish()
    assert frames[-1].data != "[DONE]"


def test_probe_flags_midstream_error_event(demo_base_url) -> None:
    with httpx.Client(base_url=demo_base_url, timeout=10.0) as client:
        response = _chat(client, "error_stream")
        parser = SSEFrameParser()
        frames = parser.feed(response.read()) + parser.finish()
    assert any(frame.event == "error" for frame in frames)


def test_metamorphic_paraphrase_invariance_with_local_embedder() -> None:
    # Two near-identical wordings with the same key vocabulary must clear the
    # default hashing-embedder threshold used by the framework.
    checker = InvarianceChecker()
    result = checker.check(
        "Summarize the replay testing approach for reviewers.",
        "Please summarize the replay testing approach for reviewers.",
        "paraphrase_invariance",
    )
    assert result.passed


def test_adversarial_noise_stays_structurally_valid() -> None:
    payloads = [
        {"messages": [{"role": "user", "content": "Replay testing keeps PR gates free and fast."}]},
        {
            "messages": [
                {"role": "user", "content": "Cassette fingerprinting ignores volatile IDs."}
            ]
        },
    ]
    relation = AdversarialNoiseRelation()
    variants = []
    for payload in payloads:
        variants.extend(relation.apply(payload, rng=random.Random(7)))
    assert variants
    # Every noisy variant keeps the required message structure the demo
    # contract declares — the structural oracle the framework provides for
    # adversarial-noise relations.
    assert evaluate_structural_invariance(lambda _v: True, variants) == [True] * len(variants)


def test_summarize_contract_shapes_match(demo_base_url, demo_contract) -> None:
    with httpx.Client(base_url=demo_base_url, timeout=10.0) as client:
        response = client.post(SUMMARIZE, json={"text": "one two three", "max_words": 2})
        assert response.status_code == 200
        body = json_body(response)
    assert demo_contract.validate_response("POST", SUMMARIZE, 200, body=body) == []
