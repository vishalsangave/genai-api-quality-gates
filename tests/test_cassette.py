from __future__ import annotations

import pytest
import yaml

from driftgate.recorder.cassette import (
    REDACTED,
    CassetteInteraction,
    CassetteMissError,
    CassetteMode,
    CassetteStore,
    RecordedRequest,
    RecordedResponse,
)
from driftgate.recorder.fingerprint import fingerprint_request


def _interaction() -> CassetteInteraction:
    fp = fingerprint_request(
        "POST", "/v1/chat", {"Authorization": "Bearer secret"}, {"prompt": "hi"}
    )
    return CassetteInteraction(
        fingerprint=fp.digest,
        canonical=fp.canonical,
        request=RecordedRequest(
            method="POST",
            path="/v1/chat",
            headers={"Authorization": "Bearer secret"},
            body={"prompt": "hi"},
        ),
        response=RecordedResponse(
            status=200,
            headers={"Authorization": "Bearer response-secret", "content-type": "application/json"},
            body={"answer": "hello"},
        ),
    )


def test_cassette_round_trip_redacts_secrets(tmp_path) -> None:
    path = tmp_path / "chat.yaml"
    store = CassetteStore(path, CassetteMode.RECORD)
    interaction = _interaction()
    store.record(interaction)
    store.save()

    persisted = yaml.safe_load(path.read_text())
    assert persisted["interactions"][0]["request"]["headers"]["authorization"] == REDACTED
    assert persisted["interactions"][0]["response"]["headers"]["authorization"] == REDACTED

    replay = CassetteStore(path, CassetteMode.REPLAY_STRICT)
    fp = fingerprint_request(
        "POST", "/v1/chat", {"Authorization": "Bearer another"}, {"prompt": "hi"}
    )
    found = replay.lookup(fp)
    assert found is not None
    assert found.response.body == {"answer": "hello"}


def test_duplicate_fingerprints_are_replayed_fifo(tmp_path) -> None:
    path = tmp_path / "fifo.yaml"
    store = CassetteStore(path, CassetteMode.RECORD)
    first = _interaction()
    second = _interaction()
    first.response.body = {"answer": "first"}
    second.response.body = {"answer": "second"}
    store.record(first)
    store.record(second)
    store.save()

    replay = CassetteStore(path, CassetteMode.REPLAY_STRICT)
    fp = fingerprint_request("POST", "/v1/chat", {}, {"prompt": "hi"})
    assert replay.lookup(fp).response.body == {"answer": "first"}  # type: ignore[union-attr]
    assert replay.lookup(fp).response.body == {"answer": "second"}  # type: ignore[union-attr]


def test_strict_mode_raises_on_miss(tmp_path) -> None:
    store = CassetteStore(tmp_path / "empty.yaml", CassetteMode.REPLAY_STRICT)
    fp = fingerprint_request("GET", "/not-recorded", {}, None)

    with pytest.raises(CassetteMissError, match="No cassette match"):
        store.lookup(fp)


def test_auto_record_maps_to_lenient_replay() -> None:
    assert CassetteMode.from_traffic_mode("auto_record") is CassetteMode.REPLAY_LENIENT
