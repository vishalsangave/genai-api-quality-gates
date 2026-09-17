"""Versioned YAML cassette storage with redaction and strict/lenient replay."""

from __future__ import annotations

import os
from collections import defaultdict, deque
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from driftgate.recorder.fingerprint import Fingerprint
from driftgate.streaming.probe import SSEFrame

REDACTED = "[REDACTED]"
REDACT_HEADERS = frozenset({"authorization", "cookie", "set-cookie", "x-api-key", "api-key"})


class CassetteMissError(AssertionError):
    """Raised in replay-strict mode when no canonical request match exists."""


class CassetteFormatError(ValueError):
    """Raised for an incompatible or malformed cassette document."""


class CassetteMode(StrEnum):
    RECORD = "record"
    REPLAY_STRICT = "replay_strict"
    REPLAY_LENIENT = "replay_lenient"
    LIVE = "live"

    @classmethod
    def from_traffic_mode(cls, mode: str) -> CassetteMode:
        mapping = {
            "record": cls.RECORD,
            "replay_strict": cls.REPLAY_STRICT,
            "replay_lenient": cls.REPLAY_LENIENT,
            "live": cls.LIVE,
            "auto_record": cls.REPLAY_LENIENT,
        }
        try:
            return mapping[mode]
        except KeyError as error:
            raise ValueError(f"Unknown traffic mode {mode!r}; valid: {sorted(mapping)}") from error


class RecordedRequest(BaseModel):
    method: str
    path: str
    query: dict[str, list[str]] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    body: Any = None


class RecordedSSEFrame(BaseModel):
    event: str | None = None
    id: str | None = None
    data: str

    @classmethod
    def from_frame(cls, frame: SSEFrame) -> RecordedSSEFrame:
        return cls(event=frame.event, id=frame.id, data=frame.data)

    def to_frame(self) -> SSEFrame:
        return SSEFrame(event=self.event, id=self.id, data=self.data)


class RecordedResponse(BaseModel):
    status: int
    headers: dict[str, str] = Field(default_factory=dict)
    body: Any = None
    is_stream: bool = False
    sse_events: list[RecordedSSEFrame] = Field(default_factory=list)


class CassetteInteraction(BaseModel):
    fingerprint: str
    canonical: str
    request: RecordedRequest
    response: RecordedResponse


class Cassette(BaseModel):
    version: Literal[1] = 1
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    service: str | None = None
    interactions: list[CassetteInteraction] = Field(default_factory=list)


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    """Return lower-cased headers with secrets redacted before persistence."""
    return {
        key.lower(): REDACTED if key.lower() in REDACT_HEADERS else value
        for key, value in headers.items()
    }


class CassetteStore:
    """One cassette file with an in-memory FIFO replay index.

    FIFO order is significant: two semantically identical prompt requests may
    return different responses in a recorded dialogue; each replay consumes
    the next recorded interaction instead of always returning the first.
    """

    def __init__(
        self,
        path: str | Path,
        mode: CassetteMode,
        *,
        record_on_miss: bool = False,
        service: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.mode = mode
        self.record_on_miss = record_on_miss
        self.service = service
        self.cassette = Cassette(service=service)
        self._index: dict[str, deque[CassetteInteraction]] = defaultdict(deque)
        self.misses: list[Fingerprint] = []
        self.load()

    def load(self) -> None:
        self._index.clear()
        if not self.path.exists():
            self.cassette = Cassette(service=self.service)
            return
        raw = yaml.safe_load(self.path.read_text())
        if raw is None:
            self.cassette = Cassette(service=self.service)
            return
        try:
            cassette = Cassette.model_validate(raw)
        except Exception as error:  # Pydantic gives detail, preserve context
            raise CassetteFormatError(f"Cannot load cassette {self.path}: {error}") from error
        if cassette.version != 1:
            raise CassetteFormatError(
                f"Unsupported cassette version {cassette.version} in {self.path}"
            )
        self.cassette = cassette
        for interaction in self.cassette.interactions:
            self._index[interaction.fingerprint].append(interaction)

    def lookup(self, fp: Fingerprint) -> CassetteInteraction | None:
        candidates = self._index.get(fp.digest)
        if candidates:
            # Do not consume an entry whose digest collides or whose broad
            # ignore lists happened to collapse a semantically different call.
            for _ in range(len(candidates)):
                candidate = candidates.popleft()
                if candidate.canonical == fp.canonical:
                    return candidate
                candidates.append(candidate)
        self.misses.append(fp)
        if self.mode is CassetteMode.REPLAY_STRICT:
            raise CassetteMissError(
                f"No cassette match for {fp.method} {fp.path}\nCanonical request: {fp.canonical}"
            )
        return None

    def record(self, interaction: CassetteInteraction) -> None:
        interaction.response.headers = redact_headers(interaction.response.headers)
        interaction.request.headers = redact_headers(interaction.request.headers)
        self.cassette.interactions.append(interaction)
        self._index[interaction.fingerprint].append(interaction)

    def save(self) -> None:
        if (
            self.mode not in (CassetteMode.RECORD, CassetteMode.REPLAY_LENIENT)
            and not self.record_on_miss
        ):
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        content = yaml.safe_dump(
            self.cassette.model_dump(mode="json"), sort_keys=False, allow_unicode=True
        )
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            tmp.write_text(content)
            os.replace(tmp, self.path)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    def iter_interactions(self) -> Iterable[CassetteInteraction]:
        return iter(self.cassette.interactions)
