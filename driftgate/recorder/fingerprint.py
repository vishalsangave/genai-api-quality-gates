"""AST-aware request fingerprinting for the cassette replay engine.

Unlike a raw-byte HTTP mocker (VCR-style), a request is matched on a
*normalized* canonical form that:

- drops volatile fields (user/session/request/correlation/trace IDs,
  timestamps, nonces) from JSON bodies,
- drops volatile headers (``Authorization``, cookies, API keys, dates, and
  any header that looks like a request/correlation/trace ID),
- collapses whitespace permutations in free-text payloads, and
- for payloads that look like source code, normalizes on the parsed AST
  (via ``ast.dump(..., include_attributes=False)``) so formatting and
  position differences never change the fingerprint, while a genuine
  semantic change to the code always does.

Fingerprinting must never raise: any failure anywhere in the pipeline (a
string that merely resembles code but isn't valid Python, an exotic value
type, ...) falls back to the safest available normalization rather than
propagating an exception into the caller's request path.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from pydantic import BaseModel, Field

__all__ = [
    "Fingerprint",
    "FingerprintConfig",
    "fingerprint_request",
    "canonicalize_value",
    "normalize_string",
]

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

_DEFAULT_IGNORE_FIELDS = frozenset(
    {
        "user_id",
        "userid",
        "session_id",
        "sessionid",
        "request_id",
        "requestid",
        "correlation_id",
        "trace_id",
        "traceid",
        "timestamp",
        "created_at",
        "createdat",
        "createdtimestamp",
        "updated_at",
        "updatedat",
        "nonce",
        "uuid",
    }
)

_DEFAULT_IGNORE_HEADERS = frozenset(
    {
        "authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "api-key",
        "user-agent",
        "host",
        "content-length",
        "date",
        "expires",
        "last-modified",
        "accept-encoding",
        "connection",
    }
)

_DEFAULT_IGNORE_HEADER_PATTERNS = (
    r"^x-(request|correlation|trace)[-_]id$",
    r".*-date$",
    r"^x-timestamp$",
)


class FingerprintConfig(BaseModel):
    """Tunables for :func:`fingerprint_request`. Every list is overridable."""

    ignore_fields: frozenset[str] = Field(default_factory=lambda: _DEFAULT_IGNORE_FIELDS)
    ignore_headers: frozenset[str] = Field(default_factory=lambda: _DEFAULT_IGNORE_HEADERS)
    ignore_header_patterns: tuple[str, ...] = _DEFAULT_IGNORE_HEADER_PATTERNS
    ignore_query_params: frozenset[str] = frozenset()
    sort_list_items: bool = False
    enable_ast_normalization: bool = True
    min_code_len: int = 24

    model_config = {"frozen": True}


# --------------------------------------------------------------------------
# Result type
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Fingerprint:
    """The result of normalizing and hashing one HTTP request."""

    digest: str
    """sha256 hex digest of the canonical document."""

    canonical: str
    """The canonical JSON string that was hashed. Kept for debugging and for
    the secondary equality check cassette lookups perform on a digest hit."""

    method: str
    path: str


# --------------------------------------------------------------------------
# String / code normalization
# --------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")
_CODE_KEYWORD_RE = re.compile(
    r"\b(def|class|import|from|return|function|const|let|var|if|elif|else|for|while|try|except)\b"
)


def _looks_like_code(s: str, min_len: int) -> bool:
    """Cheap heuristic gate before attempting an ``ast.parse`` call.

    Requires a newline or semicolon (a real signal of statement structure)
    plus a keyword, an assignment, or balanced parens — narrow enough to
    avoid mis-firing on ordinary prose, permissive enough to catch real
    code snippets in a prompt payload.
    """
    if len(s) < min_len:
        return False
    if "\n" not in s and ";" not in s:
        return False
    has_keyword = bool(_CODE_KEYWORD_RE.search(s))
    has_assignment = "=" in s
    has_balanced_parens = "(" in s and s.count("(") == s.count(")")
    return has_keyword or has_assignment or has_balanced_parens


def normalize_string(s: str, cfg: FingerprintConfig) -> str:
    """Normalize one string value for fingerprinting.

    Tries AST normalization on the *original* string first (Python syntax
    is whitespace-sensitive — collapsing whitespace before parsing would
    destroy indentation-based block structure and make every multi-line
    snippet fail to parse). Only on ``SyntaxError`` — or any other failure,
    since fingerprinting must never raise — does it fall back to the
    whitespace-collapsed string.
    """
    if cfg.enable_ast_normalization and _looks_like_code(s, cfg.min_code_len):
        try:
            tree = ast.parse(s, mode="exec")
            return ast.dump(tree, annotate_fields=True, include_attributes=False)
        except SyntaxError:
            pass
        except Exception:  # noqa: BLE001 - fingerprinting must never throw
            pass
    return _WHITESPACE_RE.sub(" ", s).strip()


# --------------------------------------------------------------------------
# Structural canonicalization
# --------------------------------------------------------------------------


def _normalize_number(value: int | float) -> str:
    """Unify int/float representations so ``150`` and ``150.0`` fingerprint
    identically."""
    return repr(round(float(value), 9))


def _sort_key(value: Any) -> tuple[str, str]:
    return (type(value).__name__, repr(value))


def canonicalize_value(value: Any, cfg: FingerprintConfig) -> Any:
    """Recursively canonicalize one JSON-like value for fingerprinting."""
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, val in value.items():
            if isinstance(key, str) and key.lower() in cfg.ignore_fields:
                continue
            result[str(key)] = canonicalize_value(val, cfg)
        return result
    if isinstance(value, (list, tuple)):
        items = [canonicalize_value(v, cfg) for v in value]
        if cfg.sort_list_items:
            try:
                items = sorted(items, key=_sort_key)
            except TypeError:
                pass
        return items
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return _normalize_number(value)
    if isinstance(value, str):
        return normalize_string(value, cfg)
    if value is None:
        return None
    return str(value)


# --------------------------------------------------------------------------
# Request decomposition
# --------------------------------------------------------------------------


def _split_path_query(path: str) -> tuple[str, dict[str, list[str]]]:
    parsed = urlsplit(path)
    query: dict[str, list[str]] = {}
    for key, val in parse_qsl(parsed.query, keep_blank_values=True):
        query.setdefault(key, []).append(val)
    return parsed.path, query


def _canonicalize_query(
    query: Mapping[str, list[str]], cfg: FingerprintConfig
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for key, values in query.items():
        if key in cfg.ignore_query_params:
            continue
        out[key] = sorted(values)
    return out


def _filter_headers(headers: Mapping[str, str], cfg: FingerprintConfig) -> dict[str, str]:
    patterns = [re.compile(p) for p in cfg.ignore_header_patterns]
    out: dict[str, str] = {}
    for key, val in headers.items():
        lowered = key.lower()
        if lowered in cfg.ignore_headers:
            continue
        if any(p.match(lowered) for p in patterns):
            continue
        out[lowered] = val
    return out


def _parse_body(body: bytes | str | Mapping[str, Any] | list[Any] | None) -> Any:
    """Best-effort parse of a request/response body into a JSON-like value.

    Non-JSON text (including source code prompt payloads) is returned as a
    plain string so it flows through :func:`normalize_string`, which is
    exactly where AST normalization applies. Non-UTF-8 bytes are hashed
    rather than included verbatim, since binary content can't be
    meaningfully canonicalized as text.
    """
    if body is None:
        return None
    if isinstance(body, (Mapping, list)):
        return body
    if isinstance(body, bytes):
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            return {"__binary_sha256__": hashlib.sha256(body).hexdigest()}
    elif isinstance(body, str):
        text = body
    else:
        text = str(body)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def fingerprint_request(
    method: str,
    path: str,
    headers: Mapping[str, str],
    body: bytes | str | Mapping[str, Any] | list[Any] | None,
    *,
    cfg: FingerprintConfig | None = None,
) -> Fingerprint:
    """Compute the AST-aware fingerprint of one HTTP request.

    ``path`` may include a query string; it is split and canonicalized
    separately from the path segment itself.
    """
    cfg = cfg or FingerprintConfig()
    path_only, query = _split_path_query(path)
    canonical_doc = {
        "method": method.upper(),
        "path": path_only,
        "query": _canonicalize_query(query, cfg),
        "headers": _filter_headers(headers, cfg),
        "body": canonicalize_value(_parse_body(body), cfg),
    }
    canonical_json = json.dumps(
        canonical_doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return Fingerprint(
        digest=digest, canonical=canonical_json, method=method.upper(), path=path_only
    )
