from __future__ import annotations

from driftgate.recorder.fingerprint import FingerprintConfig, fingerprint_request, normalize_string


def test_volatile_values_and_whitespace_do_not_change_fingerprint() -> None:
    first = fingerprint_request(
        "post",
        "/v1/chat?debug=1",
        {"Authorization": "Bearer first", "X-Request-ID": "abc", "X-Mode": "safe"},
        {
            "user_id": "user-one",
            "request_id": "request-one",
            "prompt": "  summarize\n this document ",
            "messages": [{"role": "user", "content": "Hello   world"}],
        },
    )
    second = fingerprint_request(
        "POST",
        "/v1/chat?debug=1",
        {"Authorization": "Bearer second", "X-Request-ID": "xyz", "X-Mode": "safe"},
        {
            "user_id": "user-two",
            "request_id": "request-two",
            "prompt": "summarize this\t document",
            "messages": [{"role": "user", "content": "Hello world"}],
        },
    )

    assert first.digest == second.digest
    assert "Authorization" not in first.canonical
    assert "user-one" not in first.canonical


def test_semantic_body_change_changes_fingerprint() -> None:
    first = fingerprint_request("POST", "/v1/chat", {}, {"prompt": "summarize the report"})
    second = fingerprint_request("POST", "/v1/chat", {}, {"prompt": "delete the report"})

    assert first.digest != second.digest


def test_code_like_strings_normalize_on_ast_not_formatting() -> None:
    config = FingerprintConfig()
    compact = "def add(a, b):\n    return a + b\n"
    spaced = "def add( a,b ):\n    return a+b\n"

    assert normalize_string(compact, config) == normalize_string(spaced, config)


def test_invalid_code_falls_back_to_collapsed_whitespace() -> None:
    value = "def broken(:\n  ???"
    assert normalize_string(value, FingerprintConfig()) == "def broken(: ???"


def test_query_order_and_numeric_shape_are_stable() -> None:
    first = fingerprint_request("GET", "/v1/orders?b=2&a=1", {}, {"amount": 150})
    second = fingerprint_request("GET", "/v1/orders?a=1&b=2", {}, {"amount": 150.0})

    assert first.digest == second.digest
