from __future__ import annotations

import json

import httpx
import pytest
import yaml
from typer.testing import CliRunner

from driftgate.cli import app
from driftgate.datasets import (
    append_unique,
    body_operations,
    check_dataset,
    examples_from_cassettes,
    fix_dataset,
    generate_from_service,
    render_template,
    request_bodies_from_spec,
    sample_from_schema,
)

SPEC = {
    "openapi": "3.1.0",
    "paths": {
        "/chat": {
            "post": {
                "requestBody": {
                    "content": {
                        "application/json": {"schema": {"$ref": "#/components/schemas/Chat"}}
                    }
                }
            }
        },
        "/orders": {
            "post": {
                "requestBody": {
                    "content": {
                        "application/json": {
                            "example": {"sku": "A1", "qty": 2},
                            "schema": {"type": "object", "required": ["sku"],
                                       "properties": {"sku": {"type": "string"}}},
                        }
                    }
                }
            }
        },
        "/health": {"get": {}},
    },
    "components": {"schemas": {"Chat": {
        "type": "object",
        "required": ["prompt", "temperature", "mode", "tags"],
        "properties": {
            "prompt": {"type": "string"},
            "temperature": {"type": "number", "minimum": 0},
            "mode": {"enum": ["fast", "slow"]},
            "tags": {"type": "array", "minItems": 2, "items": {"type": "string", "format": "email"}},
            "optional": {"type": "string"},
        },
    }}},
}


def _write(path, *lines):
    path.write_text("\n".join(lines) + "\n")
    return path


def test_check_reports_each_problem_with_line_numbers(tmp_path):
    ds = _write(
        tmp_path / "d.jsonl",
        json.dumps({"prompt": "a", "response": "b"}),
        "{not json",
        json.dumps([1, 2]),
        json.dumps({"prompt": "only"}),
        json.dumps({"prompt": "", "response": "x"}),
        json.dumps({"prompt": "a", "response": "b"}),
        "",
    )
    report = check_dataset(ds)
    assert [(i.line, i.kind) for i in report.issues] == [
        (2, "invalid_json"), (3, "not_an_object"), (4, "missing_key"), (5, "empty_value"), (6, "duplicate"),
    ]
    assert len(report.errors) == 4 and len(report.examples) == 1 and not report.ok


def test_check_handles_missing_and_empty_files(tmp_path):
    assert check_dataset(tmp_path / "nope.jsonl").issues[0].kind == "unreadable"
    empty = tmp_path / "e.jsonl"
    empty.write_text("")
    assert not check_dataset(empty).ok and "no examples" in check_dataset(empty).summary()


def test_fix_keeps_valid_unique_lines_and_backs_up(tmp_path):
    good = json.dumps({"prompt": "a", "response": "b"})
    ds = _write(tmp_path / "d.jsonl", good, "{bad", good)
    report, backup = fix_dataset(ds)
    assert backup and backup.read_text().count("\n") == 3
    assert ds.read_text() == good + "\n" and check_dataset(ds).ok
    assert fix_dataset(ds)[1] is None  # clean file: untouched


def test_append_unique_skips_existing(tmp_path):
    ds = tmp_path / "sub" / "d.jsonl"
    ex = {"prompt": {"q": 1}, "response": {"a": 2}}
    assert append_unique(ds, [ex, ex]) == (1, 1)
    assert append_unique(ds, [ex, {"prompt": "x", "response": "y"}]) == (1, 1)


def test_sample_from_schema_resolves_refs_and_respects_constraints():
    body = sample_from_schema(SPEC["components"]["schemas"]["Chat"], SPEC)
    assert body == {
        "prompt": "example", "temperature": 0.0, "mode": "fast",
        "tags": ["user@example.com", "user@example.com"],
    }


def test_request_bodies_prefer_spec_examples_and_find_operations():
    assert body_operations(SPEC) == [("POST", "/chat"), ("POST", "/orders")]
    assert request_bodies_from_spec(SPEC, "POST", "/orders") == [
        {"sku": "A1", "qty": 2}, {"sku": "example"},
    ]
    with pytest.raises(ValueError, match="no JSON request body"):
        request_bodies_from_spec(SPEC, "GET", "/health")


def test_render_template_replaces_prompt_at_any_depth():
    tpl = {"messages": [{"role": "user", "content": "Q: {prompt}"}], "n": 1}
    assert render_template(tpl, "hi") == {"messages": [{"role": "user", "content": "Q: hi"}], "n": 1}


def _cassette(path):
    def item(method, p, status, req, resp):
        return {"fingerprint": "f", "canonical": "c",
                "request": {"method": method, "path": p, "body": req},
                "response": {"status": status, "body": resp}}
    path.write_text(yaml.safe_dump({"interactions": [
        item("POST", "/chat", 200, {"prompt": "hi"}, {"answer": "hello"}),
        item("POST", "/chat", 500, {"prompt": "boom"}, {"error": "x"}),
        item("GET", "/items", 200, None, {"items": []}),
        item("POST", "/other", 200, {"prompt": "z"}, {"answer": "z"}),
    ]}))
    return path


def test_examples_from_cassettes_keeps_only_successful_bodied_exchanges(tmp_path):
    cassette = _cassette(tmp_path / "c.yaml")
    assert len(examples_from_cassettes(cassette)) == 2
    assert examples_from_cassettes(cassette, endpoint="/chat") == [
        {"prompt": {"prompt": "hi"}, "response": {"answer": "hello"}}
    ]
    assert examples_from_cassettes(tmp_path, method="GET") == []  # directory form, no bodies


def _service(handler):
    return httpx.Client(base_url="http://svc", transport=httpx.MockTransport(handler))


def test_generate_from_service_keeps_2xx_and_explains_skips():
    def handler(request):
        body = json.loads(request.content)
        if body["prompt"] == "bad":
            return httpx.Response(422, json={"error": "no"})
        if body["prompt"] == "empty":
            return httpx.Response(200, json={})
        if body["prompt"] == "sse":
            return httpx.Response(200, text="data: hi\n\n", headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"answer": body["prompt"].upper()})

    bodies = [{"prompt": p} for p in ("ok", "bad", "empty", "sse")]
    with _service(handler) as client:
        result = generate_from_service(bodies, method="POST", endpoint="/chat", client=client)
    assert [e["response"] for e in result.examples] == [{"answer": "OK"}, "data: hi\n\n"]
    assert result.skipped == ["request 2: HTTP 422", "request 3: empty response body"]


def test_cli_check_and_fix(tmp_path):
    runner = CliRunner()
    good = json.dumps({"prompt": "a", "response": "b"})
    ds = _write(tmp_path / "d.jsonl", good, "{bad")
    result = runner.invoke(app, ["dataset", "check", str(ds)])
    assert result.exit_code == 1 and "line 2: invalid_json" in result.output and "--fix" in result.output
    result = runner.invoke(app, ["dataset", "check", str(ds), "--fix"])
    assert result.exit_code == 0 and "original saved" in result.output
    assert runner.invoke(app, ["dataset", "check", str(ds)]).exit_code == 0

    missing = runner.invoke(app, ["dataset", "check", str(tmp_path / "nope.jsonl")])
    assert missing.exit_code == 1 and "dataset generate" in missing.output


def _patch_service(monkeypatch, handler):
    real = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))


def test_cli_generate_requires_exactly_one_source(tmp_path):
    result = CliRunner().invoke(app, ["dataset", "generate", "--out", str(tmp_path / "o.jsonl")])
    assert result.exit_code == 2 and "exactly one" in result.output


def test_cli_generate_from_cassettes(tmp_path):
    out = tmp_path / "g.jsonl"
    cassette = _cassette(tmp_path / "c.yaml")
    result = CliRunner().invoke(app, ["dataset", "generate", "--from-cassettes", str(cassette),
                                      "--endpoint", "/chat", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert check_dataset(out).ok and len(check_dataset(out).examples) == 1


def test_cli_generate_from_prompts_with_template(tmp_path, monkeypatch):
    seen = []
    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"reply": "ok"})
    _patch_service(monkeypatch, handler)
    prompts = tmp_path / "p.txt"
    prompts.write_text("first\n\nsecond\n")
    out = tmp_path / "g.jsonl"
    result = CliRunner().invoke(app, [
        "dataset", "generate", "--prompts-file", str(prompts), "--endpoint", "/v1/chat",
        "--request-template", '{"messages": [{"role": "user", "content": "{prompt}"}]}',
        "--out", str(out), "--base-url", "http://svc"])
    assert result.exit_code == 0, result.output
    assert [b["messages"][0]["content"] for b in seen] == ["first", "second"]
    assert len(check_dataset(out).examples) == 2


def test_cli_generate_from_spec_picks_the_only_operation_or_asks(tmp_path, monkeypatch):
    _patch_service(monkeypatch, lambda r: httpx.Response(200, json={"echo": json.loads(r.content)}))
    spec = tmp_path / "s.yaml"
    spec.write_text(yaml.safe_dump(SPEC))
    out = tmp_path / "g.jsonl"
    ambiguous = CliRunner().invoke(app, ["dataset", "generate", "--spec", str(spec), "--out", str(out)])
    assert ambiguous.exit_code == 2 and "POST /chat, POST /orders" in ambiguous.output

    result = CliRunner().invoke(app, ["dataset", "generate", "--spec", str(spec), "--endpoint", "/orders",
                                      "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert [e["prompt"] for e in check_dataset(out).examples] == [{"sku": "A1", "qty": 2}, {"sku": "example"}]


def test_cli_generate_fails_when_service_unreachable(tmp_path, monkeypatch):
    def handler(request):
        raise httpx.ConnectError("refused")
    _patch_service(monkeypatch, handler)
    prompts = tmp_path / "p.txt"
    prompts.write_text("hi\n")
    out = tmp_path / "g.jsonl"
    result = CliRunner().invoke(app, ["dataset", "generate", "--prompts-file", str(prompts),
                                      "--endpoint", "/chat", "--out", str(out)])
    assert result.exit_code == 1 and "No usable examples" in result.output and not out.exists()
