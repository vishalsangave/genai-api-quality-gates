from __future__ import annotations

import httpx
import pytest

from driftgate import pytest_plugin as plugin
from driftgate.config import DriftGateConfig, JudgeConfig
from driftgate.metamorphic.evaluator import (
    EmbeddingSimilarityJudge,
    SemanticSimilarityJudge,
)


def _evaluator(**judge):
    fixture = plugin.driftgate_evaluator
    fn = getattr(fixture, "__wrapped__", None) or fixture._get_wrapped_function()
    return fn(DriftGateConfig(judge=JudgeConfig(**judge)))


def test_default_provider_is_an_offline_judge_that_can_fail():
    judge = _evaluator()
    assert isinstance(judge, EmbeddingSimilarityJudge)
    assert judge.compare("p", "The result is 4.", "The result is 4.").passed
    assert not judge.compare("p", "The result is 4.", "completely unrelated pwned").passed
    assert not judge.compare("p", "The result is 4.", "").passed


def test_similarity_threshold_is_configurable():
    strict = _evaluator(similarity_threshold=1.0)
    assert not strict.compare("p", "The result is 4.", "The result is four.").passed
    lax = _evaluator(similarity_threshold=0.0)
    assert lax.compare("p", "The result is 4.", "The result is four.").passed


def test_missing_api_key_fails_closed_by_default(monkeypatch):
    monkeypatch.delenv("DRIFT_GATE_JUDGE_API_KEY", raising=False)
    for provider in ("anthropic", "openai", "google"):
        with pytest.raises(pytest.fail.Exception, match="requires DRIFT_GATE_JUDGE_API_KEY"):
            _evaluator(provider=provider)


def test_missing_api_key_skips_only_when_opted_in(monkeypatch):
    monkeypatch.delenv("DRIFT_GATE_JUDGE_API_KEY", raising=False)
    with pytest.raises(pytest.skip.Exception):
        _evaluator(provider="google", on_error="skip")


def test_unreachable_judge_fails_instead_of_skipping_or_passing():
    config = JudgeConfig(provider="ollama")
    client = httpx.Client(
        base_url="http://judge.invalid",
        transport=httpx.MockTransport(lambda r: httpx.Response(401, json={"error": "bad key"})),
    )
    with pytest.raises(pytest.fail.Exception, match="Could not get a verdict from Ollama"):
        plugin._post_json(client, config, "Ollama", "/api/generate", {})

    skipping = JudgeConfig(provider="ollama", on_error="skip")
    with pytest.raises(pytest.skip.Exception):
        plugin._post_json(client, skipping, "Ollama", "/api/generate", {})


def test_llm_judge_uses_curated_template_and_parses_verdict(monkeypatch):
    monkeypatch.setenv("DRIFT_GATE_JUDGE_API_KEY", "k")
    prompts: list[str] = []
    judge = SemanticSimilarityJudge(lambda p: (prompts.append(p), "ok\nVERDICT: PASS")[1])
    verdict = judge.compare("what is 2+2", "It is 4 {x}", "4 {y}")
    assert verdict.passed
    assert "Golden Response:\nIt is 4 {x}" in prompts[0] and "New Response:\n4 {y}" in prompts[0]


def test_llm_judge_retries_once_then_fails_closed_on_garbage():
    calls = []
    judge = SemanticSimilarityJudge(lambda p: (calls.append(p), "no verdict here")[1])
    verdict = judge.compare("p", "g", "a")
    assert not verdict.passed and len(calls) == 2


def test_unknown_provider_is_a_usage_error():
    config = DriftGateConfig(judge=JudgeConfig())
    object.__setattr__(config.judge, "provider", "mystery")
    fixture = plugin.driftgate_evaluator
    fn = getattr(fixture, "__wrapped__", None) or fixture._get_wrapped_function()
    with pytest.raises(pytest.UsageError):
        fn(config)
