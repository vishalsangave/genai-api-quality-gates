"""Offline tests for the judge factory (no network, no real SDKs)."""

from __future__ import annotations

import sys
import types

import pytest

from driftgate.config import JudgeConfig
from driftgate.metamorphic.evaluator import (
    DeterministicRubricJudge,
    LLMRubricJudge,
)
from driftgate.metamorphic.factory import build_judge


def test_deterministic_provider_builds_deterministic_judge() -> None:
    judge = build_judge(JudgeConfig(provider="deterministic"))
    assert isinstance(judge, DeterministicRubricJudge)


def test_unknown_provider_raises() -> None:
    config = JudgeConfig(provider="deterministic")
    object.__setattr__(config, "provider", "gemini")
    with pytest.raises(ValueError, match="gemini"):
        build_judge(config)


def _install_fake_anthropic(monkeypatch, calls: list[dict]) -> None:
    class _Text:
        def __init__(self, text: str) -> None:
            self.text = text

    class _Message:
        def __init__(self, text: str) -> None:
            self.content = [_Text(text)]

    class _Messages:
        def create(self, **kwargs: object) -> _Message:
            calls.append(kwargs)
            return _Message("reasoning\nVERDICT: PASS")

    class _Anthropic:
        def __init__(self, api_key: str | None = None) -> None:
            assert api_key == "test-key"
            self.messages = _Messages()

    module = types.ModuleType("anthropic")
    module.Anthropic = _Anthropic  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", module)


def test_anthropic_provider_wires_lazy_transport(monkeypatch) -> None:
    calls: list[dict] = []
    _install_fake_anthropic(monkeypatch, calls)
    monkeypatch.setenv("DRIFT_GATE_JUDGE_API_KEY", "test-key")

    judge = build_judge(JudgeConfig(provider="anthropic", api_key_env="DRIFT_GATE_JUDGE_API_KEY"))
    assert isinstance(judge, LLMRubricJudge)
    verdict = judge.score("instruction", "candidate output", "rubric")
    assert verdict.passed
    assert calls and calls[0]["model"] == "claude-haiku-4-5-20251001"
    assert calls[0]["temperature"] == 0
    assert "rubric" in str(calls[0]["messages"][0]["content"])


def test_openai_provider_wires_lazy_transport(monkeypatch) -> None:
    calls: list[dict] = []

    class _Message:
        content = "analysis\nVERDICT: PASS"

    class _Choice:
        message = _Message()

    class _Completion:
        choices = [_Choice()]

    class _Completions:
        def create(self, **kwargs: object) -> _Completion:
            calls.append(kwargs)
            return _Completion()

    class _Chat:
        completions = _Completions()

    class _OpenAI:
        def __init__(self, api_key: str | None = None) -> None:
            assert api_key == "test-key"
            self.chat = _Chat()

    module = types.ModuleType("openai")
    module.OpenAI = _OpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)
    monkeypatch.setenv("DRIFT_GATE_JUDGE_API_KEY", "test-key")

    judge = build_judge(JudgeConfig(provider="openai", api_key_env="DRIFT_GATE_JUDGE_API_KEY"))
    assert isinstance(judge, LLMRubricJudge)
    assert judge.score("instruction", "answer", "rubric").passed
    assert judge.score("instruction", "answer", "rubric").passed  # idempotent wiring
    assert calls and calls[0]["model"] == "gpt-4o-mini"
    assert calls[0]["temperature"] == 0


def test_missing_api_key_raises_before_any_call(monkeypatch) -> None:
    _install_fake_anthropic(monkeypatch, [])
    monkeypatch.delenv("DRIFT_GATE_JUDGE_API_KEY", raising=False)
    config = JudgeConfig(provider="anthropic", api_key_env="DRIFT_GATE_JUDGE_API_KEY")
    with pytest.raises(RuntimeError, match="DRIFT_GATE_JUDGE_API_KEY"):
        build_judge(config)
