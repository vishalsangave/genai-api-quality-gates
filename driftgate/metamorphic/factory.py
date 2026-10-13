"""Judge construction from ``JudgeConfig`` — the single wiring point.

``build_judge`` maps the configured provider to a ``Judge`` implementation.
SDK imports happen lazily inside the transports so importing this module (or
anything on the PR-gate path) never pulls in ``anthropic`` or ``openai``.
See docs/ARCHITECTURE.md "Layer 3" for the deterministic-vs-LLM tiering rule.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from driftgate.metamorphic.evaluator import (
    DeterministicRubricJudge,
    LLMRubricJudge,
)

if TYPE_CHECKING:
    from driftgate.config import JudgeConfig
    from driftgate.metamorphic.evaluator import Judge


def build_judge(config: JudgeConfig) -> Judge:
    if config.provider == "deterministic":
        return DeterministicRubricJudge()
    if config.provider == "anthropic":
        return LLMRubricJudge(_anthropic_transport(config))
    if config.provider == "openai":
        return LLMRubricJudge(_openai_transport(config))
    if config.provider == "google":
        return LLMRubricJudge(_google_transport(config))
    raise ValueError(f"unknown judge provider: {config.provider!r}")


def _resolve_api_key(config: JudgeConfig) -> str:
    key = os.environ.get(config.api_key_env, "")
    if not key:
        raise RuntimeError(
            f"judge provider {config.provider!r} requires environment variable "
            f"{config.api_key_env!r} to be set"
        )
    return key


def _anthropic_transport(config: JudgeConfig):
    import anthropic

    client = anthropic.Anthropic(api_key=_resolve_api_key(config))
    model = config.anthropic.model

    def transport(prompt: str) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=512,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    return transport


def _google_transport(config: JudgeConfig):
    import httpx

    client = httpx.Client(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        timeout=config.google.timeout,
        headers={"x-goog-api-key": _resolve_api_key(config)},
    )
    model = config.google.model

    def transport(prompt: str) -> str:
        response = client.post(
            f"/models/{model}:generateContent",
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0},
            },
        )
        response.raise_for_status()
        candidates = response.json().get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return " ".join(part.get("text", "") for part in parts)

    return transport


def _openai_transport(config: JudgeConfig):
    import openai

    client = openai.OpenAI(api_key=_resolve_api_key(config))
    model = config.openai.model

    def transport(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    return transport
