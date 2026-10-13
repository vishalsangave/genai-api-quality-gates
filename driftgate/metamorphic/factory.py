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

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def build_judge(config: JudgeConfig) -> Judge:
    if config.provider == "deterministic":
        return DeterministicRubricJudge()
    if config.provider == "anthropic":
        return LLMRubricJudge(_anthropic_transport(config))
    if config.provider == "openai":
        return LLMRubricJudge(_openai_transport(config))
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
    model = config.model or DEFAULT_ANTHROPIC_MODEL

    def transport(prompt: str) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=512,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    return transport


def _openai_transport(config: JudgeConfig):
    import openai

    client = openai.OpenAI(api_key=_resolve_api_key(config))
    model = config.model or DEFAULT_OPENAI_MODEL

    def transport(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    return transport
