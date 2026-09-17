"""Deterministic, zero-network metamorphic input transformations."""

from __future__ import annotations

import copy
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


class MetamorphicRelation(Protocol):
    name: str

    def apply(self, payload: dict[str, Any], *, rng: random.Random) -> list[dict[str, Any]]:
        """Return deterministic transformed variants of a source request."""


_WORD_REWRITES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bplease\b", re.IGNORECASE), "kindly"),
    (re.compile(r"\bsummarize\b", re.IGNORECASE), "condense"),
    (re.compile(r"\bshow\b", re.IGNORECASE), "display"),
    (re.compile(r"\bhelp\b", re.IGNORECASE), "assist"),
    (re.compile(r"\buse\b", re.IGNORECASE), "employ"),
)


def _message_paths(payload: Mapping[str, Any]) -> list[tuple[int, str]]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return []
    result: list[tuple[int, str]] = []
    for index, message in enumerate(messages):
        if isinstance(message, Mapping) and isinstance(message.get("content"), str):
            result.append((index, message["content"]))
    return result


@dataclass(frozen=True)
class ParaphraseRelation:
    name: str = "paraphrase_invariance"

    def apply(self, payload: dict[str, Any], *, rng: random.Random) -> list[dict[str, Any]]:
        del rng
        result = copy.deepcopy(payload)
        changed = False
        for index, content in _message_paths(result):
            transformed = content
            for pattern, replacement in _WORD_REWRITES:
                transformed = pattern.sub(replacement, transformed, count=1)
            if transformed == content:
                transformed = f"Could you {content.rstrip('.?')}?"
            result["messages"][index]["content"] = transformed
            changed = True
        if not changed and isinstance(result.get("prompt"), str):
            prompt = result["prompt"]
            result["prompt"] = f"Could you {prompt.rstrip('.?')}?"
            changed = True
        return [result] if changed else []


@dataclass(frozen=True)
class EntitySwapRelation:
    pools: Mapping[str, Sequence[str]]
    name: str = "entity_substitution"

    def apply(self, payload: dict[str, Any], *, rng: random.Random) -> list[dict[str, Any]]:
        result = copy.deepcopy(payload)
        replacements: dict[str, str] = {}
        for pool in self.pools.values():
            candidates = [item for item in pool if isinstance(item, str)]
            for source in candidates:
                if source in replacements:
                    continue
                options = [candidate for candidate in candidates if candidate != source]
                if options:
                    replacements[source] = rng.choice(options)
        if not replacements:
            return []

        # Single pass — sequential substitution lets circular swaps undo.
        entity_pattern = re.compile(
            r"\b("
            + "|".join(re.escape(source) for source in sorted(replacements, key=len, reverse=True))
            + r")\b"
        )

        def replace_text(text: str) -> str:
            return entity_pattern.sub(lambda match: replacements[match.group(0)], text)

        changed = False
        for index, content in _message_paths(result):
            transformed = replace_text(content)
            changed = changed or transformed != content
            result["messages"][index]["content"] = transformed
        if isinstance(result.get("prompt"), str):
            prompt = result["prompt"]
            transformed = replace_text(prompt)
            changed = changed or transformed != prompt
            result["prompt"] = transformed
        return [result] if changed else []


@dataclass(frozen=True)
class ContextOrderPermutationRelation:
    name: str = "retrieval_context_order_invariance"

    def apply(self, payload: dict[str, Any], *, rng: random.Random) -> list[dict[str, Any]]:
        messages = payload.get("messages")
        if not isinstance(messages, list) or len(messages) < 3:
            return []
        result = copy.deepcopy(payload)
        copied_messages = result["messages"]
        # System messages remain first; the final non-system user request stays
        # last. Only intermediate context is shuffled.
        start = (
            1
            if isinstance(copied_messages[0], Mapping)
            and copied_messages[0].get("role") == "system"
            else 0
        )
        end = len(copied_messages) - 1
        if end - start < 2:
            return []
        context = copied_messages[start:end]
        shuffled = list(context)
        rng.shuffle(shuffled)
        if shuffled == context:
            shuffled = list(reversed(context))
        result["messages"][start:end] = shuffled
        return [result]


@dataclass(frozen=True)
class AdversarialNoiseRelation:
    name: str = "adversarial_noise_invariance"

    def apply(self, payload: dict[str, Any], *, rng: random.Random) -> list[dict[str, Any]]:
        del rng
        result = copy.deepcopy(payload)

        def noise(text: str) -> str:
            text = re.sub(r"\s+", " ", text).strip()
            return f"Hey — {text} ... thanks!"

        changed = False
        for index, content in _message_paths(result):
            result["messages"][index]["content"] = noise(content)
            changed = True
        if isinstance(result.get("prompt"), str):
            result["prompt"] = noise(result["prompt"])
            changed = True
        return [result] if changed else []


def build_relations(
    relation_names: Sequence[str], *, entity_pools: Mapping[str, Sequence[str]] | None = None
) -> list[MetamorphicRelation]:
    """Build relation implementations from declarative config names."""
    mapping: dict[str, MetamorphicRelation] = {
        "paraphrase_invariance": ParaphraseRelation(),
        "entity_substitution": EntitySwapRelation(entity_pools or {}),
        "adversarial_noise_invariance": AdversarialNoiseRelation(),
        "retrieval_context_order_invariance": ContextOrderPermutationRelation(),
    }
    unknown = [name for name in relation_names if name not in mapping]
    if unknown:
        raise ValueError(f"Unknown metamorphic relations: {unknown}")
    return [mapping[name] for name in relation_names]
