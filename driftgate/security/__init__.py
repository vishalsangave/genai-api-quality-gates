"""Adversarial testing for agents: attack library and success detectors."""

from driftgate.security.detectors import Finding, assert_attack_resisted, make_canary, scan
from driftgate.security.prompt_attacks import (
    ALL_ATTACKS,
    ATTACK_CATEGORIES,
    attacks_for,
    embed_in_document,
)

__all__ = [
    "ALL_ATTACKS",
    "ATTACK_CATEGORIES",
    "Finding",
    "assert_attack_resisted",
    "attacks_for",
    "embed_in_document",
    "make_canary",
    "scan",
]
