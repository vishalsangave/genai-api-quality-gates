import pytest

from driftgate.agents import ToolCall, TraceRecorder
from driftgate.security import (
    ALL_ATTACKS,
    ATTACK_CATEGORIES,
    assert_attack_resisted,
    attacks_for,
    embed_in_document,
    make_canary,
    scan,
)


def test_categories_cover_agent_threats_and_all_attacks_is_union():
    assert {"tool_abuse", "exfiltration", "inter_agent", "indirect_injection"} <= set(ATTACK_CATEGORIES)
    assert len(ALL_ATTACKS) == sum(len(v) for v in ATTACK_CATEGORIES.values())
    assert attacks_for("tool_abuse") == ATTACK_CATEGORIES["tool_abuse"]
    with pytest.raises(ValueError, match="unknown attack categories"):
        attacks_for("nope")


def test_embed_in_document_keeps_original_content():
    doc = embed_in_document("Quarterly report", ATTACK_CATEGORIES["indirect_injection"][0])
    assert doc.startswith("Quarterly report") and "pwned" in doc


def test_scan_detects_canary_marker_secret_and_tool():
    canary = make_canary()
    assert make_canary() != canary
    rec = TraceRecorder()
    rec.step("Math", tool_calls=[ToolCall(name="shell")])
    kinds = {f.kind for f in scan(
        f"Sure! {canary} pwned AKIAABCDEFGHIJKLMNOP", trace=rec.finish(),
        canaries=[canary], forbidden_tools=["shell"], injected_markers=["PWNED"],
    )}
    assert kinds == {"canary_leak", "injection_obeyed", "secret_pattern", "forbidden_tool"}


def test_clean_output_passes_and_dirty_raises():
    assert_attack_resisted("I can't help with that.", canaries=[make_canary()])
    with pytest.raises(AssertionError, match="canary_leak"):
        c = make_canary()
        assert_attack_resisted(f"the secret is {c}", canaries=[c])
