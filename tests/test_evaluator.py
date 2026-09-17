from __future__ import annotations

from driftgate.metamorphic.evaluator import (
    DeterministicRubricJudge,
    HashingEmbedder,
    InvarianceChecker,
    LLMRubricJudge,
    RubricRequirements,
    cosine_similarity,
    parse_verdict,
)


def test_hashing_embedder_is_deterministic_and_similarity_is_sane() -> None:
    embedder = HashingEmbedder()
    vectors = embedder.embed(
        ["summarize this report", "summarize this report", "volcano eruption forecast"]
    )

    assert vectors[0].tolist() == vectors[1].tolist()
    assert cosine_similarity(vectors[0], vectors[1]) == 1.0
    assert cosine_similarity(vectors[0], vectors[2]) < 0.5


def test_invariance_checker_returns_binary_result() -> None:
    checker = InvarianceChecker(threshold=0.5)
    result = checker.check("summarize report", "summarize the report", "paraphrase_invariance")
    assert result.passed
    assert result.similarity >= result.threshold


def test_deterministic_judge_and_fail_closed_verdict_parser() -> None:
    judge = DeterministicRubricJudge(RubricRequirements(required_keywords=("alice",)))
    assert judge.score("", "Alice has access", "", rubric_id="entity").passed
    assert not judge.score("", "No person named here", "", rubric_id="entity").passed
    assert parse_verdict("reasoning\nVERDICT: PASS").passed
    assert not parse_verdict("looks correct").passed


def test_llm_adapter_retries_unparsable_output_once() -> None:
    responses = iter(["not a verdict", "reasoning\nVERDICT: PASS"])
    judge = LLMRubricJudge(lambda _prompt: next(responses))
    verdict = judge.score("instruction", "answer", "rubric")

    assert verdict.passed
