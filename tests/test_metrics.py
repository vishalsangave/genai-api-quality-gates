import pytest

from driftgate.agents import ToolCall, TraceRecorder
from driftgate.metamorphic.evaluator import JudgeVerdict
from driftgate.metrics import (
    AnswerRelevancy,
    Faithfulness,
    GEval,
    HandoffCorrectness,
    StepEfficiency,
    TaskCompletion,
    TestCase,
    ToolCorrectness,
    assert_test,
    evaluate,
)


class FakeJudge:
    def __init__(self, failing=()):
        self.failing, self.seen = failing, []

    def score(self, instruction, output, rubric, *, rubric_id=None):
        self.seen.append((instruction, rubric))
        ok = not any(f in rubric for f in self.failing)
        return JudgeVerdict(ok, "reason\nVERDICT: " + ("PASS" if ok else "FAIL"), rubric_id)


def _case(**kw):
    rec = TraceRecorder()
    rec.step("Router", handoff_to="Math")
    rec.step("Math", tool_calls=[ToolCall(name="calculator")])
    return TestCase(input="2+2?", actual_output="4", trace=rec.finish(), **kw)


def test_tool_correctness_f1():
    full = ToolCorrectness().measure(_case(expected_tools=["calculator"]))
    assert full.score == 1.0 and full.passed
    half = ToolCorrectness().measure(_case(expected_tools=["calculator", "search"]))
    assert half.score == pytest.approx(2 / 3) and not half.passed
    assert ToolCorrectness().measure(TestCase("q", "a")).score == 0.0


def test_handoff_correctness_is_order_aware():
    assert HandoffCorrectness().measure(_case(expected_path=["Router", "Math"])).passed
    r = HandoffCorrectness().measure(_case(expected_path=["Math", "Router"]))
    assert r.score == 0.5 and not r.passed


def test_step_efficiency_decays():
    assert StepEfficiency().measure(_case(expected_steps=2)).score == 1.0
    assert StepEfficiency().measure(_case(expected_steps=1)).score == 0.5


def test_rubric_metric_scores_fraction_and_explains_failures():
    judge = FakeJudge(failing=["unrelated"])
    r = AnswerRelevancy(judge).measure(_case())
    assert r.score == 0.5 and not r.passed and "unrelated" in r.reason


def test_faithfulness_passes_context_to_judge_only_when_needed():
    judge = FakeJudge()
    Faithfulness(judge).measure(_case(context=["Paris is in France"]))
    assert all("Paris is in France" in i for i, _ in judge.seen)
    judge2 = FakeJudge()
    TaskCompletion(judge2).measure(_case(context=["ignored"]))
    assert all("ignored" not in i for i, _ in judge2.seen)


def test_assert_test_and_evaluate_report():
    ok = _case(expected_tools=["calculator"])
    assert_test(ok, [ToolCorrectness(), GEval("Polite", ["Is it polite?"], FakeJudge())])
    bad = _case(expected_tools=["search"])
    with pytest.raises(AssertionError, match=r"FAIL.*ToolCorrectness"):
        assert_test(bad, [ToolCorrectness()])
    report = evaluate([ok, bad], [ToolCorrectness()])
    assert not report.passed and report.summary().count("[FAIL]") == 1
