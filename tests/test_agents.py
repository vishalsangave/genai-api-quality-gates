import pytest

from driftgate.agents import (
    AgentContract,
    AgentTrace,
    ContractRegistry,
    ToolCall,
    TraceBudget,
    TraceRecorder,
    assert_trace,
    check_compatibility,
    check_trace,
)


def _contract(name, *, tools=(), handoffs=None, inp=None, out=None):
    return AgentContract(
        name=name, version="1.0.0", description=name, allowed_tools=list(tools),
        allowed_handoffs=handoffs, input_schema=inp or {}, output_schema=out or {},
    )


def _trace():
    rec = TraceRecorder("book a flight")
    rec.step("Router", input={"q": "x"}, output={"intent": "travel"}, handoff_to="Planner", tokens=10, cost_usd=0.01)
    rec.step("Planner", output={"plan": ["search"]}, tool_calls=[ToolCall(name="search")], handoff_to="Booker", tokens=20)
    rec.step("Booker", tool_calls=[ToolCall(name="book"), ToolCall(name="pay")], tokens=30)
    return rec.finish("done")


def test_trace_aggregates_and_roundtrips(tmp_path):
    t = _trace()
    assert t.agents_visited == ["Router", "Planner", "Booker"]
    assert t.handoffs == [("Router", "Planner"), ("Planner", "Booker")]
    assert t.tool_names == ["search", "book", "pay"]
    assert t.total_tokens == 60
    t.save(tmp_path / "nested" / "t.json")
    assert AgentTrace.load(tmp_path / "nested" / "t.json") == t


def test_budget_violations_are_reported():
    v = check_trace(_trace(), budget=TraceBudget(max_steps=2, max_tool_calls=2, max_tokens=50))
    assert {x.rule for x in v} == {"max_steps", "max_tool_calls", "max_tokens"}


def test_loop_detection_identical_input_and_cycles():
    rec = TraceRecorder()
    for _ in range(4):
        rec.step("Worker", input="same")
    rules = {x.rule for x in check_trace(rec.finish(), budget=TraceBudget(max_repeats=3))}
    assert rules == {"loop"}

    cyc = TraceRecorder()
    cyc.step("A", handoff_to="B")
    cyc.step("B", handoff_to="A")
    assert any("cycle" in x.message for x in check_trace(cyc.finish()))


def test_path_tools_and_edges():
    t = _trace()
    v = check_trace(
        t, expected_path=["Router", "Booker"], required_tools=["refund"],
        forbidden_tools=["pay"], allowed_edges=[("Router", "Planner")],
    )
    assert {x.rule for x in v} == {"expected_path", "required_tools", "forbidden_tools", "allowed_edges"}


def test_assert_trace_raises_with_all_violations():
    with pytest.raises(AssertionError, match=r"2 agent-trace violation"):
        assert_trace(_trace(), budget=TraceBudget(max_steps=1, max_tokens=1))


def test_contract_enforces_schema_tools_and_handoffs():
    router = _contract("Router", handoffs=["Planner"], out={"type": "object", "required": ["intent"]})
    planner = _contract("Planner", tools=["search"], handoffs=["Booker"])
    booker = _contract("Booker", tools=["book"])
    reg = ContractRegistry([router, planner, booker])
    v = reg.check_trace(_trace())
    assert [(x.agent, x.rule) for x in v] == [("Booker", "allowed_tools")]  # 'pay' not allowed
    assert "pay" in v[0].message

    bad = AgentTrace(steps=[_trace().steps[0].model_copy(update={"output": {}, "handoff_to": "Booker"})])
    rules = {x.rule for x in router.check_trace(bad)}
    assert rules == {"output_schema", "allowed_handoffs"}


def test_unknown_agents_flagged_unless_opted_out():
    reg = ContractRegistry([_contract("Router")])
    assert any(x.rule == "unknown_agent" for x in reg.check_trace(_trace()))
    assert not reg.check_trace(_trace(), require_known_agents=False)


def test_compatibility_detects_breaking_changes():
    producer = _contract("P", out={"properties": {"id": {"type": "integer"}}, "required": []})
    consumer = _contract("C", inp={"properties": {"id": {"type": "string"}, "q": {"type": "string"}},
                                   "required": ["id", "q"]})
    msgs = [x.message for x in check_compatibility(producer, consumer)]
    assert any("never declares" in m for m in msgs)
    assert any("may omit" in m for m in msgs)
    assert any("'integer' from producer" in m for m in msgs)


def test_registry_graph_check_and_dir_loading(tmp_path):
    (tmp_path / "a.yaml").write_text(
        "name: A\nversion: '1'\ndescription: a\nallowed_handoffs: [B, Ghost]\n"
        "output_schema: {properties: {x: {type: string}}, required: [x]}\n")
    (tmp_path / "b.yaml").write_text(
        "name: B\nversion: '1'\ndescription: b\ninput_schema: {properties: {x: {type: string}}, required: [x]}\n")
    reg = ContractRegistry.from_dir(tmp_path)
    assert reg.names == ["A", "B"]
    v = reg.check_graph()
    assert [(x.agent, x.rule) for x in v] == [("A", "unknown_agent")]
    with pytest.raises(ValueError, match="duplicate"):
        reg.add(_contract("A"))
