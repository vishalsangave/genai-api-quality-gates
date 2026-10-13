import pytest
from httpx import Client

from driftgate.metamorphic.evaluator import SemanticComparer

# This marker tells pytest to run this test function for each entry in the specified golden dataset.
# The 'prompt' and 'golden_response' arguments will be automatically populated from the file.
@pytest.mark.golden_dataset("examples/tests/golden_multi_agent.jsonl")
def test_golden_regressions_with_judge(
    prompt: dict,
    golden_response: dict,
    driftgate_client: Client,
    driftgate_evaluator: SemanticComparer,
):
    """
    This test demonstrates how to use a golden dataset and an LLM-as-a-Judge for regression testing.
    It sends a prompt from the dataset to the service and uses the evaluator to check if the
    new response is semantically equivalent to the golden one.
    """
    # The service base URL is configured in driftgate.config.yaml.
    # The driftgate_client fixture handles recording and replaying cassettes.
    response = driftgate_client.post("/chat", json=prompt)
    actual_response = response.json()

    # The driftgate_evaluator fixture provides a judge (powered by Ollama or another LLM)
    # that can compare the two responses.
    verdict = driftgate_evaluator.compare(
        prompt=str(prompt),
        golden_response=str(golden_response),
        actual_response=str(actual_response),
    )

    # The test passes if the judge returns a 'PASS' verdict.
    # The rationale from the judge provides a clear explanation on failure.
    assert verdict.passed, f"LLM Judge failed the semantic similarity check:\n{verdict.rationale}"


def test_math_agent_tool_calls(driftgate_client: Client):
    """
    This test demonstrates how to make assertions on the tool calls made by an agent.
    """
    prompt = {"prompt": "calculate 2 + 2"}
    response = driftgate_client.post("/chat", json=prompt)
    data = response.json()

    # 1. Assert that the correct agent was dispatched.
    assert data["agent_name"] == "MathAgent"

    # 2. Assert that the agent made the expected number of tool calls.
    assert len(data["tool_calls"]) == 1

    # 3. Assert on the specifics of the tool call.
    tool_call = data["tool_calls"][0]
    assert tool_call["tool_name"] == "calculator"
    assert tool_call["tool_input"] == prompt["prompt"]

import jsonschema

@pytest.mark.agent_contract("examples/multi_agent_service/math_agent_contract.yaml")
def test_math_agent_contract(driftgate_client: Client, driftgate_agent_contract):
    """
    This test demonstrates how to use an Agent Contract to validate an agent's output.
    """
    prompt = {"prompt": "calculate 10 / 2"}
    response = driftgate_client.post("/chat", json=prompt)
    data = response.json()

    # The driftgate_agent_contract fixture provides the loaded and validated contract.
    # We can use the output_schema from the contract to validate the response data.
    try:
        jsonschema.validate(instance=data, schema=driftgate_agent_contract.output_schema)
    except jsonschema.exceptions.ValidationError as e:
        pytest.fail(f"Response did not match the agent contract's output schema:\n{e}")

    # We can also use the contract to validate the tool calls.
    for tool_call in data.get("tool_calls", []):
        assert tool_call["tool_name"] in driftgate_agent_contract.allowed_tools, \
            f"Agent used a disallowed tool: {tool_call['tool_name']}"


@pytest.mark.red_team
def test_prompt_injection(attack_prompt: str, driftgate_client: Client):
    """
    This test uses the @pytest.mark.red_team marker to run a battery of prompt injection attacks.
    It asserts that the service returns a default refusal and does not execute a hijacked instruction.
    """
    response = driftgate_client.post("/chat", json={"prompt": attack_prompt})
    data = response.json()

    # For our mock service, a successful defense is returning the default response.
    # For a real LLM, you might check that the response contains a refusal phrase
    # like "I cannot fulfill that request."
    assert data["agent_name"] == "DefaultAgent"
    assert "I'm not sure how to handle that" in data["response"]


def test_math_run_trace_is_captured_and_within_budget(driftgate_client: Client, driftgate_traces):
    """Traces returned by the service are captured automatically (no manual wiring)."""
    from driftgate.agents import TraceBudget, assert_trace

    driftgate_client.post("/chat", json={"prompt": "calculate 6 * 7"})

    assert_trace(
        driftgate_traces.last,
        budget=TraceBudget(max_steps=4, max_tool_calls=1),
        expected_path=["Orchestrator", "MathAgent"],
        required_tools=["calculator"],
        forbidden_tools=["shell"],
    )
