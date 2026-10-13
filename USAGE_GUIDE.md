# DriftGate Usage Guide

This guide provides a detailed overview of DriftGate's features and how to use them to ensure the quality, security, and reliability of your AI applications.

## Table of Contents

1.  [Configuration (`driftgate.config.yaml`)](#1-configuration-driftgateconfigyaml)
2.  [Golden Dataset Testing](#2-golden-dataset-testing)
3.  [LLM-as-a-Judge](#3-llm-as-a-judge)
4.  [Multi-Agent Contract Testing](#4-multi-agent-contract-testing)
5.  [Red Teaming & Security](#5-red-teaming--security)

---

## 1. Configuration (`driftgate.config.yaml`)

The `driftgate.config.yaml` file is the central control panel for the framework. The `driftgate init` command generates a well-commented version of this file.

### Key Sections:

*   `service` & `traffic_mode`: Defines the target service and how API traffic should be handled (e.g., `record`, `replay_strict`, `auto_record`).
*   `contract`: Configures Layer-1 validation against an OpenAPI specification.
*   `judge`: **(New)** Configures the LLM-as-a-Judge provider (Ollama, Anthropic, OpenAI) and model.
*   `governor`: Sets the statistical thresholds for making `PROMOTE/HOLD/ROLLBACK` decisions in a CI/CD environment.

---

## 2. Golden Dataset Testing

Golden dataset testing is the best way to prevent regressions in your model's behavior for critical, known inputs.

### Workflow:

**Step 1: Record Real Traffic**

Run your tests in `record` mode to capture live interactions into `.yaml` files called "cassettes".

```bash
uv run pytest --cassette-mode=record
```

**Step 2: Interactively Approve Golden Examples**

Use the `dataset approve` CLI command to review the recorded interactions. For each prompt/response pair, you can approve (`y`) or reject (`n`) it.

```bash
# Point the command at a cassette file from the previous step
uv run driftgate dataset approve cassettes/your_test_cassette.yaml --out my_golden_dataset.jsonl
```

Approved pairs are saved to a `.jsonl` file, where each line is a JSON object containing the `prompt` and `response`.

**Step 3: Create a Regression Test**

In your test suite, create a test function that uses the `@pytest.mark.golden_dataset` marker.

```python
import pytest
from httpx import Client
from driftgate.metamorphic.evaluator import SemanticComparer

@pytest.mark.golden_dataset("my_golden_dataset.jsonl")
def test_golden_regressions(
    prompt: dict,
    golden_response: dict,
    driftgate_client: Client,
    driftgate_evaluator: SemanticComparer, # The LLM Judge fixture
):
    response = driftgate_client.post("/chat", json=prompt)
    verdict = driftgate_evaluator.compare(
        prompt=str(prompt),
        golden_response=str(golden_response),
        actual_response=str(response.json()),
    )
    assert verdict.passed, verdict.rationale
```

This single test function automatically expands into a test for *every single entry* in your golden dataset.

---

## 3. LLM-as-a-Judge

A simple `assert response == golden_response` is too brittle for non-deterministic LLMs. The LLM-as-a-Judge solves this by using a powerful LLM to check if the *meaning* is the same.

### Configuration:

In your `driftgate.config.yaml`, configure your chosen provider:

```yaml
judge:
  provider: "ollama" # Or "anthropic", "openai"

  ollama:
    model: "phi3"
    base_url: "http://localhost:11434"

  # anthropic:
  #   model: "claude-3-haiku-20240307"

  # openai:
  #   model: "gpt-4-turbo"
```

API keys for cloud providers are loaded securely from the environment variable specified by `api_key_env`.

### Usage:

The `driftgate_evaluator` pytest fixture provides a ready-to-use judge instance in your tests. See the golden dataset example above for a practical use case.

---

## 4. Multi-Agent Contract Testing

To ensure reliable interactions in a multi-agent system, you can define and test against Agent Contracts.

### Workflow:

**Step 1: Define the Contract**

Create a `.yaml` file for your agent, defining its role, allowed tools, and I/O schemas.

```yaml
# math_agent_contract.yaml
name: "MathAgent"
description: "Handles mathematical calculations."
allowed_tools:
  - "calculator"
output_schema:
  type: "object"
  properties:
    agent_name: { "const": "MathAgent" }
    tool_calls: { "type": "array" }
  required: ["agent_name", "tool_calls"]
```

**Step 2: Create a Contract Test**

Use the `@pytest.mark.agent_contract` marker and the `driftgate_agent_contract` fixture.

```python
import pytest
from httpx import Client
import jsonschema

@pytest.mark.agent_contract("path/to/math_agent_contract.yaml")
def test_math_agent_contract(driftgate_client: Client, driftgate_agent_contract):
    response = driftgate_client.post("/chat", json={"prompt": "calculate 10 / 2"})
    data = response.json()

    # 1. Validate the response against the contract's output schema
    jsonschema.validate(instance=data, schema=driftgate_agent_contract.output_schema)

    # 2. Validate that only allowed tools were used
    for tool_call in data.get("tool_calls", []):
        assert tool_call["tool_name"] in driftgate_agent_contract.allowed_tools
```

---

## 5. Red Teaming & Security

DriftGate can automatically run a suite of adversarial prompts against your service to probe for vulnerabilities.

### Usage:

Simply add the `@pytest.mark.red_team` marker to a test function. The framework will automatically parameterize the test, running it once for every attack string in the built-in library.

```python
import pytest
from httpx import Client

@pytest.mark.red_team
def test_prompt_injection(attack_prompt: str, driftgate_client: Client):
    response = driftgate_client.post("/chat", json={"prompt": attack_prompt})
    data = response.json()

    # Assert that the service returns a safe, default refusal and is not hijacked.
    assert data["agent_name"] == "DefaultAgent"
    assert "I'm not sure how to handle that" in data["response"]
```

## Multi-agent testing (20+ agents)

Everything below is plain Python, so it runs in any pytest suite and costs nothing unless you add an LLM judge.

**1. Record a trace of the run** and keep it as a replayable fixture:

```python
from driftgate.agents import TraceRecorder, ToolCall

rec = TraceRecorder(task="book a flight")
rec.step("Router", output={"intent": "travel"}, handoff_to="Planner", tokens=120)
rec.step("Planner", tool_calls=[ToolCall(name="search")], tokens=300)
trace = rec.finish("done")
trace.save("traces/book_flight.json")   # AgentTrace.load(...) in CI
```

**2. Enforce contracts for every agent at once.** Keep one YAML per agent in a directory:

```python
from driftgate.agents import ContractRegistry, assert_trace, TraceBudget

registry = ContractRegistry.from_dir("contracts/")
assert not registry.check_graph()              # handoff edges are schema-compatible
assert_trace(
    trace,
    registry=registry,                         # schemas, allowed tools, allowed handoffs
    budget=TraceBudget(max_steps=12, max_cost_usd=0.10, max_repeats=3),  # runaway loops
    expected_path=["Router", "Planner"],
    forbidden_tools=["shell"],
)
```

**3. Score behaviour, DeepEval-style.** Deterministic metrics read the trace; rubric metrics need an explicit judge (there is no silent default):

```python
from driftgate.metrics import TestCase, ToolCorrectness, HandoffCorrectness, Faithfulness, assert_test

case = TestCase(input="book a flight", actual_output=answer, context=docs,
                trace=trace, expected_tools=["search"], expected_path=["Router", "Planner"])
assert_test(case, [ToolCorrectness(), HandoffCorrectness(), Faithfulness(judge)])
```

**4. Red-team by category**, including attacks that arrive through tool output rather than the user:

```python
@pytest.mark.red_team("tool_abuse", "exfiltration", "inter_agent")
def test_resists(attack_prompt, driftgate_client): ...

from driftgate.security import make_canary, embed_in_document, assert_attack_resisted
canary = make_canary()                          # plant in the system prompt / secret store
assert_attack_resisted(output, trace=trace, canaries=[canary], forbidden_tools=["send_email"])
```

## Golden datasets: check them, or generate one for your application

A golden dataset is JSONL: `{"prompt": <request body>, "response": <approved response body>}` per line.

**If the file is missing, empty or malformed, the test is skipped, not errored**, with a note saying how to create it
(`pytest -rs` shows the reason). Teams can add `@pytest.mark.golden_dataset(...)` before the dataset exists and nothing
looks broken. Set `datasets.on_problem: fail` in `driftgate.config.yaml` to make CI strict.

```bash
# Diagnose a file (line numbers for every problem); --fix drops bad/duplicate lines and keeps a .bak
driftgate dataset check golden_dataset.jsonl [--fix]

# Create one for YOUR application. Pick one source:
driftgate dataset generate --from-cassettes cassettes/ --endpoint /chat      # convert recorded traffic
driftgate dataset generate --spec openapi.yaml --endpoint /chat              # requests built from the spec's schemas/examples
driftgate dataset generate --prompts-file prompts.txt --endpoint /chat \
    --request-template '{"messages": [{"role": "user", "content": "{prompt}"}]}'   # your own prompts, your request shape
```

`--spec` and `--prompts-file` send real requests to the running service (`--base-url`, or `service.base_url` in the config)
and keep its 2xx responses. The dataset records what the service does today, so review it before treating it as the baseline.
Re-running `generate` appends only new examples.
