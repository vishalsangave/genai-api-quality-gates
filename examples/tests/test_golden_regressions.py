import pytest
from httpx import Client

from driftgate.metamorphic.evaluator import SemanticComparer

# Datasets are normally created with `driftgate dataset approve`. The path is resolved as given,
# then relative to this file. A missing, empty or malformed dataset skips this test with a note on how to
# create it (set `datasets.on_problem: fail` in driftgate.config.yaml for strict CI).
@pytest.mark.golden_dataset("golden_multi_agent.jsonl")
def test_golden_regression(
    prompt: dict,
    golden_response: dict,
    driftgate_client: Client,
    driftgate_evaluator: SemanticComparer,
):
    """
    This test will be run for each entry in golden_dataset.jsonl.
    """
    # Send the prompt from the dataset to your service
    response = driftgate_client.post("/chat", json=prompt)
    actual_response = response.json()

    # Use the LLM-as-a-Judge to compare the responses
    verdict = driftgate_evaluator.compare(
        prompt=str(prompt),
        golden_response=str(golden_response),
        actual_response=str(actual_response),
    )

    # Assert that the verdict is a pass, with a helpful message on failure
    assert verdict.passed, f"LLM Judge failed the semantic similarity check:\n{verdict.rationale}"