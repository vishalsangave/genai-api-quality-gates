"""Teaching suite for testing non-deterministic (AI-shaped) endpoints.

The order & export mock exposes two endpoints whose *content* varies per
request while the *contract* stays fixed. Each test below is a copy-paste
pattern adopters reuse for their own non-deterministic endpoints:

- the contract pins shape, never values — validation passes on every draw;
- output diversity is asserted with overwhelming-probability margins;
- meaning stability is asserted with the hashing-embedder invariance oracle
  (plus its fail-side check, so a weakened oracle cannot pass silently);
- a seed query parameter pins sampled values exactly;
- metamorphic relations and the deterministic rubric judge compose with
  random outputs; and
- generated adversarial payloads still land in the declared status set.

All randomness is server-side; every assertion here is shape-, diversity-,
or statistics-based (exact values only via the explicit seed).
"""

from __future__ import annotations

import os
import random

import httpx
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from driftgate.config import JudgeConfig
from driftgate.contracts.validator import ContractValidator
from driftgate.metamorphic.evaluator import (
    DeterministicRubricJudge,
    InvarianceChecker,
    RubricRequirements,
    evaluate_structural_invariance,
)
from driftgate.metamorphic.relations import EntitySwapRelation
from driftgate.pytest_plugin import record_run_metrics
from examples.tests.conftest import EXPRESS_DIR

BEARER = "Bearer mock-jwt-bearer-token-12345"
DIVERGENT_TEXT = "Refund processing failed upstream."


@pytest.fixture(scope="module")
def nondet_contract() -> ContractValidator:
    return ContractValidator.from_file(EXPRESS_DIR / "openapi.yaml")


@pytest.fixture(scope="module")
def nondet_client(express_base_url: str) -> httpx.Client:
    """Client for the non-deterministic demo endpoints, with availability guard.

    The endpoints are vendored additions: an external ``DRIFT_GATE_EXPRESS_URL``
    pointing at the unmodified reference server answers 404 where the vendored
    mock answers 401, and this suite skips instead of failing there.
    """
    client = httpx.Client(base_url=express_base_url, timeout=10.0)
    probe = client.get("/recommendations")
    if probe.status_code == 404:
        pytest.skip("external server predates the non-deterministic demo endpoints")
    yield client
    client.close()


def _auth(client: httpx.Client, method: str, path: str, **kwargs: object) -> httpx.Response:
    return client.request(method, path, headers={"Authorization": BEARER}, **kwargs)


def test_recommendations_contract_holds_on_every_draw(
    nondet_client: httpx.Client, nondet_contract: ContractValidator
) -> None:
    for _ in range(12):
        response = _auth(nondet_client, "GET", "/recommendations")
        assert response.status_code == 200
        body = response.json()
        assert nondet_contract.validate_response("GET", "/v1/recommendations", 200, body=body) == []


def test_recommendations_are_actually_nondeterministic(nondet_client: httpx.Client) -> None:
    seen = {
        tuple(
            sorted(
                item["sku"]
                for item in _auth(nondet_client, "GET", "/recommendations").json()[
                    "recommendations"
                ]
            )
        )
        for _ in range(20)
    }
    assert len(seen) >= 5, f"endpoint appears cached: only {len(seen)} distinct sets in 20 draws"


def test_recommendations_seed_is_reproducible(nondet_client: httpx.Client) -> None:
    first = _auth(nondet_client, "GET", "/recommendations", params={"seed": 42}).json()
    second = _auth(nondet_client, "GET", "/recommendations", params={"seed": 42}).json()
    other = _auth(nondet_client, "GET", "/recommendations", params={"seed": 43}).json()
    # generatedAt stays volatile: the seed pins sampled values, not metadata.
    assert first["requestId"] == second["requestId"]
    assert first["recommendations"] == second["recommendations"]
    assert (
        first["recommendations"] != other["recommendations"]
        or first["requestId"] != other["requestId"]
    )


def test_insights_contract_holds_on_every_draw(
    nondet_client: httpx.Client, nondet_contract: ContractValidator
) -> None:
    for _ in range(6):
        response = _auth(
            nondet_client,
            "POST",
            "/insights",
            json={"customerId": "cust-42", "prompt": "How healthy is the order pipeline?"},
        )
        assert response.status_code == 200
        assert (
            nondet_contract.validate_response("POST", "/v1/insights", 200, body=response.json())
            == []
        )


def test_insights_outputs_vary_but_meaning_is_stable(nondet_client: httpx.Client) -> None:
    draws = [
        _auth(
            nondet_client,
            "POST",
            "/insights",
            json={"customerId": "cust-42", "prompt": "Summarize order pipeline health."},
        ).json()["insight"]
        for _ in range(20)
    ]
    distinct = set(draws)
    assert len(distinct) >= 4, f"outputs collapsed to {len(distinct)} distinct strings"
    checker = InvarianceChecker()
    pair = checker.check(draws[0], draws[1], "live_draw_pair")
    assert pair.passed, f"draws diverged below threshold: similarity={pair.similarity}"


def test_invariance_oracle_catches_real_drift(nondet_client: httpx.Client) -> None:
    live = _auth(
        nondet_client,
        "POST",
        "/insights",
        json={"customerId": "cust-42", "prompt": "Summarize order pipeline health."},
    ).json()["insight"]
    checker = InvarianceChecker()
    result = checker.check(live, DIVERGENT_TEXT, "live_draw_pair")
    assert not result.passed, "oracle lost its teeth: unrelated text still passed"


def test_insights_metamorphic_entity_swap_stays_valid(
    nondet_client: httpx.Client, nondet_contract: ContractValidator
) -> None:
    payload = {
        "customerId": "cust-alpha",
        "prompt": "Flag anomalies for customer cust-alpha this week.",
    }
    relation = EntitySwapRelation(pools={"customers": ["cust-alpha", "cust-beta", "cust-gamma"]})
    variants = relation.apply(payload, rng=random.Random(7))
    assert variants, "entity swap produced no variant"
    variant = variants[0]
    assert variant["prompt"] != payload["prompt"]

    def _contract_valid(body: dict) -> bool:
        return nondet_contract.validate_response("POST", "/v1/insights", 200, body=body) == []

    for request_payload in (payload, variant):
        response = _auth(nondet_client, "POST", "/insights", json=request_payload)
        assert response.status_code == 200
    baseline_body = _auth(nondet_client, "POST", "/insights", json=payload).json()
    variant_body = _auth(nondet_client, "POST", "/insights", json=variant).json()
    assert evaluate_structural_invariance(_contract_valid, [baseline_body, variant_body]) == [
        True,
        True,
    ]


def test_insights_rubric_judge_passes_every_draw(nondet_client: httpx.Client) -> None:
    judge = DeterministicRubricJudge(
        RubricRequirements(
            required_keywords=("order", "status", "dashboard"),
            forbidden_patterns=(r"\brefund\b",),
        )
    )
    for _ in range(3):
        output = _auth(
            nondet_client,
            "POST",
            "/insights",
            json={"customerId": "cust-42", "prompt": "Judge this operational summary."},
        ).json()["insight"]
        verdict = judge.score("Summarize order pipeline health.", output, "ops-insight-rubric")
        assert verdict.passed, verdict.rationale


insights_payload_strategy = st.one_of(
    st.fixed_dictionaries(
        {
            "customerId": st.one_of(st.none(), st.integers(), st.text(max_size=60)),
            "prompt": st.one_of(st.none(), st.integers(), st.text(max_size=200)),
        }
    ),
    st.none(),
    st.integers(),
    st.text(max_size=80),
)

seed_strategy = st.one_of(
    st.none(), st.integers(min_value=-10, max_value=1_000_000), st.text(max_size=20)
)


class TestGeneratedTraffic:
    @settings(
        max_examples=40,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
        derandomize=True,
    )
    @given(payload=insights_payload_strategy)
    def test_generated_payloads_never_break_the_insights_endpoint(
        self, nondet_client: httpx.Client, nondet_contract: ContractValidator, payload
    ) -> None:
        response = _auth(nondet_client, "POST", "/insights", json=payload)
        assert response.status_code < 500, response.text[:300]
        assert response.status_code in {200, 400}
        if response.status_code == 200:
            assert (
                nondet_contract.validate_response("POST", "/v1/insights", 200, body=response.json())
                == []
            )

    @settings(
        max_examples=40,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
        derandomize=True,
    )
    @given(seed=seed_strategy)
    def test_generated_seeds_never_break_the_recommendations_endpoint(
        self, nondet_client: httpx.Client, nondet_contract: ContractValidator, seed
    ) -> None:
        params = {"seed": seed} if seed is not None else None
        response = _auth(nondet_client, "GET", "/recommendations", params=params)
        assert response.status_code < 500, response.text[:300]
        assert response.status_code in {200, 400}
        if response.status_code == 200:
            assert (
                nondet_contract.validate_response(
                    "GET", "/v1/recommendations", 200, body=response.json()
                )
                == []
            )


INSIGHT_INSTRUCTION = "How healthy is the order pipeline today?"
INSIGHT_RUBRIC = (
    "The insight must directly address the instruction about order pipeline "
    "health. It must be a coherent claim about order status propagation to "
    "dashboards. It must not fabricate specific numeric metrics, and it must "
    "not discuss unrelated topics such as refunds or payments."
)


def _llm_judge_or_skip() -> object:
    """Build the configured LLM judge, or skip when the metered path is unavailable.

    Execution contract: the test runs only when a judge is actually configured —
    the API key env resolves AND the provider SDK is installed. Anything else
    skips with the fix in the skip reason. The PR gate (-m "not nightly")
    never reaches these tests.
    """
    if not os.environ.get("DRIFT_GATE_JUDGE_API_KEY"):
        pytest.skip("DRIFT_GATE_JUDGE_API_KEY is not set; LLM-judge evaluation not requested")
    try:
        import anthropic  # noqa: F401
    except ImportError:
        pytest.skip("anthropic SDK not installed; run: uv sync --extra judge")
    from driftgate.metamorphic.factory import build_judge

    config = JudgeConfig(
        provider=os.environ.get("DRIFT_GATE_JUDGE_PROVIDER", "anthropic"),
        model=os.environ.get("DRIFT_GATE_JUDGE_MODEL"),
        api_key_env="DRIFT_GATE_JUDGE_API_KEY",
    )
    return build_judge(config)


@pytest.mark.nightly
def test_llm_judge_scores_insight_draws(
    nondet_client: httpx.Client, pytestconfig: pytest.Config
) -> None:
    judge = _llm_judge_or_skip()
    samples: list[float] = []
    for _ in range(5):
        output = _auth(
            nondet_client,
            "POST",
            "/insights",
            json={"customerId": "cust-42", "prompt": INSIGHT_INSTRUCTION},
        ).json()["insight"]
        verdict = judge.score(
            INSIGHT_INSTRUCTION, output, INSIGHT_RUBRIC, rubric_id="nightly-insight"
        )
        assert verdict.passed, f"judge fail-closed: {verdict.rationale}"
        samples.append(1.0 if verdict.passed else 0.0)
    record_run_metrics(
        pytestconfig,
        task_success_rate=sum(samples) / len(samples),
        task_success_samples=samples,
    )


@pytest.mark.nightly
def test_llm_judge_semantic_invariance_pair(nondet_client: httpx.Client) -> None:
    judge = _llm_judge_or_skip()
    draws = [
        _auth(
            nondet_client,
            "POST",
            "/insights",
            json={"customerId": "cust-42", "prompt": INSIGHT_INSTRUCTION},
        ).json()["insight"]
        for _ in range(2)
    ]
    rubric = (
        "Two candidate insights are shown. PASS only if both convey the same "
        "meaning about order status propagation to dashboards. FAIL if either "
        "asserts something the other does not."
    )
    verdict = judge.score(
        f"Compare these two outputs:\n1. {draws[0]}\n2. {draws[1]}",
        f"1. {draws[0]}\n2. {draws[1]}",
        rubric,
        rubric_id="nightly-semantic-invariance",
    )
    assert verdict.passed, verdict.rationale
