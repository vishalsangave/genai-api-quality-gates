from __future__ import annotations

import random

from driftgate.metamorphic.relations import (
    AdversarialNoiseRelation,
    ContextOrderPermutationRelation,
    EntitySwapRelation,
    ParaphraseRelation,
    build_relations,
)


def _payload() -> dict:
    return {
        "messages": [
            {"role": "system", "content": "Be accurate"},
            {"role": "user", "content": "Please summarize Alice's report"},
            {"role": "user", "content": "Context document one"},
            {"role": "user", "content": "Give the final answer"},
        ]
    }


def test_paraphrase_and_noise_transform_payloads() -> None:
    payload = _payload()
    paraphrased = ParaphraseRelation().apply(payload, rng=random.Random(1))[0]
    noisy = AdversarialNoiseRelation().apply(payload, rng=random.Random(1))[0]

    assert paraphrased["messages"][1]["content"] != payload["messages"][1]["content"]
    assert noisy["messages"][1]["content"].startswith("Hey")
    assert payload["messages"][1]["content"].startswith("Please")


def test_entity_swap_and_context_permutation_are_seeded() -> None:
    payload = _payload()
    swapped = EntitySwapRelation({"names": ["Alice", "Bob"]}).apply(payload, rng=random.Random(5))[
        0
    ]
    first = ContextOrderPermutationRelation().apply(payload, rng=random.Random(3))[0]
    second = ContextOrderPermutationRelation().apply(payload, rng=random.Random(3))[0]

    assert "Bob" in swapped["messages"][1]["content"]
    assert first == second
    assert first["messages"][0] == payload["messages"][0]
    assert first["messages"][-1] == payload["messages"][-1]


def test_relation_builder_rejects_unknown_names() -> None:
    assert len(build_relations(["paraphrase_invariance"])) == 1
    try:
        build_relations(["not-real"])
    except ValueError as error:
        assert "Unknown" in str(error)
    else:
        raise AssertionError("unknown relation should raise")
