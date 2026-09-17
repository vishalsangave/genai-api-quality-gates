from __future__ import annotations

import numpy as np
import pytest

from driftgate.governor.statistics import EmptyMetricError, bootstrap_ci, mann_kendall


def test_bootstrap_is_reproducible_and_brackets_mean() -> None:
    values = [1, 2, 3, 4, 5]
    first = bootstrap_ci(values, rng=np.random.default_rng(7))
    second = bootstrap_ci(values, rng=np.random.default_rng(7))

    assert first == second
    assert first.low <= np.mean(values) <= first.high
    assert first.n == 5


def test_bootstrap_degenerate_and_invalid_samples() -> None:
    assert bootstrap_ci([4.2]).degenerate
    with pytest.raises(EmptyMetricError):
        bootstrap_ci([])
    with pytest.raises(ValueError, match="finite"):
        bootstrap_ci([1.0, float("nan")])


def test_mann_kendall_detects_trends_and_ties() -> None:
    rising = mann_kendall([1, 2, 3, 4, 5, 6])
    falling = mann_kendall([6, 5, 4, 3, 2, 1])
    flat = mann_kendall([3, 3, 3, 3, 3])
    short = mann_kendall([1, 2, 3])

    assert rising.trend == "increasing"
    assert falling.trend == "decreasing"
    assert flat.trend == "no_trend"
    assert short.trend == "insufficient_data"
