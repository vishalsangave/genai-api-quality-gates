"""Distribution-aware statistics for DriftGate's release governor.

No SciPy dependency is required: bootstrap confidence intervals use NumPy and
the Mann-Kendall test uses the normal distribution's complementary error
function from the standard library. Both functions are deterministic when a
seeded ``numpy.random.Generator`` is provided.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import erfc, sqrt
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


class EmptyMetricError(ValueError):
    """A metric had no observations; a confidence interval would be fiction."""


@dataclass(frozen=True)
class CIResult:
    low: float
    high: float
    point: float
    level: float
    n: int
    degenerate: bool


@dataclass(frozen=True)
class MKResult:
    s: float
    var_s: float
    z: float
    p_value: float
    trend: Literal["increasing", "decreasing", "no_trend", "insufficient_data"]


def _as_finite_array(values: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float).ravel()
    if array.size == 0:
        raise EmptyMetricError("Cannot calculate a confidence interval from zero observations")
    if not np.isfinite(array).all():
        raise ValueError("Metric observations must be finite numbers (no NaN or infinity)")
    return array


def bootstrap_ci(
    values: ArrayLike,
    *,
    stat: Callable[..., float | NDArray[np.float64]] = np.mean,
    B: int = 10_000,
    level: float = 0.95,
    rng: np.random.Generator | None = None,
    block_size: int = 5_000,
) -> CIResult:
    """Calculate a non-parametric bootstrap confidence interval.

    ``B=10,000`` is the project default. Sampling is vectorized, but is
    chunked when an enormous metric vector would otherwise create a giant
    ``B × n`` index matrix. ``stat`` must accept ``axis=1`` for the vectorized
    fast path; a small fallback maps rows one at a time for custom callables.
    """
    if B <= 0:
        raise ValueError("B must be greater than zero")
    if not 0 < level < 1:
        raise ValueError("level must be between zero and one")
    if block_size <= 0:
        raise ValueError("block_size must be greater than zero")

    samples = _as_finite_array(values)
    n = samples.size
    try:
        point = float(stat(samples))
    except TypeError:
        point = float(np.mean(samples))

    if n == 1:
        return CIResult(samples[0], samples[0], point, level, n, True)

    generator = rng or np.random.default_rng()
    boot_stats = np.empty(B, dtype=float)
    offset = 0
    # Keep sampled value matrices under ~B * block_size elements rather than
    # accidentally allocating B*n for very large run histories.
    while offset < B:
        count = min(block_size, B - offset)
        indices = generator.integers(0, n, size=(count, n))
        resamples = samples[indices]
        try:
            computed = np.asarray(stat(resamples, axis=1), dtype=float)
        except TypeError:
            computed = np.asarray([float(stat(row)) for row in resamples], dtype=float)
        boot_stats[offset : offset + count] = computed
        offset += count

    alpha = (1.0 - level) / 2.0
    low, high = np.percentile(boot_stats, [alpha * 100.0, (1.0 - alpha) * 100.0])
    return CIResult(float(low), float(high), point, level, n, False)


def mann_kendall(series: ArrayLike, *, alpha: float = 0.05, min_n: int = 4) -> MKResult:
    """Run a tie-corrected two-sided Mann-Kendall monotonic trend test.

    ``S`` is deliberately returned raw. Its meaning is mapped to business
    health by ``MetricDirection`` in ``gate.py``: an increasing latency trend
    is unhealthy, while an increasing task-success trend is healthy.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if min_n < 2:
        raise ValueError("min_n must be at least two")

    values = _as_finite_array(series)
    n = values.size
    if n < min_n:
        return MKResult(0.0, 0.0, 0.0, 1.0, "insufficient_data")

    # Lower triangle holds value[j] - value[i] for j > i, matching the
    # conventional S = sum(sign(x_j - x_i)) definition.
    differences = np.subtract.outer(values, values)
    s = float(np.sign(differences)[np.tril_indices(n, k=-1)].sum())
    _, counts = np.unique(values, return_counts=True)
    tie_correction = sum(int(t) * (int(t) - 1) * (2 * int(t) + 5) for t in counts if t > 1)
    var_s = (n * (n - 1) * (2 * n + 5) - tie_correction) / 18.0

    if var_s == 0:
        return MKResult(s, var_s, 0.0, 1.0, "no_trend")
    if s > 0:
        z = (s - 1.0) / sqrt(var_s)
    elif s < 0:
        z = (s + 1.0) / sqrt(var_s)
    else:
        z = 0.0
    p_value = erfc(abs(z) / sqrt(2.0))
    if p_value >= alpha:
        trend: Literal["increasing", "decreasing", "no_trend", "insufficient_data"] = "no_trend"
    else:
        trend = "increasing" if z > 0 else "decreasing"
    return MKResult(s, var_s, z, p_value, trend)
