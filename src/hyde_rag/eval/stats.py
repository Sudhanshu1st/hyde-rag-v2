"""Uncertainty for per-query metrics (D-EVAL-4).

* paired bootstrap over queries -> 95% CI of a system's mean and of its difference to a baseline
* paired randomisation (sign-flip) test -> two-sided p-value vs the baseline
* Holm-Bonferroni correction across the systems compared on one dataset
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_RESAMPLES = 10_000
SEED = 2212_10496  # arXiv id of the HyDE paper; fixed for reproducibility


def _aligned(a: dict[str, float], b: dict[str, float] | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    keys = sorted(a)
    if b is not None and set(b) != set(a):
        raise ValueError("Paired statistics need the same query set for both systems.")
    x = np.array([a[k] for k in keys])
    y = np.array([b[k] for k in keys]) if b is not None else None
    return x, y


def bootstrap_ci(
    per_query: dict[str, float], *, n: int = N_RESAMPLES, seed: int = SEED, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Mean and percentile bootstrap CI over queries."""
    x, _ = _aligned(per_query)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


@dataclass(frozen=True)
class PairedComparison:
    delta: float  # mean(system) - mean(baseline)
    ci_low: float
    ci_high: float
    p_value: float  # two-sided paired randomisation test
    p_holm: float | None = None


def compare_paired(
    system: dict[str, float],
    baseline: dict[str, float],
    *,
    n: int = N_RESAMPLES,
    seed: int = SEED,
    alpha: float = 0.05,
) -> PairedComparison:
    x, y = _aligned(system, baseline)
    d = x - y
    rng = np.random.default_rng(seed)
    # Bootstrap CI of the mean difference (resampling queries keeps the pairing).
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    # Randomisation test: under H0 each per-query difference is equally likely to flip sign.
    signs = rng.choice([-1.0, 1.0], size=(n, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    observed = abs(d.mean())
    p = (np.sum(null >= observed - 1e-12) + 1) / (n + 1)
    return PairedComparison(float(d.mean()), float(lo), float(hi), float(p))


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[name] = running
    return adjusted
