"""Paired statistics for the sensitivity matrix.

Seeds are few (3), so a t-interval is wider than the effect it estimates. The
primary interval is a bootstrap over statements, stratified by seed: statements
are resampled inside each seed, then averaged across seeds. MDE is derived, not
asserted.
"""
from __future__ import annotations

import math
from statistics import NormalDist
from typing import Iterable, Mapping, Sequence

import numpy as np


def paired_bootstrap(per_seed: Mapping[str, Sequence[float]], n_boot: int = 1000, seed: int = 42, alpha: float = 0.05):
    """Mean paired difference and its percentile interval.

    per_seed: {seed_label: [delta per statement]}
    """
    if not per_seed:
        return {"mean": None, "lo": None, "hi": None, "n_statements": 0}
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(v, dtype=float) for v in per_seed.values() if len(v)]
    means = []
    for _ in range(n_boot):
        seed_means = [float(rng.choice(a, size=len(a), replace=True).mean()) for a in arrays]
        means.append(float(np.mean(seed_means)))
    return {
        "mean": float(np.mean([a.mean() for a in arrays])),
        "lo": float(np.percentile(means, 100 * alpha / 2)),
        "hi": float(np.percentile(means, 100 * (1 - alpha / 2))),
        "n_statements": int(sum(len(a) for a in arrays)),
    }


def mde(sd: float, n: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable effect: delta = (z_{1-a/2} + z_{1-b}) * sd / sqrt(n)."""
    z_a = NormalDist().inv_cdf(1 - alpha / 2)
    z_b = NormalDist().inv_cdf(power)
    return (z_a + z_b) * sd / math.sqrt(n)


def required_n(sd: float, delta: float, alpha: float = 0.05, power: float = 0.8) -> int:
    z_a = NormalDist().inv_cdf(1 - alpha / 2)
    z_b = NormalDist().inv_cdf(power)
    return math.ceil(((z_a + z_b) * sd / delta) ** 2)
