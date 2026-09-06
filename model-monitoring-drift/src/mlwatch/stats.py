"""Statistics used by the drift detectors, written from their definitions.

PSI, Jensen-Shannon divergence on binned proportions, Benjamini-Hochberg control of the false
discovery rate across features, a bootstrap interval for PSI, and the binning helpers that
turn a baseline's quantile edges into proportions.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

EPS = 1e-4


def quantile_edges(values: np.ndarray, n_bins: int = 10) -> list[float]:
    """Interior bin edges at the reference quantiles (deduplicated, so heavy ties collapse bins)."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return []
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    return [float(e) for e in np.unique(np.quantile(values, qs))]


def bin_proportions(values: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """Proportions over len(edges)+1 bins: (-inf, e1), [e1, e2), ..., [ek, +inf)."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    n_bins = len(edges) + 1
    if len(values) == 0:
        return np.zeros(n_bins)
    idx = np.searchsorted(np.asarray(edges, dtype=np.float64), values, side="right")
    counts = np.bincount(idx, minlength=n_bins).astype(np.float64)
    return counts / counts.sum()


def psi(reference: np.ndarray, current: np.ndarray, eps: float = EPS) -> float:
    """Population Stability Index Σ (c - r)·ln(c / r) over aligned proportions, floored at eps.

    Rules of thumb used in credit-risk monitoring: < 0.10 stable, 0.10-0.25 investigate,
    > 0.25 significant shift. It is symmetric in r and c and unbounded above.
    """
    r = np.clip(np.asarray(reference, dtype=np.float64), eps, None)
    c = np.clip(np.asarray(current, dtype=np.float64), eps, None)
    if r.shape != c.shape:
        raise ValueError("proportion vectors must align")
    return float(np.sum((c - r) * np.log(c / r)))


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence in bits: 0 for identical distributions, 1 for disjoint ones."""
    p = np.clip(np.asarray(p, dtype=np.float64), eps, None)
    q = np.clip(np.asarray(q, dtype=np.float64), eps, None)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log2(p / m)))
    kl_qm = float(np.sum(q * np.log2(q / m)))
    return max(0.0, min(1.0, 0.5 * kl_pm + 0.5 * kl_qm))


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """BH-adjusted p-values (monotone, capped at 1); NaNs stay NaN and are not counted."""
    p = np.asarray(p_values, dtype=np.float64)
    out = np.full(p.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    m = len(valid)
    if m == 0:
        return [float(x) for x in out]
    order = valid[np.argsort(p[valid])]
    ranked = p[order] * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return [float(x) for x in out]


def bootstrap_psi(
    values: np.ndarray,
    edges: Sequence[float],
    reference_proportions: np.ndarray,
    *,
    n_boot: int = 200,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile interval for PSI under resampling of the *current* window."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        sample = values[rng.integers(0, len(values), len(values))]
        stats[b] = psi(reference_proportions, bin_proportions(sample, edges))
    alpha = (1 - level) / 2
    return float(np.quantile(stats, alpha)), float(np.quantile(stats, 1 - alpha))


def categorical_proportions(
    values: Sequence[str], categories: Sequence[str]
) -> tuple[np.ndarray, float]:
    """Proportions aligned to `categories` plus the share of values outside them."""
    counts = dict.fromkeys(categories, 0)
    unseen = 0
    for v in values:
        if v in counts:
            counts[v] += 1
        else:
            unseen += 1
    n = len(values)
    if n == 0:
        return np.zeros(len(categories)), 0.0
    props = np.array([counts[c] / n for c in categories], dtype=np.float64)
    return props, unseen / n
