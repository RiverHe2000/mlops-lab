"""Uncertainty and stability statistics: bootstrap CIs, paired deltas, PSI."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

Metric = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    lower: float
    upper: float
    level: float
    n_boot: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        return f"{self.estimate:+.4f} [{self.lower:+.4f}, {self.upper:+.4f}]"


def _percentile_ci(values: np.ndarray, level: float) -> tuple[float, float]:
    alpha = (1.0 - level) / 2.0
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan"), float("nan")
    return float(np.quantile(finite, alpha)), float(np.quantile(finite, 1.0 - alpha))


def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    metric: Metric,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> BootstrapResult:
    """Percentile bootstrap over rows for a single model's metric."""
    y = np.asarray(y)
    p = np.asarray(p)
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = metric(y[idx], p[idx])
    lo, hi = _percentile_ci(stats, level)
    return BootstrapResult(float(metric(y, p)), lo, hi, level, n_boot)


def paired_bootstrap_delta(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    metric: Metric,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> BootstrapResult:
    """CI for metric(b) - metric(a) resampling the *same* rows for both models.

    Pairing removes the between-sample variance that an unpaired comparison would add, which
    is what makes small-holdout model comparisons decidable at all.
    """
    y = np.asarray(y)
    p_a = np.asarray(p_a)
    p_b = np.asarray(p_b)
    if not (len(y) == len(p_a) == len(p_b)):
        raise ValueError("paired inputs must have equal length")
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[b] = metric(y[idx], p_b[idx]) - metric(y[idx], p_a[idx])
    lo, hi = _percentile_ci(deltas, level)
    return BootstrapResult(float(metric(y, p_b) - metric(y, p_a)), lo, hi, level, n_boot)


def psi(reference: np.ndarray, current: np.ndarray, *, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index with quantile bins fitted on the reference distribution.

    PSI = Σ_b (c_b - r_b) · ln(c_b / r_b), with proportions floored at `eps` so an empty bin
    cannot produce an infinite score. Rules of thumb: < 0.10 stable, 0.10-0.25 watch, > 0.25 shift.
    """
    reference = np.asarray(reference, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    if len(reference) == 0 or len(current) == 0:
        raise ValueError("empty inputs")
    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:  # constant reference: PSI is 0 iff current is identical
        return 0.0 if bool(np.all(current == reference[0])) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, edges)[0] / len(reference)
    c = np.histogram(current, edges)[0] / len(current)
    r = np.clip(r, eps, None)
    c = np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))
