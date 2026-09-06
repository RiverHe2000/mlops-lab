"""Credit-scoring metrics implemented from their definitions (NumPy only).

Everything here is cross-checked against scikit-learn in the tests where an equivalent exists
(AUC, Brier, log loss); KS, ECE, decile lift and the cost metrics are checked against
hand-computed values.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-12


def _check(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y {y.shape} vs p {p.shape}")
    if len(y) == 0:
        raise ValueError("empty inputs")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must be 0/1")
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("p must lie in [0, 1]")
    return y, p


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties given the average rank (Mann-Whitney convention)."""
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """AUC = P(score_pos > score_neg) + ½·P(tie), via rank sums (the Mann-Whitney U)."""
    y, p = _check(y, p)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(p)
    u = ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def gini(auc: float) -> float:
    return 2.0 * auc - 1.0


def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    """Kolmogorov-Smirnov separation: max |F_bad(s) - F_good(s)| over score thresholds."""
    y, p = _check(y, p)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ys = y[order]
    ps = p[order]
    cum_pos = np.cumsum(ys) / n_pos
    cum_neg = np.cumsum(1 - ys) / n_neg
    # evaluate only at the last index of each distinct score so ties are handled correctly
    last = np.r_[ps[1:] != ps[:-1], True]
    return float(np.max(np.abs(cum_pos[last] - cum_neg[last])))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    y, p = _check(y, p)
    return float(np.mean((p - y) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:
    y, p = _check(y, p)
    pc = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))


def calibration_table(
    y: np.ndarray, p: np.ndarray, n_bins: int = 10, strategy: str = "uniform"
) -> pd.DataFrame:
    """Per-bin count, mean predicted PD and observed default rate."""
    y, p = _check(y, p)
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
        edges[0], edges[-1] = 0.0, 1.0
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows: list[dict[str, float]] = []
    for b in range(len(edges) - 1):
        mask = idx == b
        n = int(mask.sum())
        rows.append(
            {
                "bin": float(b),
                "lower": float(edges[b]),
                "upper": float(edges[b + 1]),
                "n": float(n),
                "mean_pred": float(p[mask].mean()) if n else float("nan"),
                "obs_rate": float(y[mask].mean()) if n else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(
    y: np.ndarray, p: np.ndarray, n_bins: int = 10, strategy: str = "uniform"
) -> float:
    """ECE = Σ_b (n_b / n) · |mean_pred_b - obs_rate_b| over non-empty bins."""
    table = calibration_table(y, p, n_bins=n_bins, strategy=strategy)
    table = table[table["n"] > 0]
    n = float(table["n"].sum())
    gaps = (table["mean_pred"] - table["obs_rate"]).abs()
    return float((table["n"] / n * gaps).sum())


def decile_lift(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Rank applicants by PD (highest first), report bad rate, capture and lift per decile."""
    y, p = _check(y, p)
    order = np.argsort(-p, kind="mergesort")
    ys = y[order]
    splits = np.array_split(np.arange(len(ys)), n_bins)
    base = ys.mean() if ys.mean() > 0 else EPS
    total_bad = max(int(ys.sum()), 1)
    rows: list[dict[str, float]] = []
    captured = 0
    for d, idx in enumerate(splits, start=1):
        if len(idx) == 0:
            continue
        bad = int(ys[idx].sum())
        captured += bad
        rows.append(
            {
                "decile": float(d),
                "n": float(len(idx)),
                "bad_rate": float(bad / len(idx)),
                "cum_bad_capture": float(captured / total_bad),
                "lift": float((bad / len(idx)) / base),
            }
        )
    return pd.DataFrame(rows)


def confusion_at(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    """Decision rule: predict default (decline) when p >= threshold."""
    y, p = _check(y, p)
    pred = (p >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "approval_rate": (tn + fn) / len(y),
        "bad_rate_among_approved": fn / (tn + fn) if tn + fn else float("nan"),
    }


def expected_cost(
    y: np.ndarray, p: np.ndarray, threshold: float, cost_fn: float = 5.0, cost_fp: float = 1.0
) -> float:
    """Mean cost per applicant under the dataset cost matrix (missed default = 5, lost good = 1)."""
    c = confusion_at(y, p, threshold)
    return float((cost_fn * c["fn"] + cost_fp * c["fp"]) / len(np.asarray(y)))


def optimal_threshold_by_cost(
    y: np.ndarray, p: np.ndarray, cost_fn: float = 5.0, cost_fp: float = 1.0, grid: int = 199
) -> tuple[float, float]:
    """Grid search on (0, 1); returns (threshold, cost). Ties resolve to the higher threshold."""
    y, p = _check(y, p)
    thresholds = np.linspace(0.005, 0.995, grid)
    costs = np.array([expected_cost(y, p, t, cost_fn, cost_fp) for t in thresholds])
    best = np.flatnonzero(costs == costs.min())[-1]
    return float(thresholds[best]), float(costs[best])


def summarize(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    y, p = _check(y, p)
    auc = roc_auc(y, p)
    conf = confusion_at(y, p, threshold)
    return {
        "n": float(len(y)),
        "positive_rate": float(y.mean()),
        "auc": auc,
        "gini": gini(auc),
        "ks": ks_statistic(y, p),
        "brier": brier(y, p),
        "log_loss": log_loss(y, p),
        "ece": expected_calibration_error(y, p),
        "threshold": float(threshold),
        "expected_cost": expected_cost(y, p, threshold),
        "approval_rate": conf["approval_rate"],
        "recall": conf["recall"],
        "precision": conf["precision"],
        "bad_rate_among_approved": conf["bad_rate_among_approved"],
    }


def slice_metrics(
    y: np.ndarray, p: np.ndarray, groups: pd.Series, threshold: float
) -> pd.DataFrame:
    """Per-group performance and decision rates (fairness / stability view)."""
    y, p = _check(y, p)
    g = groups.astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    for value in sorted(set(g.tolist())):
        mask = g == value
        ys, ps = y[mask], p[mask]
        conf = confusion_at(ys, ps, threshold)
        rows.append(
            {
                "group": value,
                "n": int(mask.sum()),
                "positive_rate": float(ys.mean()),
                "auc": roc_auc(ys, ps),
                "approval_rate": conf["approval_rate"],
                "bad_rate_among_approved": conf["bad_rate_among_approved"],
            }
        )
    return pd.DataFrame(rows)
