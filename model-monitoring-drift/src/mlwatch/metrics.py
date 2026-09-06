"""Classifier metrics for the performance checks (rank AUC, KS, Brier, log loss, ECE)."""

from __future__ import annotations

import numpy as np


def _check(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    if y.shape != p.shape or len(y) == 0:
        raise ValueError("y and p must be non-empty and aligned")
    return y, p


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    y, p = _check(y, p)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    sorted_p = p[order]
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    u = ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    y, p = _check(y, p)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ys, ps = y[order], p[order]
    cum_pos = np.cumsum(ys) / n_pos
    cum_neg = np.cumsum(1 - ys) / n_neg
    last = np.r_[ps[1:] != ps[:-1], True]
    return float(np.max(np.abs(cum_pos[last] - cum_neg[last])))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    y, p = _check(y, p)
    return float(np.mean((p - y) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:
    y, p = _check(y, p)
    pc = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    y, p = _check(y, p)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def summarize(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    y, p = _check(y, p)
    return {
        "n_labels": float(len(y)),
        "positive_rate": float(y.mean()),
        "auc": roc_auc(y, p),
        "ks": ks_statistic(y, p),
        "brier": brier(y, p),
        "log_loss": log_loss(y, p),
        "ece": expected_calibration_error(y, p),
        "decline_rate": float((p >= threshold).mean()),
    }
