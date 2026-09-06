"""Realised performance once labels arrive (they always arrive late in credit: a default is
observed months after the decision), compared with the baseline's labelled performance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .baseline import Baseline
from .metrics import summarize


@dataclass
class PerformanceResult:
    n_scored: int
    n_labelled: int
    label_coverage: float
    min_labels: int
    metrics: dict[str, float] | None
    baseline: dict[str, float] | None
    auc_drop: float | None = None
    brier_increase: float | None = None

    @property
    def evaluated(self) -> bool:
        return self.metrics is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def join_labels(frame: pd.DataFrame, labels: pd.DataFrame | None) -> pd.DataFrame:
    """Left join on request_id; unlabelled rows keep NaN labels."""
    if labels is None or len(labels) == 0:
        out = frame.copy()
        out["label"] = np.nan
        return out
    cols = ["request_id", "label"]
    return frame.merge(labels[cols].drop_duplicates("request_id"), on="request_id", how="left")


def performance_check(
    baseline: Baseline, frame: pd.DataFrame, labels: pd.DataFrame | None, *, min_labels: int
) -> PerformanceResult:
    joined = join_labels(frame, labels)
    labelled = joined[joined["label"].notna()]
    n_scored = len(frame)
    n_labelled = len(labelled)
    coverage = n_labelled / n_scored if n_scored else 0.0
    result = PerformanceResult(
        n_scored, n_labelled, coverage, min_labels, None, baseline.performance
    )
    if n_labelled < min_labels:
        return result
    y = labelled["label"].to_numpy(dtype=np.int64)
    s = labelled["score"].to_numpy(dtype=np.float64)
    if y.min() == y.max():
        return result  # a single class: AUC undefined, nothing to compare
    metrics = summarize(y, s, baseline.score.threshold)
    result.metrics = metrics
    if baseline.performance:
        result.auc_drop = baseline.performance["auc"] - metrics["auc"]
        result.brier_increase = metrics["brier"] - baseline.performance["brier"]
    return result
