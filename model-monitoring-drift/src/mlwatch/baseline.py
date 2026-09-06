"""The reference profile every window is compared against.

Built once from a trusted sample (the training data scored by the model, or the first weeks
of production), versioned with the model, and stored as JSON. Numeric features keep quantile
bin edges, proportions and a capped reference sample (for KS / Wasserstein); categorical
features keep their category proportions; the score keeps fixed-width bins and the decline
rate; optional labelled performance anchors the performance checks.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .metrics import summarize
from .stats import bin_proportions, quantile_edges

FeatureKind = Literal["numeric", "categorical"]
SCORE_EDGES = [round(0.1 * i, 1) for i in range(1, 10)]


class NumericProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    n: int
    missing_rate: float
    mean: float
    std: float
    min: float
    max: float
    quantiles: dict[str, float]
    bin_edges: list[float]
    bin_proportions: list[float]
    reference_sample: list[float]


class CategoricalProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    n: int
    missing_rate: float
    categories: dict[str, float]


class ScoreProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n: int
    mean: float
    bin_edges: list[float] = Field(default_factory=lambda: list(SCORE_EDGES))
    bin_proportions: list[float]
    threshold: float
    decline_rate: float


class Baseline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    created_utc: str
    model_version: str
    n: int
    feature_types: dict[str, FeatureKind]
    numeric: dict[str, NumericProfile]
    categorical: dict[str, CategoricalProfile]
    score: ScoreProfile
    performance: dict[str, float] | None = None

    @property
    def feature_names(self) -> list[str]:
        return list(self.feature_types)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(mode="json"), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Baseline:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def infer_feature_types(frame: pd.DataFrame, columns: list[str]) -> dict[str, FeatureKind]:
    types: dict[str, FeatureKind] = {}
    for c in columns:
        series = frame[c]
        if pd.api.types.is_bool_dtype(series):
            types[c] = "categorical"
        elif pd.api.types.is_numeric_dtype(series):
            types[c] = "numeric"
        else:
            coerced = pd.to_numeric(series, errors="coerce")
            non_null = series.notna().sum()
            types[c] = (
                "numeric" if non_null and coerced.notna().sum() == non_null else "categorical"
            )
    return types


def numeric_profile(
    name: str, series: pd.Series, *, n_bins: int, max_sample: int, seed: int
) -> NumericProfile:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        raise ValueError(f"{name}: no numeric values to profile")
    edges = quantile_edges(finite, n_bins)
    rng = np.random.default_rng(seed)
    sample = finite if len(finite) <= max_sample else rng.choice(finite, max_sample, replace=False)
    qs = {
        f"p{int(q * 100):02d}": float(np.quantile(finite, q))
        for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
    }
    return NumericProfile(
        name=name,
        n=len(values),
        missing_rate=float(1 - len(finite) / len(values)),
        mean=float(finite.mean()),
        std=float(finite.std(ddof=0)),
        min=float(finite.min()),
        max=float(finite.max()),
        quantiles=qs,
        bin_edges=edges,
        bin_proportions=[float(p) for p in bin_proportions(finite, edges)],
        reference_sample=sorted(float(v) for v in sample),
    )


def categorical_profile(name: str, series: pd.Series) -> CategoricalProfile:
    non_null = series.dropna().astype(str)
    if len(non_null) == 0:
        raise ValueError(f"{name}: no values to profile")
    counts = non_null.value_counts()
    return CategoricalProfile(
        name=name,
        n=len(series),
        missing_rate=float(1 - len(non_null) / len(series)),
        categories={str(k): float(v / len(non_null)) for k, v in sorted(counts.items())},
    )


def score_profile(scores: pd.Series, threshold: float) -> ScoreProfile:
    values = pd.to_numeric(scores, errors="coerce").to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("no scores to profile")
    return ScoreProfile(
        n=len(values),
        mean=float(values.mean()),
        bin_proportions=[float(p) for p in bin_proportions(values, SCORE_EDGES)],
        threshold=float(threshold),
        decline_rate=float((values >= threshold).mean()),
    )


def build_baseline(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    threshold: float = 0.5,
    labels: pd.Series | None = None,
    model_version: str = "unknown",
    feature_types: dict[str, FeatureKind] | None = None,
    n_bins: int = 10,
    max_sample: int = 2000,
    seed: int = 0,
) -> Baseline:
    if len(frame) == 0:
        raise ValueError("cannot build a baseline from an empty frame")
    types = feature_types or infer_feature_types(frame, feature_columns)
    numeric = {
        c: numeric_profile(c, frame[c], n_bins=n_bins, max_sample=max_sample, seed=seed)
        for c, kind in types.items()
        if kind == "numeric"
    }
    categorical = {
        c: categorical_profile(c, frame[c]) for c, kind in types.items() if kind == "categorical"
    }
    performance: dict[str, float] | None = None
    if labels is not None:
        mask = labels.notna().to_numpy()
        if mask.sum() >= 2:
            y = labels[mask].to_numpy(dtype=np.int64)
            s = frame.loc[mask, "score"].to_numpy(dtype=np.float64)
            performance = summarize(y, s, threshold)
    return Baseline(
        created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        model_version=model_version,
        n=len(frame),
        feature_types=types,
        numeric=numeric,
        categorical=categorical,
        score=score_profile(frame["score"], threshold),
        performance=performance,
    )
