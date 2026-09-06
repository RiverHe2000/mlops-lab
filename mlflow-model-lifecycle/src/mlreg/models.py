"""Estimator factory, full pipeline and reason codes."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import DataSchema, ModelConfig
from .features import build_preprocessor, feature_names


def build_estimator(cfg: ModelConfig, seed: int) -> Any:
    params = dict(cfg.params)
    if cfg.kind == "logistic_regression":
        return LogisticRegression(random_state=seed, **params)
    if cfg.kind == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(random_state=seed, **params)
    if cfg.kind == "random_forest":
        return RandomForestClassifier(random_state=seed, **params)
    raise ValueError(f"unknown model kind {cfg.kind!r}")  # pragma: no cover - guarded by pydantic


def build_pipeline(schema: DataSchema, cfg: ModelConfig, seed: int) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(schema)), ("clf", build_estimator(cfg, seed))])


def predict_pd(pipeline: Any, X: pd.DataFrame) -> np.ndarray:
    proba = pipeline.predict_proba(X)
    return np.asarray(proba[:, 1], dtype=np.float64)


def supports_reason_codes(pipeline: Any) -> bool:
    return hasattr(pipeline.named_steps["clf"], "coef_")


def reason_codes(pipeline: Any, X: pd.DataFrame, top_k: int = 3) -> list[list[str]]:
    """Top-k features pushing the PD *up* for each row (linear models only).

    Contribution of feature j for a row = coef_j * x_j on the transformed design matrix, so
    the codes are exact decompositions of the logit, not approximations.
    """
    if not supports_reason_codes(pipeline):
        return [[] for _ in range(len(X))]
    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]
    design = np.asarray(prep.transform(X), dtype=np.float64)
    coef = np.asarray(clf.coef_, dtype=np.float64).reshape(-1)
    contrib = design * coef
    names = feature_names(prep)
    out: list[list[str]] = []
    for row in contrib:
        order = np.argsort(-row)
        codes = [names[j] for j in order[:top_k] if row[j] > 0]
        out.append(codes)
    return out


def global_importance(pipeline: Any) -> pd.DataFrame:
    """Coefficients for linear models, impurity importances for forests, else empty."""
    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]
    names = feature_names(prep)
    if hasattr(clf, "coef_"):
        values = np.asarray(clf.coef_, dtype=np.float64).reshape(-1)
        kind = "coefficient"
    elif hasattr(clf, "feature_importances_"):
        values = np.asarray(clf.feature_importances_, dtype=np.float64)
        kind = "impurity_importance"
    else:
        return pd.DataFrame(columns=["feature", "value", "kind"])
    df = pd.DataFrame({"feature": names, "value": values, "kind": kind})
    return df.reindex(df["value"].abs().sort_values(ascending=False).index).reset_index(drop=True)
