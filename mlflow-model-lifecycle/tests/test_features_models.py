from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from mlreg.config import DataSchema, ModelConfig
from mlreg.features import build_preprocessor, feature_names
from mlreg.models import (
    build_pipeline,
    global_importance,
    predict_pd,
    reason_codes,
    supports_reason_codes,
)


def test_preprocessor_width_and_names(schema: DataSchema, frame: pd.DataFrame) -> None:
    prep = build_preprocessor(schema)
    X = frame[schema.feature_columns]
    Z = prep.fit_transform(X)
    n_cat = sum(len(schema.column(c).categories or []) for c in schema.categorical_features)
    assert Z.shape == (len(X), len(schema.numeric_features) + n_cat)
    names = feature_names(prep)
    assert len(names) == Z.shape[1]
    assert names[: len(schema.numeric_features)] == schema.numeric_features
    assert {"purpose=car_new", "checking_status=lt_0", "other_installment_plans=bank"} <= set(names)


def test_unknown_category_maps_to_zeros(schema: DataSchema, frame: pd.DataFrame) -> None:
    prep = build_preprocessor(schema).fit(frame[schema.feature_columns])
    row = frame[schema.feature_columns].head(1).copy()
    row["purpose"] = "yacht"
    z = prep.transform(row)[0]
    names = feature_names(prep)
    purpose_cols = [i for i, n in enumerate(names) if n.startswith("purpose=")]
    assert np.all(z[purpose_cols] == 0)


def test_design_matrix_is_stable_across_refits(schema: DataSchema, frame: pd.DataFrame) -> None:
    """Categories come from the contract, so a missing category never changes the columns."""
    full = build_preprocessor(schema).fit(frame[schema.feature_columns])
    subset = frame[frame["purpose"] != "car_new"][schema.feature_columns]
    partial = build_preprocessor(schema).fit(subset)
    assert feature_names(full) == feature_names(partial)


@pytest.mark.parametrize(
    ("kind", "params", "importance_kind"),
    [
        ("logistic_regression", {"C": 1.0, "max_iter": 300}, "coefficient"),
        ("hist_gradient_boosting", {"max_iter": 20}, None),
        ("random_forest", {"n_estimators": 20}, "impurity_importance"),
    ],
)
def test_build_pipeline_kinds(
    kind: str,
    params: dict[str, object],
    importance_kind: str | None,
    schema: DataSchema,
    frame: pd.DataFrame,
) -> None:
    pipe = build_pipeline(schema, ModelConfig(kind=kind, params=params), seed=0)
    X = frame[schema.feature_columns].head(300)
    y = frame["default"].to_numpy()[:300]
    pipe.fit(X, y)
    p = predict_pd(pipe, X)
    assert p.shape == (300,)
    assert np.all((p >= 0) & (p <= 1))
    imp = global_importance(pipe)
    if importance_kind is None:
        assert imp.empty
    else:
        assert len(imp) == len(feature_names(pipe.named_steps["prep"]))
        assert imp["kind"].iloc[0] == importance_kind
        assert imp["value"].abs().is_monotonic_decreasing
    assert supports_reason_codes(pipe) == (kind == "logistic_regression")


def test_reason_codes_are_exact_logit_decomposition(
    schema: DataSchema, frame: pd.DataFrame
) -> None:
    cfg = ModelConfig(kind="logistic_regression", params={"C": 1.0, "max_iter": 500})
    pipe = build_pipeline(schema, cfg, seed=0)
    X = frame[schema.feature_columns]
    pipe.fit(X, frame["default"].to_numpy())
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    head = X.head(5)
    contrib = np.asarray(prep.transform(head)) * clf.coef_.reshape(-1)
    logit = contrib.sum(axis=1) + clf.intercept_[0]
    np.testing.assert_allclose(1 / (1 + np.exp(-logit)), predict_pd(pipe, head), atol=1e-9)
    codes = reason_codes(pipe, head, top_k=3)
    names = feature_names(prep)
    for row, c in zip(contrib, codes, strict=True):
        expected = [names[j] for j in np.argsort(-row)[:3] if row[j] > 0]
        assert c == expected
        assert len(c) <= 3


def test_reason_codes_empty_for_trees(schema: DataSchema, frame: pd.DataFrame) -> None:
    pipe = build_pipeline(
        schema, ModelConfig(kind="random_forest", params={"n_estimators": 10}), seed=0
    )
    pipe.fit(frame[schema.feature_columns].head(200), frame["default"].to_numpy()[:200])
    assert reason_codes(pipe, frame[schema.feature_columns].head(3)) == [[], [], []]


def test_global_importance_empty_for_unknown_estimator(
    schema: DataSchema, frame: pd.DataFrame
) -> None:
    pipe = Pipeline([("prep", build_preprocessor(schema)), ("clf", DummyClassifier())])
    pipe.fit(frame[schema.feature_columns].head(50), frame["default"].to_numpy()[:50])
    assert global_importance(pipe).empty
    assert not supports_reason_codes(pipe)
