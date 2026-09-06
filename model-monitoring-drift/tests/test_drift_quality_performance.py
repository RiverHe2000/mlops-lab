from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mlwatch.baseline import Baseline
from mlwatch.config import QualityRules
from mlwatch.drift import categorical_drift, feature_drifts, numeric_drift, score_drift
from mlwatch.performance import join_labels, performance_check
from mlwatch.quality import quality_checks

from .conftest import make_window


def test_numeric_drift_null_and_shift(baseline: Baseline) -> None:
    same, _ = make_window("none", 0.0, seed=3)
    d0 = numeric_drift(baseline.numeric["x_utilisation"], same["x_utilisation"], n_boot=50)
    assert d0.kind == "numeric" and d0.n == 1000 and d0.psi < 0.05 and d0.p_value > 0.01
    assert (
        d0.psi_ci_low <= d0.psi <= d0.psi_ci_high
        and d0.wasserstein_sd is not None
        and d0.wasserstein_sd < 0.15
    )
    shifted, _ = make_window("covariate_shift", 1.0, seed=3)
    d1 = numeric_drift(baseline.numeric["x_utilisation"], shifted["x_utilisation"], n_boot=50)
    assert d1.psi > 0.25 and d1.p_value < 1e-6 and d1.js > d0.js
    assert d1.shift_summary.startswith("mean +") and 0.8 < (d1.wasserstein_sd or 0) < 1.2
    empty = numeric_drift(baseline.numeric["x_income"], pd.Series([np.nan] * 5))
    assert math.isnan(empty.psi) and empty.missing_rate == 1.0
    assert set(d1.to_dict()) >= {"name", "psi", "p_value", "shift_summary"}


def test_categorical_drift(baseline: Baseline) -> None:
    same, _ = make_window("none", 0.0, seed=4)
    d0 = categorical_drift(baseline.categorical["c_region"], same["c_region"])
    assert d0.test == "chi2" and d0.psi < 0.05 and d0.p_value > 0.001 and d0.unseen_share == 0.0
    unseen, _ = make_window("quality_unseen", 0.10, seed=4)
    d1 = categorical_drift(baseline.categorical["c_region"], unseen["c_region"])
    assert (
        d1.unseen_share is not None
        and 0.07 < d1.unseen_share < 0.13
        and "unseen" in d1.shift_summary
    )
    skewed = pd.Series(["NSW"] * 900 + ["VIC"] * 100)
    d2 = categorical_drift(baseline.categorical["c_region"], skewed)
    assert d2.psi > 0.25 and d2.p_value < 1e-6 and d2.shift_summary.startswith("NSW +")
    empty = categorical_drift(baseline.categorical["c_region"], pd.Series([None, None]))
    assert math.isnan(empty.psi) and empty.missing_rate == 1.0


def test_score_drift_and_feature_drifts(baseline: Baseline) -> None:
    same, _ = make_window("none", 0.0, seed=5)
    s0 = score_drift(baseline.score, same["score"])
    assert s0.psi < 0.05 and abs(s0.decline_rate_change) < 0.05
    shifted, _ = make_window("score_shift", 0.3, seed=5)
    s1 = score_drift(baseline.score, shifted["score"])
    assert s1.psi > 0.25 and s1.decline_rate_change < -0.05 and s1.mean_change < 0
    assert math.isnan(score_drift(baseline.score, pd.Series(dtype="float64")).psi)

    drifts = feature_drifts(baseline, shifted.drop(columns=["x_age"]), n_boot=30)
    assert [d.name for d in drifts] == list(baseline.feature_types)
    by_name = {d.name: d for d in drifts}
    assert by_name["x_age"].missing_rate == 1.0 and by_name["x_age"].p_adjusted is None
    adjusted = [d.p_adjusted for d in drifts if d.p_adjusted is not None]
    raw = [d.p_value for d in drifts if d.p_adjusted is not None]
    assert all(a >= r - 1e-12 for a, r in zip(adjusted, raw, strict=True))


def test_quality_checks(baseline: Baseline) -> None:
    rules = QualityRules()
    clean, _ = make_window("none", 0.0, seed=6)
    assert quality_checks(baseline, clean, rules) == []
    assert quality_checks(baseline, clean.head(0), rules)[0].kind == "row_count"

    broken = clean.copy()
    broken.loc[broken.index[:150], "x_income"] = np.nan
    broken.loc[broken.index[:60], "c_region"] = "TAS"
    broken.loc[broken.index[200:250], "x_age"] = 99.0
    broken["x_tenure"] = broken["x_tenure"].astype(object)
    broken.loc[broken.index[:5], "x_tenure"] = "n/a"
    broken.loc[broken.index[1], "request_id"] = broken.loc[broken.index[0], "request_id"]
    broken.loc[broken.index[2], "request_id"] = broken.loc[broken.index[0], "request_id"]
    broken.loc[broken.index[3], "request_id"] = broken.loc[broken.index[0], "request_id"]
    broken.loc[broken.index[4:15], "request_id"] = broken.loc[broken.index[0], "request_id"]
    issues = {
        (i.feature, i.kind): i
        for i in quality_checks(baseline, broken.drop(columns=["c_product"]), rules)
    }
    assert ("x_income", "missing_rate") in issues and issues[
        ("x_income", "missing_rate")
    ].value == pytest.approx(0.15)
    assert ("c_region", "unseen_category") in issues and "TAS" in issues[
        ("c_region", "unseen_category")
    ].detail
    assert ("x_age", "out_of_range") in issues
    assert ("x_tenure", "type_error") in issues and "n/a" in issues[
        ("x_tenure", "type_error")
    ].detail
    assert ("request_id", "duplicate_ids") in issues
    assert ("c_product", "missing_column") in issues
    assert all(
        set(i.to_dict()) == {"feature", "kind", "value", "threshold", "detail"}
        for i in issues.values()
    )


def test_performance_check(baseline: Baseline) -> None:
    frame, labels = make_window("none", 0.0, seed=8)
    unlabelled = performance_check(baseline, frame, None, min_labels=200)
    assert unlabelled.metrics is None and unlabelled.n_labelled == 0 and not unlabelled.evaluated
    few = performance_check(baseline, frame, labels.head(50), min_labels=200)
    assert few.metrics is None and few.n_labelled == 50 and few.label_coverage == 0.05
    full = performance_check(baseline, frame, labels, min_labels=200)
    assert full.metrics is not None and full.auc_drop is not None and abs(full.auc_drop) < 0.1
    assert full.brier_increase is not None and full.n_labelled == 1000
    one_class = labels.copy()
    one_class["label"] = 1
    assert performance_check(baseline, frame, one_class, min_labels=200).metrics is None
    joined = join_labels(frame, labels.head(10))
    assert joined["label"].notna().sum() == 10 and len(joined) == 1000
    assert join_labels(frame, pd.DataFrame(columns=["request_id", "label"]))["label"].isna().all()
    drifted_frame, drifted_labels = make_window("concept_drift", 1.8, seed=8)
    worse = performance_check(baseline, drifted_frame, drifted_labels, min_labels=200)
    assert worse.auc_drop is not None and worse.auc_drop > 0.05
    assert set(worse.to_dict()) >= {"n_scored", "n_labelled", "metrics", "auc_drop"}


def test_reference_fixture_is_stationary(reference: SimpleNamespace, baseline: Baseline) -> None:
    drifts = feature_drifts(baseline, reference.frame, n_boot=20)
    assert max(d.psi for d in drifts) < 0.01  # a baseline compared with itself
