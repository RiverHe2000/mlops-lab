from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import ks_2samp
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from mlreg import metrics as M


def _data(n: int = 500, ties: bool = False, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    p = np.clip(rng.normal(0.3 + 0.3 * y, 0.25), 0, 1)
    if ties:
        p = np.round(p, 1)
    return y, p


@pytest.mark.parametrize("ties", [False, True])
def test_auc_matches_sklearn(ties: bool) -> None:
    y, p = _data(ties=ties)
    assert M.roc_auc(y, p) == pytest.approx(roc_auc_score(y, p), abs=1e-12)


def test_auc_extremes() -> None:
    y = np.array([0, 0, 1, 1])
    assert M.roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert M.roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert M.roc_auc(y, np.full(4, 0.5)) == 0.5
    assert math.isnan(M.roc_auc(np.ones(3, dtype=int), np.array([0.1, 0.2, 0.3])))
    assert M.gini(0.75) == pytest.approx(0.5)


def test_ks_hand_computed_and_matches_scipy() -> None:
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.1])
    assert M.ks_statistic(y, p) == pytest.approx(2 / 3)
    assert M.ks_statistic(np.array([0, 1, 0, 1]), np.full(4, 0.5)) == 0.0
    assert math.isnan(M.ks_statistic(np.zeros(3, dtype=int), np.array([0.1, 0.2, 0.3])))
    yr, pr = _data(ties=True)
    assert M.ks_statistic(yr, pr) == pytest.approx(ks_2samp(pr[yr == 1], pr[yr == 0]).statistic)


def test_brier_and_log_loss_match_sklearn() -> None:
    y, p = _data()
    assert M.brier(y, p) == pytest.approx(brier_score_loss(y, p))
    interior = np.clip(p, 1e-6, 1 - 1e-6)  # sklearn clips exact 0/1 with a different eps
    assert M.log_loss(y, interior) == pytest.approx(log_loss(y, interior))


def test_calibration_table_and_ece() -> None:
    p = np.repeat([0.1, 0.5, 0.9], 100)
    y = np.concatenate(
        [
            np.r_[np.ones(10), np.zeros(90)],
            np.r_[np.ones(50), np.zeros(50)],
            np.r_[np.ones(90), np.zeros(10)],
        ]
    )
    y = y.astype(int)
    table = M.calibration_table(y, p)
    assert table["n"].sum() == 300
    assert table.loc[table["n"] > 0, "mean_pred"].tolist() == pytest.approx([0.1, 0.5, 0.9])
    assert M.expected_calibration_error(y, p) == pytest.approx(0.0, abs=1e-12)
    y_bad = y.copy()
    y_bad[200:] = 0  # the 0.9 bucket now has a 0 % default rate
    assert M.expected_calibration_error(y_bad, p) == pytest.approx(0.3)
    q = M.calibration_table(y, p, n_bins=3, strategy="quantile")
    assert q["n"].sum() == 300
    with pytest.raises(ValueError, match="strategy"):
        M.calibration_table(y, p, strategy="bogus")


def test_decile_lift_on_perfect_ranking() -> None:
    p = np.linspace(0, 1, 100)
    y = (p > 0.7).astype(int)  # 30 bads, all at the top
    table = M.decile_lift(y, p)
    assert len(table) == 10
    assert table["bad_rate"].iloc[:3].tolist() == [1.0, 1.0, 1.0]
    assert table["lift"].iloc[0] == pytest.approx(1 / 0.3)
    assert table["cum_bad_capture"].iloc[2] == pytest.approx(1.0)
    assert table["cum_bad_capture"].is_monotonic_increasing
    assert table["cum_bad_capture"].iloc[-1] == pytest.approx(1.0)


def test_confusion_and_cost_hand_computed() -> None:
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1])
    c = M.confusion_at(y, p, 0.5)
    assert (c["tp"], c["fn"], c["fp"], c["tn"]) == (1, 1, 1, 1)
    assert c["precision"] == c["recall"] == c["specificity"] == c["approval_rate"] == 0.5
    assert c["bad_rate_among_approved"] == 0.5
    assert M.expected_cost(y, p, 0.5) == pytest.approx((5 * 1 + 1 * 1) / 4)
    # decline everyone: no approvals → nan approved-bad-rate, cost = number of goods
    all_decline = M.confusion_at(y, p, 0.05)
    assert math.isnan(all_decline["bad_rate_among_approved"])
    assert M.expected_cost(y, p, 0.05) == pytest.approx(2 / 4)


def test_optimal_threshold_minimises_cost_and_reacts_to_costs() -> None:
    y, p = _data(n=800, seed=3)
    t, c = M.optimal_threshold_by_cost(y, p)
    grid = np.linspace(0.005, 0.995, 199)
    assert c == pytest.approx(min(M.expected_cost(y, p, g) for g in grid))
    assert c == pytest.approx(M.expected_cost(y, p, t))
    t_symmetric, _ = M.optimal_threshold_by_cost(y, p, cost_fn=1.0, cost_fp=1.0)
    assert t < t_symmetric  # missing a default is expensive → decline earlier


def test_summarize_keys_and_slices() -> None:
    y, p = _data()
    s = M.summarize(y, p, 0.4)
    assert {
        "n",
        "auc",
        "gini",
        "ks",
        "brier",
        "log_loss",
        "ece",
        "expected_cost",
        "approval_rate",
    } <= set(s)
    assert s["n"] == 500 and s["threshold"] == 0.4
    groups = pd.Series(["a"] * 250 + ["b"] * 250)
    table = M.slice_metrics(y, p, groups, 0.4)
    assert table["group"].tolist() == ["a", "b"]
    assert table["n"].tolist() == [250, 250]
    single = M.slice_metrics(
        np.array([0, 0, 1]), np.array([0.1, 0.2, 0.9]), pd.Series(["g", "g", "h"]), 0.5
    )
    assert math.isnan(single.loc[single["group"] == "g", "auc"].iloc[0])


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="shape"):
        M.roc_auc(np.array([0, 1]), np.array([0.5]))
    with pytest.raises(ValueError, match="empty"):
        M.brier(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="0/1"):
        M.brier(np.array([0, 2]), np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="lie in"):
        M.brier(np.array([0, 1]), np.array([0.1, 1.2]))


@given(
    st.lists(
        st.tuples(st.integers(0, 1), st.floats(0, 1).map(lambda x: round(x, 6))),
        min_size=2,
        max_size=60,
    ).filter(lambda xs: len({a for a, _ in xs}) == 2)
)
@settings(max_examples=60, deadline=None)
def test_metric_properties(pairs: list[tuple[int, float]]) -> None:
    y = np.array([a for a, _ in pairs])
    p = np.array([b for _, b in pairs])
    auc = M.roc_auc(y, p)
    assert 0.0 <= auc <= 1.0
    assert M.roc_auc(y, 1 - p) == pytest.approx(1 - auc, abs=1e-9)
    assert 0.0 <= M.ks_statistic(y, p) <= 1.0
    assert 0.0 <= M.brier(y, p) <= 1.0
    assert 0.0 <= M.expected_calibration_error(y, p) <= 1.0
