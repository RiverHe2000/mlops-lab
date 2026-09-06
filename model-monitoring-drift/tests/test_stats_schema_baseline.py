from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlwatch import stats
from mlwatch.baseline import (
    Baseline,
    build_baseline,
    categorical_profile,
    infer_feature_types,
    numeric_profile,
    score_profile,
)
from mlwatch.metrics import roc_auc, summarize
from mlwatch.schema import (
    CaptureRecord,
    LabelRecord,
    captures_to_frame,
    feature_columns,
    labels_to_frame,
    read_jsonl,
    select_window,
    write_jsonl,
)
from mlwatch.simulate import FEATURES


# --- stats -----------------------------------------------------------------------------------
def test_quantile_edges_and_bins() -> None:
    values = np.arange(1000, dtype=float)
    edges = stats.quantile_edges(values, 10)
    assert len(edges) == 9 and edges == sorted(edges)
    props = stats.bin_proportions(values, edges)
    assert len(props) == 10 and props.sum() == pytest.approx(1.0)
    assert np.allclose(props, 0.1, atol=0.002)
    assert stats.quantile_edges(np.array([]), 10) == []
    assert len(stats.quantile_edges(np.ones(50), 10)) == 1  # ties collapse to one edge
    assert stats.bin_proportions(np.array([]), edges).sum() == 0.0
    assert stats.bin_proportions(np.array([np.nan, 5.0]), edges).sum() == pytest.approx(1.0)


def test_psi_and_js_properties() -> None:
    r = np.array([0.1, 0.2, 0.3, 0.4])
    assert stats.psi(r, r) == 0.0
    c = np.array([0.4, 0.3, 0.2, 0.1])
    assert stats.psi(r, c) == pytest.approx(stats.psi(c, r))
    assert stats.psi(r, c) > 0.25
    assert (
        stats.psi(r, np.array([0.0, 0.0, 0.0, 1.0])) > 1.0
    )  # empty bins are floored, not infinite
    with pytest.raises(ValueError, match="align"):
        stats.psi(r, np.array([0.5, 0.5]))
    assert stats.js_divergence(r, r) == pytest.approx(0.0, abs=1e-9)
    assert stats.js_divergence(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(
        1.0, abs=1e-6
    )
    assert 0.0 < stats.js_divergence(r, c) < 1.0


@given(
    st.lists(st.floats(0.001, 1.0), min_size=2, max_size=12),
    st.lists(st.floats(0.001, 1.0), min_size=2, max_size=12),
)
@settings(max_examples=80, deadline=None)
def test_psi_is_nonnegative(a: list[float], b: list[float]) -> None:
    k = min(len(a), len(b))
    r = np.array(a[:k]) / sum(a[:k])
    c = np.array(b[:k]) / sum(b[:k])
    assert stats.psi(r, c) >= -1e-12
    assert 0.0 <= stats.js_divergence(r, c) <= 1.0


def test_benjamini_hochberg() -> None:
    adjusted = stats.benjamini_hochberg([0.01, 0.04, 0.03, 0.20])
    assert adjusted == pytest.approx([0.04, 0.0533333, 0.0533333, 0.20], abs=1e-6)
    with_nan = stats.benjamini_hochberg([0.01, float("nan"), 0.5])
    assert math.isnan(with_nan[1]) and with_nan[0] == pytest.approx(0.02) and with_nan[2] == 0.5
    assert stats.benjamini_hochberg([]) == []
    assert all(math.isnan(v) for v in stats.benjamini_hochberg([float("nan")]))
    assert max(stats.benjamini_hochberg([0.9, 0.95, 0.99])) <= 1.0


def test_bootstrap_psi_and_categorical_proportions() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    edges = stats.quantile_edges(ref, 10)
    ref_props = stats.bin_proportions(ref, edges)
    cur = rng.normal(0.5, 1, 1000)
    point = stats.psi(ref_props, stats.bin_proportions(cur, edges))
    lo, hi = stats.bootstrap_psi(cur, edges, ref_props, n_boot=100, seed=1)
    assert lo <= point <= hi and lo > 0
    lo2, hi2 = stats.bootstrap_psi(np.array([1.0]), edges, ref_props)
    assert math.isnan(lo2) and math.isnan(hi2)
    props, unseen = stats.categorical_proportions(["a", "a", "b", "z"], ["a", "b"])
    assert props.tolist() == pytest.approx([0.5, 0.25]) and unseen == 0.25
    assert stats.categorical_proportions([], ["a"])[0].tolist() == [0.0]


# --- metrics ---------------------------------------------------------------------------------
def test_metrics_basics() -> None:
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert math.isnan(roc_auc(np.ones(3, dtype=int), np.array([0.1, 0.2, 0.3])))
    s = summarize(y, np.array([0.1, 0.6, 0.7, 0.9]), 0.5)
    assert s["decline_rate"] == 0.75 and s["auc"] == 1.0 and 0 <= s["ece"] <= 1
    with pytest.raises(ValueError):
        summarize(np.array([]), np.array([]), 0.5)


# --- schema ----------------------------------------------------------------------------------
def test_records_roundtrip_and_frames(tmp_path: Path) -> None:
    naive = CaptureRecord(
        request_id="r1", ts=datetime(2026, 1, 1, 12, 0), features={"a": 1.0, "c": "x"}, score=0.3
    )
    assert naive.ts.tzinfo is not None
    records = [
        naive,
        CaptureRecord(
            request_id="r2",
            ts=datetime(2026, 1, 2, tzinfo=UTC),
            features={"a": None, "c": "y"},
            score=0.9,
            decision="decline",
        ),
    ]
    path = tmp_path / "cap.jsonl"
    assert write_jsonl(path, records) == 2
    back = read_jsonl(path, CaptureRecord)
    assert back == records
    frame = captures_to_frame(back)
    assert feature_columns(frame) == ["a", "c"]
    assert str(frame["ts"].dt.tz) == "UTC"
    missing_value = frame["a"].tolist()[1]
    assert missing_value != missing_value  # NaN survives the round trip
    assert captures_to_frame([]).empty
    labels = labels_to_frame(
        [
            LabelRecord(request_id="r1", label=1),
            LabelRecord(request_id="r1", label=0, label_ts=datetime(2026, 2, 1)),
        ]
    )
    assert len(labels) == 1 and labels["label"].iloc[0] == 0  # last one wins
    assert labels_to_frame([]).empty
    start, end = datetime(2026, 1, 1, 6, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC)
    assert [r.request_id for r in select_window(records, start, end)] == ["r1"]
    assert len(select_window(records, None, None)) == 2
    (tmp_path / "bad.jsonl").write_text('{"request_id": "x"}\n')
    with pytest.raises(ValueError, match="jsonl:1"):
        read_jsonl(tmp_path / "bad.jsonl", CaptureRecord)
    with pytest.raises(ValueError):
        CaptureRecord(request_id="r", ts=datetime.now(UTC), features={}, score=1.5)


# --- baseline --------------------------------------------------------------------------------
def test_feature_type_inference() -> None:
    frame = pd.DataFrame(
        {"n": [1, 2, 3], "s": ["1", "2", None], "c": ["a", "b", "a"], "b": [True, False, True]}
    )
    assert infer_feature_types(frame, ["n", "s", "c", "b"]) == {
        "n": "numeric",
        "s": "numeric",
        "c": "categorical",
        "b": "categorical",
    }


def test_profiles() -> None:
    rng = np.random.default_rng(0)
    series = pd.Series(np.r_[rng.normal(0, 1, 5000), [np.nan] * 50])
    prof = numeric_profile("x", series, n_bins=10, max_sample=500, seed=0)
    assert prof.n == 5050 and prof.missing_rate == pytest.approx(50 / 5050)
    assert len(prof.reference_sample) == 500 and prof.reference_sample == sorted(
        prof.reference_sample
    )
    assert len(prof.bin_edges) == 9 and sum(prof.bin_proportions) == pytest.approx(1.0)
    assert prof.quantiles["p50"] == pytest.approx(0.0, abs=0.1)
    with pytest.raises(ValueError, match="no numeric"):
        numeric_profile("x", pd.Series([np.nan, np.nan]), n_bins=10, max_sample=10, seed=0)
    cat = categorical_profile("c", pd.Series(["a", "a", "b", None]))
    assert (
        cat.categories == {"a": pytest.approx(2 / 3), "b": pytest.approx(1 / 3)}
        and cat.missing_rate == 0.25
    )
    with pytest.raises(ValueError):
        categorical_profile("c", pd.Series([None, None]))
    sp = score_profile(pd.Series([0.1, 0.6, 0.7, np.nan]), 0.5)
    assert sp.n == 3 and sp.decline_rate == pytest.approx(2 / 3) and len(sp.bin_proportions) == 10
    with pytest.raises(ValueError):
        score_profile(pd.Series([np.nan]), 0.5)


def test_build_baseline_roundtrip(
    baseline: Baseline, reference: SimpleNamespace, tmp_path: Path
) -> None:
    assert baseline.n == 3000 and baseline.model_version == "v1"
    assert set(baseline.feature_types) == set(FEATURES)
    assert set(baseline.numeric) == {"x_income", "x_utilisation", "x_age", "x_tenure"}
    assert set(baseline.categorical) == {"c_region", "c_product"}
    assert baseline.performance is not None and baseline.performance["auc"] > 0.7
    baseline.save(tmp_path / "b.json")
    loaded = Baseline.load(tmp_path / "b.json")
    assert loaded == baseline
    assert (
        json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))["score"]["threshold"] == 0.5
    )
    without_labels = build_baseline(reference.frame, feature_columns=FEATURES, threshold=0.5)
    assert without_labels.performance is None
    with pytest.raises(ValueError, match="empty"):
        build_baseline(reference.frame.head(0), feature_columns=FEATURES)
    forced = build_baseline(
        reference.frame, feature_columns=["x_age"], feature_types={"x_age": "categorical"}
    )
    assert "x_age" in forced.categorical
    later = reference.frame.copy()
    later["ts"] = later["ts"] + timedelta(days=1)
    assert build_baseline(later, feature_columns=FEATURES).n == 3000
