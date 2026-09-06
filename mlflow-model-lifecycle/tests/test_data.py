from __future__ import annotations

import pandas as pd
import pytest

from mlreg.config import DataSchema, SplitConfig
from mlreg.data import data_fingerprint, make_split

PINNED_FINGERPRINT = "df35f767300115b57a71d5e974c23c1b679067b2a4c93e21546479de03c14cb9"


def test_load_frame_shape_and_dtypes(frame: pd.DataFrame) -> None:
    assert frame.shape == (1000, 22)
    assert frame["default"].mean() == pytest.approx(0.30)
    assert frame["application_id"].iloc[0] == "0001"  # ids stay strings


def test_fingerprint_is_canonical_and_pinned(frame: pd.DataFrame) -> None:
    fp = data_fingerprint(frame)
    assert fp == PINNED_FINGERPRINT  # the committed snapshot; changes when the CSV changes
    assert data_fingerprint(frame[list(reversed(frame.columns))]) == fp  # column order irrelevant
    changed = frame.copy()
    changed["age_years"] = changed["age_years"] + 1
    assert data_fingerprint(changed) != fp


def test_split_is_stratified_deterministic_and_disjoint(
    frame: pd.DataFrame, schema: DataSchema
) -> None:
    ds = make_split(frame, schema, SplitConfig(test_size=0.25, seed=42))
    assert (ds.n_train, ds.n_test, ds.n_total) == (750, 250, 1000)
    assert abs(ds.y_test.mean() - ds.y_train.mean()) < 0.01
    assert len(set(ds.ids_test)) == 250
    assert ds.X_train.columns.tolist() == schema.feature_columns
    assert ds.protected_test.columns.tolist() == schema.protected_columns
    assert ds.fingerprint == PINNED_FINGERPRINT
    again = make_split(frame, schema, SplitConfig(test_size=0.25, seed=42))
    assert ds.ids_test.tolist() == again.ids_test.tolist()
    other = make_split(frame, schema, SplitConfig(test_size=0.25, seed=7))
    assert ds.ids_test.tolist() != other.ids_test.tolist()


def test_split_without_stratify(frame: pd.DataFrame, schema: DataSchema) -> None:
    ds = make_split(frame, schema, SplitConfig(test_size=0.2, seed=1, stratify=False))
    assert ds.n_test == 200
