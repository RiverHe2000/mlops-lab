from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlreg.config import DataSchema
from mlreg.schema import SchemaError, coerce_features, coerce_frame, enforce, validate_frame

from .conftest import DATA


def test_real_frame_conforms(schema: DataSchema) -> None:
    dtypes = dict.fromkeys(
        [schema.id_column, *schema.categorical_features, *schema.protected_columns], "str"
    )
    raw = pd.read_csv(DATA, dtype=dtypes)
    report = validate_frame(raw, schema)
    assert report.ok
    assert report.n_rows == 1000
    assert report.summary().startswith("OK")


def test_each_violation_kind_detected(frame: pd.DataFrame, schema: DataSchema) -> None:
    df = frame.copy()
    df.loc[0, "duration_months"] = -5.0
    df.loc[1, "age_years"] = 500.0
    df.loc[2, "purpose"] = "yacht"
    df.loc[3, "credit_amount"] = np.nan
    df.loc[4, "application_id"] = df.loc[5, "application_id"]
    df.loc[6, "default"] = 3
    df["extra"] = 1
    df = df.drop(columns=["housing"])
    report = validate_frame(df, schema)
    kinds = {v.kind for v in report.violations}
    assert kinds >= {
        "below_min",
        "above_max",
        "unknown_category",
        "null",
        "duplicate_id",
        "bad_target",
        "unexpected_column",
        "missing_column",
    }
    assert not report.ok
    payload = report.to_dict()
    assert payload["ok"] is False
    assert len(payload["violations"]) == len(report.violations)
    assert "housing: missing_column" in report.summary()
    # non-strict mode tolerates the extra column but nothing else
    lenient = validate_frame(df, schema, strict_columns=False)
    assert "unexpected_column" not in {v.kind for v in lenient.violations}


def test_not_numeric_detected(frame: pd.DataFrame, schema: DataSchema) -> None:
    df = frame.copy()
    df["duration_months"] = df["duration_months"].astype(object)
    df.loc[0, "duration_months"] = "abc"
    report = validate_frame(df, schema, strict_columns=False)
    assert any(v.kind == "not_numeric" and v.column == "duration_months" for v in report.violations)


def test_enforce_raises_and_coerces(frame: pd.DataFrame, schema: DataSchema) -> None:
    bad = frame.copy()
    bad.loc[0, "purpose"] = "yacht"
    with pytest.raises(SchemaError, match="unknown_category"):
        enforce(bad, schema)
    good = frame.copy()
    good["age_years"] = good["age_years"].astype(int)
    out = enforce(good, schema)
    assert out["age_years"].dtype == np.float64
    assert out["default"].dtype == np.int64
    assert out.columns.tolist() == [c.name for c in schema.columns]


def test_enforce_without_target(frame: pd.DataFrame, schema: DataSchema) -> None:
    df = frame.drop(columns=["default"])
    out = enforce(df, schema, require_target=False)
    assert "default" not in out.columns
    with pytest.raises(SchemaError):
        coerce_frame(df, schema)


def test_coerce_features_order_and_missing(frame: pd.DataFrame, schema: DataSchema) -> None:
    shuffled = frame[list(reversed(frame.columns))]
    feats = coerce_features(shuffled, schema)
    assert feats.columns.tolist() == schema.feature_columns
    with pytest.raises(SchemaError, match="missing feature"):
        coerce_features(frame.drop(columns=["purpose"]), schema)
