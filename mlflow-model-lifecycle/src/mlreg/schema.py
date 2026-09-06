"""Data-contract validation and coercion.

Validation never mutates the input; `enforce` validates then returns a coerced copy with the
dtypes the rest of the pipeline (and the MLflow signature) expects: float64 for numeric
columns, string for categorical ones, int for the target.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import DataSchema


class SchemaError(ValueError):
    """Raised when a frame violates the data contract."""


@dataclass(frozen=True)
class Violation:
    column: str
    kind: str
    detail: str
    count: int = 1


@dataclass
class ValidationReport:
    n_rows: int
    violations: list[Violation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "ok": self.ok,
            "violations": [asdict(v) for v in self.violations],
        }

    def summary(self) -> str:
        if self.ok:
            return f"OK: {self.n_rows} rows conform to the schema"
        lines = [f"{len(self.violations)} violation(s) in {self.n_rows} rows:"]
        lines += [f"  - {v.column}: {v.kind} ({v.count}) {v.detail}" for v in self.violations]
        return "\n".join(lines)


def validate_frame(
    df: pd.DataFrame,
    schema: DataSchema,
    *,
    strict_columns: bool = True,
    require_target: bool = True,
) -> ValidationReport:
    """Check presence, nulls, numeric range, categorical domain, target domain, id uniqueness."""
    report = ValidationReport(n_rows=len(df))
    expected = [c.name for c in schema.columns if require_target or c.role != "target"]
    missing = [c for c in expected if c not in df.columns]
    for name in missing:
        report.violations.append(Violation(name, "missing_column", "column absent from frame"))
    if strict_columns:
        known = {c.name for c in schema.columns}
        for name in df.columns:
            if name not in known:
                report.violations.append(Violation(str(name), "unexpected_column", "not in schema"))

    for col in schema.columns:
        if col.name in missing or col.name not in df.columns:
            continue
        series = df[col.name]
        n_null = int(series.isna().sum())
        if n_null:
            report.violations.append(Violation(col.name, "null", "missing values", n_null))
        non_null = series.dropna()
        if col.kind == "numeric":
            parsed = pd.to_numeric(non_null, errors="coerce")
            bad = int(parsed.isna().sum())
            if bad:
                report.violations.append(
                    Violation(col.name, "not_numeric", "non-numeric values", bad)
                )
            parsed = parsed.dropna()
            if col.min is not None:
                n = int((parsed < col.min).sum())
                if n:
                    report.violations.append(Violation(col.name, "below_min", f"< {col.min}", n))
            if col.max is not None:
                n = int((parsed > col.max).sum())
                if n:
                    report.violations.append(Violation(col.name, "above_max", f"> {col.max}", n))
            if col.role == "target":
                n = int((~parsed.isin([0, 1])).sum())
                if n:
                    report.violations.append(
                        Violation(col.name, "bad_target", "target must be 0/1", n)
                    )
        elif col.categories is not None:
            values = non_null.astype(str)
            unknown = values[~values.isin(col.categories)]
            if len(unknown):
                sample = sorted(set(unknown.tolist()))[:5]
                report.violations.append(
                    Violation(
                        col.name,
                        "unknown_category",
                        f"not in contract: {sample}",
                        len(unknown),
                    )
                )
        if col.role == "id":
            dup = int(series.duplicated().sum())
            if dup:
                report.violations.append(
                    Violation(col.name, "duplicate_id", "ids must be unique", dup)
                )
    return report


def coerce_frame(
    df: pd.DataFrame, schema: DataSchema, *, require_target: bool = True
) -> pd.DataFrame:
    """Return a copy with contract dtypes; assumes `validate_frame` passed."""
    out = pd.DataFrame(index=df.index)
    for col in schema.columns:
        if col.name not in df.columns:
            if col.role == "target" and not require_target:
                continue
            raise SchemaError(f"missing column {col.name!r}")
        if col.role == "target":
            out[col.name] = pd.to_numeric(df[col.name]).astype(np.int64)
        elif col.kind == "numeric":
            out[col.name] = pd.to_numeric(df[col.name]).astype(np.float64)
        else:
            out[col.name] = df[col.name].astype(str)
    return out


def enforce(df: pd.DataFrame, schema: DataSchema, *, require_target: bool = True) -> pd.DataFrame:
    report = validate_frame(df, schema, strict_columns=False, require_target=require_target)
    if not report.ok:
        raise SchemaError(report.summary())
    return coerce_frame(df, schema, require_target=require_target)


def coerce_features(df: pd.DataFrame, schema: DataSchema) -> pd.DataFrame:
    """Scoring-time coercion of *feature* columns only (target and protected columns optional)."""
    missing = [c for c in schema.feature_columns if c not in df.columns]
    if missing:
        raise SchemaError(f"missing feature columns: {missing}")
    out = pd.DataFrame(index=df.index)
    for name in schema.numeric_features:
        out[name] = pd.to_numeric(df[name]).astype(np.float64)
    for name in schema.categorical_features:
        out[name] = df[name].astype(str)
    return out[schema.feature_columns]
