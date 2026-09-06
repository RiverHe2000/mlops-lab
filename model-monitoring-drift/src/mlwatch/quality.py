"""Data-quality constraints checked before drift is even interpreted: a missing-rate spike,
an unseen category or a type error is a pipeline incident, not a population change."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .baseline import Baseline
from .config import QualityRules


@dataclass
class QualityIssue:
    feature: str
    kind: str
    value: float
    threshold: float
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def quality_checks(
    baseline: Baseline, frame: pd.DataFrame, rules: QualityRules
) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    n = len(frame)
    if n == 0:
        return [QualityIssue("*", "row_count", 0.0, 1.0, "empty window")]
    if "request_id" in frame.columns:
        dup_share = float(frame["request_id"].duplicated().mean())
        if dup_share > rules.duplicate_share_warn:
            issues.append(
                QualityIssue(
                    "request_id",
                    "duplicate_ids",
                    dup_share,
                    rules.duplicate_share_warn,
                    "duplicated request ids",
                )
            )

    for name, kind in baseline.feature_types.items():
        if name not in frame.columns:
            issues.append(
                QualityIssue(name, "missing_column", 1.0, 0.0, "feature absent from the capture")
            )
            continue
        series = frame[name]
        base_missing = (
            baseline.numeric[name].missing_rate
            if kind == "numeric"
            else baseline.categorical[name].missing_rate
        )
        missing = float(series.isna().mean())
        if missing - base_missing > rules.missing_rate_increase_warn:
            issues.append(
                QualityIssue(
                    name,
                    "missing_rate",
                    missing,
                    base_missing + rules.missing_rate_increase_warn,
                    f"missing {missing:.1%} vs baseline {base_missing:.1%}",
                )
            )
        non_null = series.dropna()
        if len(non_null) == 0:
            continue
        if kind == "numeric":
            coerced = pd.to_numeric(non_null, errors="coerce")
            type_errors = float(coerced.isna().mean())
            if type_errors > rules.type_error_share_critical:
                sample = sorted({str(v) for v in non_null[coerced.isna()].head(3)})
                issues.append(
                    QualityIssue(
                        name,
                        "type_error",
                        type_errors,
                        rules.type_error_share_critical,
                        f"non-numeric values e.g. {sample}",
                    )
                )
            profile = baseline.numeric[name]
            values = coerced.dropna().to_numpy(dtype=np.float64)
            if len(values):
                out_of_range = float(((values < profile.min) | (values > profile.max)).mean())
                if out_of_range > rules.out_of_range_share_warn:
                    issues.append(
                        QualityIssue(
                            name,
                            "out_of_range",
                            out_of_range,
                            rules.out_of_range_share_warn,
                            f"outside baseline [{profile.min:.4g}, {profile.max:.4g}]",
                        )
                    )
        else:
            known = set(baseline.categorical[name].categories)
            as_str = non_null.astype(str)
            unseen_mask = ~as_str.isin(known)
            unseen = float(unseen_mask.mean())
            if unseen > rules.unseen_category_share_warn:
                sample = sorted(set(as_str[unseen_mask].head(5)))
                issues.append(
                    QualityIssue(
                        name,
                        "unseen_category",
                        unseen,
                        rules.unseen_category_share_warn,
                        f"new values {sample}",
                    )
                )
    return issues
