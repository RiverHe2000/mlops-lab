"""Per-feature and score drift between the baseline profile and a window of captured data.

Numeric features: PSI on the baseline's quantile bins (with a bootstrap CI), two-sample
Kolmogorov-Smirnov against the reference sample, Jensen-Shannon on the bins, and the
Wasserstein-1 distance in units of the baseline standard deviation.
Categorical features: PSI over the baseline categories, a two-sample chi-square test against
the baseline counts, and the share of values in categories the baseline never saw.
Score: PSI / JS on fixed 0.1-wide bins, mean shift and change in the decline rate.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

from .baseline import Baseline, CategoricalProfile, NumericProfile, ScoreProfile
from .stats import (
    benjamini_hochberg,
    bin_proportions,
    bootstrap_psi,
    categorical_proportions,
    js_divergence,
    psi,
)


@dataclass
class FeatureDrift:
    name: str
    kind: str
    n: int
    missing_rate: float
    psi: float
    psi_ci_low: float
    psi_ci_high: float
    test: str
    statistic: float
    p_value: float
    js: float
    p_adjusted: float | None = None
    wasserstein_sd: float | None = None
    unseen_share: float | None = None
    shift_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScoreDrift:
    n: int
    psi: float
    js: float
    mean: float
    mean_change: float
    decline_rate: float
    decline_rate_change: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def numeric_drift(
    profile: NumericProfile, series: pd.Series, *, n_boot: int = 200, seed: int = 0
) -> FeatureDrift:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    finite = values[np.isfinite(values)]
    n = len(values)
    missing_rate = float(1 - len(finite) / n) if n else 1.0
    ref_props = np.asarray(profile.bin_proportions)
    if len(finite) == 0:
        return FeatureDrift(
            profile.name,
            "numeric",
            n,
            missing_rate,
            math.nan,
            math.nan,
            math.nan,
            "ks",
            math.nan,
            math.nan,
            math.nan,
        )
    cur_props = bin_proportions(finite, profile.bin_edges)
    value = psi(ref_props, cur_props)
    lo, hi = bootstrap_psi(finite, profile.bin_edges, ref_props, n_boot=n_boot, seed=seed)
    reference = np.asarray(profile.reference_sample, dtype=np.float64)
    ks = sps.ks_2samp(reference, finite, method="asymp")
    scale = profile.std if profile.std > 0 else 1.0
    w = float(sps.wasserstein_distance(reference, finite)) / scale
    mean_shift = (float(finite.mean()) - profile.mean) / scale
    return FeatureDrift(
        name=profile.name,
        kind="numeric",
        n=n,
        missing_rate=missing_rate,
        psi=value,
        psi_ci_low=lo,
        psi_ci_high=hi,
        test="ks",
        statistic=float(ks.statistic),
        p_value=float(ks.pvalue),
        js=js_divergence(ref_props, cur_props),
        wasserstein_sd=w,
        shift_summary=f"mean {mean_shift:+.2f} sd, W1 {w:.2f} sd",
    )


def categorical_drift(profile: CategoricalProfile, series: pd.Series) -> FeatureDrift:
    n = len(series)
    non_null = series.dropna().astype(str).tolist()
    missing_rate = float(1 - len(non_null) / n) if n else 1.0
    categories = list(profile.categories)
    ref_props = np.array([profile.categories[c] for c in categories], dtype=np.float64)
    if not non_null:
        return FeatureDrift(
            profile.name,
            "categorical",
            n,
            missing_rate,
            math.nan,
            math.nan,
            math.nan,
            "chi2",
            math.nan,
            math.nan,
            math.nan,
            unseen_share=math.nan,
        )
    cur_props, unseen = categorical_proportions(non_null, categories)
    value = psi(ref_props, cur_props)
    # two-sample chi-square on a 2 x k contingency table (baseline sample vs window), so the
    # baseline's own sampling noise is part of the null; unseen mass is reported separately
    n_ref = max(round(profile.n * (1.0 - profile.missing_rate)), 1)
    ref_counts = np.rint(ref_props * n_ref)
    cur_counts = np.rint(cur_props * len(non_null))
    keep = (ref_counts + cur_counts) > 0
    if keep.sum() >= 2 and cur_counts.sum() > 0:
        chi = sps.chi2_contingency(
            np.vstack([ref_counts[keep], cur_counts[keep]]), correction=False
        )
        statistic, p_value = float(chi.statistic), float(chi.pvalue)
    else:
        statistic, p_value = math.nan, math.nan
    deltas = cur_props - ref_props
    top = int(np.argmax(np.abs(deltas))) if len(deltas) else 0
    summary = f"{categories[top]} {deltas[top] * 100:+.1f} pp" if len(deltas) else ""
    if unseen > 0:
        summary += f", unseen {unseen * 100:.1f} %"
    return FeatureDrift(
        name=profile.name,
        kind="categorical",
        n=n,
        missing_rate=missing_rate,
        psi=value,
        psi_ci_low=math.nan,
        psi_ci_high=math.nan,
        test="chi2",
        statistic=statistic,
        p_value=p_value,
        js=js_divergence(ref_props, cur_props),
        unseen_share=unseen,
        shift_summary=summary,
    )


def score_drift(profile: ScoreProfile, scores: pd.Series) -> ScoreDrift:
    values = pd.to_numeric(scores, errors="coerce").to_numpy(dtype=np.float64)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return ScoreDrift(0, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan)
    ref = np.asarray(profile.bin_proportions)
    cur = bin_proportions(finite, profile.bin_edges)
    decline = float((finite >= profile.threshold).mean())
    return ScoreDrift(
        n=len(finite),
        psi=psi(ref, cur),
        js=js_divergence(ref, cur),
        mean=float(finite.mean()),
        mean_change=float(finite.mean()) - profile.mean,
        decline_rate=decline,
        decline_rate_change=decline - profile.decline_rate,
    )


def feature_drifts(
    baseline: Baseline, frame: pd.DataFrame, *, n_boot: int = 200, seed: int = 0
) -> list[FeatureDrift]:
    """All features in baseline order, missing columns reported as fully missing."""
    out: list[FeatureDrift] = []
    for name, kind in baseline.feature_types.items():
        series = (
            frame[name]
            if name in frame.columns
            else pd.Series([np.nan] * len(frame), dtype="float64")
        )
        if kind == "numeric":
            out.append(numeric_drift(baseline.numeric[name], series, n_boot=n_boot, seed=seed))
        else:
            out.append(categorical_drift(baseline.categorical[name], series))
    adjusted = benjamini_hochberg([d.p_value for d in out])
    for d, p_adj in zip(out, adjusted, strict=True):
        d.p_adjusted = None if math.isnan(p_adj) else p_adj
    return out
