"""A synthetic production system with a known ground truth, so the monitor can be evaluated.

A fixed "deployed model" (a logistic scorer with frozen coefficients) scores applications
drawn from a data-generating process we control. Each scenario perturbs one thing:

    none              stationary traffic (measures false alarms)
    covariate_shift   utilisation distribution moves by `magnitude` standard deviations
    prior_shift       the true default intercept moves (base rate changes, features do not)
    concept_drift     the income coefficient in the *truth* changes; features and scores look
                      identical, only realised performance reveals it
    quality_missing   income goes missing with probability `magnitude`
    quality_unseen    a region code the baseline never saw appears with probability `magnitude`
    score_shift       a serving bug scales scores by (1 - magnitude); inputs are unchanged

Labels arrive `label_latency_days` after the decision, as in real credit portfolios.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .schema import CaptureRecord, FeatureValue, LabelRecord, write_jsonl

Scenario = Literal[
    "none",
    "covariate_shift",
    "prior_shift",
    "concept_drift",
    "quality_missing",
    "quality_unseen",
    "score_shift",
]
SCENARIOS: tuple[Scenario, ...] = (
    "none",
    "covariate_shift",
    "prior_shift",
    "concept_drift",
    "quality_missing",
    "quality_unseen",
    "score_shift",
)
NUMERIC = ["x_income", "x_utilisation", "x_age", "x_tenure"]
CATEGORICAL = ["c_region", "c_product"]
FEATURES = NUMERIC + CATEGORICAL
REGIONS = ["NSW", "VIC", "QLD", "WA", "SA"]
PRODUCTS = ["card", "personal_loan", "overdraft"]
TARGET_BY_SCENARIO: dict[str, str] = {
    "none": "",
    "covariate_shift": "x_utilisation",
    "prior_shift": "performance",
    "concept_drift": "performance",
    "quality_missing": "x_income",
    "quality_unseen": "c_region",
    "score_shift": "score",
}

# the deployed model: frozen logistic coefficients on standardised features
MODEL_COEF = {"x_income": -0.9, "x_utilisation": 1.1, "x_age": -0.3, "x_tenure": -0.4}
MODEL_INTERCEPT = -1.0
MODEL_CAT = {
    "c_product": {"card": 0.2, "personal_loan": 0.0, "overdraft": 0.5},
    "c_region": dict.fromkeys(REGIONS, 0.0),
}
MEDIANS = {"x_income": 0.0, "x_utilisation": 0.0, "x_age": 0.0, "x_tenure": 0.0}


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def model_score(frame: pd.DataFrame) -> np.ndarray:
    """What the deployed model does, including its median imputation of missing numerics."""
    z = np.full(len(frame), MODEL_INTERCEPT, dtype=np.float64)
    for name, coef in MODEL_COEF.items():
        values = (
            pd.to_numeric(frame[name], errors="coerce")
            .fillna(MEDIANS[name])
            .to_numpy(dtype=np.float64)
        )
        z += coef * values
    for name, table in MODEL_CAT.items():
        z += frame[name].map(lambda v, t=table: t.get(str(v), 0.0)).to_numpy(dtype=np.float64)
    return _sigmoid(z)


def true_logit(
    frame: pd.DataFrame, *, prior_shift: float = 0.0, concept_shift: float = 0.0
) -> np.ndarray:
    """The truth the model approximates; drift scenarios perturb this, not the scorer."""
    z = np.full(len(frame), -1.05 + prior_shift, dtype=np.float64)
    z += (-0.9 + concept_shift) * frame["x_income_true"].to_numpy(dtype=np.float64)
    z += 1.0 * frame["x_utilisation"].to_numpy(dtype=np.float64)
    z -= 0.25 * frame["x_age"].to_numpy(dtype=np.float64)
    z -= 0.45 * frame["x_tenure"].to_numpy(dtype=np.float64)
    product = {"card": 0.2, "personal_loan": 0.0, "overdraft": 0.6}
    z += frame["c_product"].map(product).to_numpy(dtype=np.float64)
    z += 0.15 * (frame["c_region"] == "QLD").to_numpy(dtype=np.float64)
    return z


@dataclass
class WindowSpec:
    scenario: Scenario
    magnitude: float
    n: int
    seed: int
    start: datetime
    hours: float = 24.0
    model_version: str = "v1"
    threshold: float = 0.5
    label_latency_days: int = 30


def generate_window(spec: WindowSpec) -> tuple[list[CaptureRecord], list[LabelRecord]]:
    rng = np.random.default_rng(spec.seed)
    n = spec.n
    frame = pd.DataFrame(
        {
            "x_income": rng.normal(0, 1, n),
            "x_utilisation": rng.normal(0, 1, n),
            "x_age": rng.normal(0, 1, n),
            "x_tenure": rng.normal(0, 1, n),
            "c_region": rng.choice(REGIONS, n, p=[0.32, 0.26, 0.2, 0.12, 0.1]),
            "c_product": rng.choice(PRODUCTS, n, p=[0.5, 0.35, 0.15]),
        }
    )
    frame["x_income_true"] = frame["x_income"]
    prior_shift = concept_shift = 0.0
    m = spec.magnitude
    if spec.scenario == "covariate_shift":
        frame["x_utilisation"] = frame["x_utilisation"] + m
    elif spec.scenario == "prior_shift":
        prior_shift = m
    elif spec.scenario == "concept_drift":
        concept_shift = m
    elif spec.scenario == "quality_missing":
        frame.loc[rng.uniform(0, 1, n) < m, "x_income"] = np.nan
    elif spec.scenario == "quality_unseen":
        frame.loc[rng.uniform(0, 1, n) < m, "c_region"] = "TAS"

    scores = model_score(frame)
    if spec.scenario == "score_shift":
        scores = np.clip(scores * (1.0 - m), 0.0, 1.0)
    p_true = _sigmoid(true_logit(frame, prior_shift=prior_shift, concept_shift=concept_shift))
    labels = (rng.uniform(0, 1, n) < p_true).astype(int)

    offsets = np.sort(rng.uniform(0, spec.hours * 3600, n))
    captures: list[CaptureRecord] = []
    label_records: list[LabelRecord] = []
    numeric_values = {name: frame[name].to_numpy(dtype=np.float64) for name in NUMERIC}
    categorical_values = {name: frame[name].astype(str).to_numpy() for name in CATEGORICAL}
    for i in range(n):
        ts = spec.start + timedelta(seconds=float(offsets[i]))
        request_id = f"{spec.start:%Y%m%d}-{spec.seed}-{i:06d}"
        features: dict[str, FeatureValue] = {}
        for name in NUMERIC:
            value = float(numeric_values[name][i])
            features[name] = None if math.isnan(value) else value
        for name in CATEGORICAL:
            features[name] = str(categorical_values[name][i])
        score = float(scores[i])
        captures.append(
            CaptureRecord(
                request_id=request_id,
                ts=ts,
                model_version=spec.model_version,
                features=features,
                score=score,
                decision="decline" if score >= spec.threshold else "approve",
            )
        )
        label_records.append(
            LabelRecord(
                request_id=request_id,
                label=int(labels[i]),
                label_ts=ts + timedelta(days=spec.label_latency_days),
            )
        )
    return captures, label_records


@dataclass
class SimulationManifest:
    scenario: str
    magnitude: float
    seed: int
    n_baseline: int
    n_window: int
    windows: list[str]
    baseline_path: str
    labels_path: str
    threshold: float
    target: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def simulate_dataset(
    out_dir: Path,
    *,
    scenario: Scenario,
    magnitude: float,
    windows: int = 3,
    n_baseline: int = 5000,
    n_window: int = 1000,
    seed: int = 0,
    threshold: float = 0.5,
    start: datetime | None = None,
    label_latency_days: int = 30,
) -> SimulationManifest:
    """Clean baseline traffic, then `windows` daily windows under `scenario`; labels for all."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = start or datetime(2026, 1, 1, tzinfo=UTC)
    base_spec = WindowSpec(
        "none",
        0.0,
        n_baseline,
        seed,
        start - timedelta(days=30),
        hours=24 * 30,
        threshold=threshold,
        label_latency_days=label_latency_days,
    )
    base_caps, base_labels = generate_window(base_spec)
    write_jsonl(out_dir / "baseline.jsonl", base_caps)
    all_labels = list(base_labels)
    names: list[str] = []
    for w in range(1, windows + 1):
        spec = WindowSpec(
            scenario,
            magnitude,
            n_window,
            seed * 1000 + w,
            start + timedelta(days=w - 1),
            threshold=threshold,
            label_latency_days=label_latency_days,
        )
        caps, labels = generate_window(spec)
        name = f"window_{w:02d}.jsonl"
        write_jsonl(out_dir / name, caps)
        all_labels.extend(labels)
        names.append(name)
    write_jsonl(out_dir / "labels.jsonl", all_labels)
    manifest = SimulationManifest(
        scenario=scenario,
        magnitude=magnitude,
        seed=seed,
        n_baseline=n_baseline,
        n_window=n_window,
        windows=names,
        baseline_path="baseline.jsonl",
        labels_path="labels.jsonl",
        threshold=threshold,
        target=TARGET_BY_SCENARIO[scenario],
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2), encoding="utf-8"
    )
    return manifest
