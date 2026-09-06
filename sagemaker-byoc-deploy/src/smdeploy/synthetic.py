"""Deterministic synthetic credit applications for examples, tests and the local simulator.

Not a real dataset: a small logistic data-generating process with numeric and categorical
features, a few missing values (so the imputers are exercised) and a ~25 % default rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "default"
ID_COLUMN = "application_id"
NUMERIC = ["age", "income", "loan_amount", "term_months", "credit_utilisation", "num_delinquencies"]
CATEGORICAL = ["employment_status", "housing", "purpose", "region"]
FEATURES = NUMERIC + CATEGORICAL


def make_credit_frame(n: int = 600, *, seed: int = 7, missing_rate: float = 0.02) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 70, n)
    income = np.round(rng.lognormal(mean=10.9, sigma=0.45, size=n), 2)
    loan = np.round(rng.lognormal(mean=9.3, sigma=0.6, size=n), 2)
    term = rng.choice([12, 24, 36, 48, 60], size=n, p=[0.15, 0.3, 0.3, 0.15, 0.1])
    util = np.clip(rng.beta(2, 5, n), 0, 1)
    delinq = rng.poisson(0.4, n)
    employment = rng.choice(
        ["employed", "self_employed", "unemployed", "retired"], n, p=[0.6, 0.2, 0.1, 0.1]
    )
    housing = rng.choice(["own", "rent", "mortgage"], n, p=[0.35, 0.4, 0.25])
    purpose = rng.choice(["car", "home_improvement", "debt_consolidation", "education", "other"], n)
    region = rng.choice(["NSW", "VIC", "QLD", "WA", "SA"], n, p=[0.32, 0.26, 0.2, 0.12, 0.1])

    logit = (
        -1.6
        + 3.6 * (util - 0.3)
        + 0.9 * delinq
        + 0.6 * (np.log(loan) - np.log(income) + 1.6)
        - 0.02 * (age - 40)
        + np.where(employment == "unemployed", 1.2, 0.0)
        + np.where(employment == "self_employed", 0.3, 0.0)
        + np.where(housing == "rent", 0.5, 0.0)
        + np.where(purpose == "debt_consolidation", 0.5, 0.0)
        + rng.normal(0, 0.35, n)
    )
    default = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(int)

    frame = pd.DataFrame(
        {
            ID_COLUMN: [f"A{i:05d}" for i in range(1, n + 1)],
            "age": age,
            "income": income,
            "loan_amount": loan,
            "term_months": term,
            "credit_utilisation": np.round(util, 4),
            "num_delinquencies": delinq,
            "employment_status": employment,
            "housing": housing,
            "purpose": purpose,
            "region": region,
            TARGET: default,
        }
    )
    if missing_rate > 0:
        for col in ("income", "employment_status"):
            mask = rng.uniform(0, 1, n) < missing_rate
            frame.loc[mask, col] = np.nan
    return frame
