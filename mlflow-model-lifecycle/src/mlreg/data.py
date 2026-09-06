"""Loading, fingerprinting and splitting the contract-conformant dataset."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DataSchema, SplitConfig
from .schema import enforce


@dataclass
class Dataset:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    protected_train: pd.DataFrame
    protected_test: pd.DataFrame
    ids_test: pd.Series
    fingerprint: str
    n_total: int

    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_test(self) -> int:
        return len(self.X_test)


def load_frame(path: str | Path, schema: DataSchema) -> pd.DataFrame:
    """Read the CSV with contract dtypes (ids and categoricals as strings) and enforce it."""
    dtypes: dict[str, str] = {schema.id_column: "str"}
    for name in schema.categorical_features + schema.protected_columns:
        dtypes[name] = "str"
    raw = pd.read_csv(path, dtype=dtypes)
    return enforce(raw, schema)


def data_fingerprint(df: pd.DataFrame) -> str:
    """SHA-256 over a canonical serialisation: sorted columns, no index, fixed float format."""
    canon = df.reindex(sorted(df.columns), axis=1)
    payload = canon.to_csv(index=False, lineterminator="\n", float_format="%.10g").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_split(df: pd.DataFrame, schema: DataSchema, split: SplitConfig) -> Dataset:
    y = df[schema.target].to_numpy(dtype=np.int64)
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=split.test_size,
        random_state=split.seed,
        stratify=y if split.stratify else None,
    )
    features = schema.feature_columns
    protected = schema.protected_columns
    tr = df.iloc[train_idx]
    te = df.iloc[test_idx]
    return Dataset(
        X_train=tr[features].reset_index(drop=True),
        X_test=te[features].reset_index(drop=True),
        y_train=y[train_idx],
        y_test=y[test_idx],
        protected_train=tr[protected].reset_index(drop=True),
        protected_test=te[protected].reset_index(drop=True),
        ids_test=te[schema.id_column].reset_index(drop=True),
        fingerprint=data_fingerprint(df),
        n_total=len(df),
    )
