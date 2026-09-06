from __future__ import annotations

import os
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlflow
import pandas as pd
import pytest

from mlreg import tracking as T
from mlreg.config import (
    CVConfig,
    DataSchema,
    ModelConfig,
    SplitConfig,
    TrackingConfig,
    TrainingConfig,
)
from mlreg.data import load_frame
from mlreg.models import build_pipeline
from mlreg.pyfunc import PDScorer, log_pd_model
from mlreg.train import TrainResult, train

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "german_credit.csv"
SCHEMA = ROOT / "data" / "schema.yaml"

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
warnings.filterwarnings("ignore")


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLREG_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLREG_ARTIFACT_LOCATION", raising=False)


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def schema() -> DataSchema:
    return DataSchema.from_yaml(SCHEMA)


@pytest.fixture(scope="session")
def frame(schema: DataSchema) -> pd.DataFrame:
    return load_frame(DATA, schema)


def tracking_for(tmp: Path, *, registered_model: str = "credit-pd-test") -> TrackingConfig:
    return TrackingConfig(
        tracking_uri=f"sqlite:///{(tmp / 'mlflow.db').as_posix()}",
        artifact_location=(tmp / "artifacts").as_posix(),
        experiment="test-exp",
        registered_model=registered_model,
    )


def make_config(
    tmp: Path,
    *,
    name: str = "t_logreg",
    kind: str = "logistic_regression",
    params: dict[str, Any] | None = None,
    folds: int = 3,
    threshold: float | str = "cost_optimal",
    registered_model: str = "credit-pd-test",
) -> TrainingConfig:
    if params is None:
        params = {"C": 1.0, "max_iter": 500} if kind == "logistic_regression" else {}
    return TrainingConfig(
        name=name,
        data_path=DATA,
        schema_path=SCHEMA,
        split=SplitConfig(test_size=0.25, seed=42),
        cv=CVConfig(folds=folds, seed=1),
        model=ModelConfig(kind=kind, params=params),
        decision_threshold=threshold,
        tracking=tracking_for(tmp, registered_model=registered_model),
        tags={"owner": "tests"},
    )


def log_quick_model(
    schema: DataSchema, frame: pd.DataFrame, tracking: TrackingConfig, *, c: float = 1.0
) -> str:
    """Fit a logistic pipeline on the whole frame and log it as a pyfunc; returns the model URI."""
    T.configure(tracking)
    cfg = ModelConfig(kind="logistic_regression", params={"C": c, "max_iter": 500})
    pipe = build_pipeline(schema, cfg, seed=0)
    X = frame[schema.feature_columns]
    pipe.fit(X, frame[schema.target].to_numpy())
    with mlflow.start_run(run_name=f"quick-C{c}"):
        info = log_pd_model(PDScorer(pipe, schema, 0.3), X.head(10))
    return str(info.model_uri)


@pytest.fixture(scope="session")
def trained(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """One tracked training run shared by read-only tests (≈ 3 s)."""
    tmp = tmp_path_factory.mktemp("store")
    cfg = make_config(tmp)
    result: TrainResult = train(cfg, n_boot=50)
    return SimpleNamespace(cfg=cfg, result=result, tmp=tmp)


@pytest.fixture
def use_trained_store(trained: SimpleNamespace) -> SimpleNamespace:
    T.configure(trained.cfg.tracking)
    return trained
