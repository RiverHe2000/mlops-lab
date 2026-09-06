"""The deployable artefact: an MLflow pyfunc wrapping the sklearn pipeline.

`predict` returns a frame with the PD, the accept/decline decision and reason codes, so the
same artefact serves `mlflow models serve`, batch scoring and the promotion gate. The
threshold is a *signature parameter* — callers can override it per request without
re-packaging the model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

from .config import DataSchema
from .models import predict_pd, reason_codes
from .schema import coerce_features

OUTPUT_COLUMNS = ["pd", "decision", "reason_codes"]


class PDScorer(mlflow.pyfunc.PythonModel):  # type: ignore[misc]
    def __init__(self, pipeline: Any, schema: DataSchema, threshold: float, top_k: int = 3) -> None:
        self.pipeline = pipeline
        self.schema_json = schema.model_dump(mode="json")
        self.threshold = float(threshold)
        self.top_k = int(top_k)

    @property
    def schema(self) -> DataSchema:
        return DataSchema.model_validate(self.schema_json)

    def score(self, X: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
        t = self.threshold if threshold is None else float(threshold)
        if not 0.0 < t < 1.0:
            raise ValueError("threshold must be in (0, 1)")
        feats = coerce_features(X, self.schema)
        pd_ = predict_pd(self.pipeline, feats)
        codes = reason_codes(self.pipeline, feats, top_k=self.top_k)
        return pd.DataFrame(
            {
                "pd": pd_,
                "decision": np.where(pd_ >= t, "decline", "approve"),
                "reason_codes": ["|".join(c) for c in codes],
            }
        )

    def predict(
        self, context: Any, model_input: pd.DataFrame, params: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        del context
        threshold = None if params is None else params.get("threshold")
        return self.score(
            pd.DataFrame(model_input), None if threshold is None else float(threshold)
        )


def build_signature(scorer: PDScorer, example: pd.DataFrame) -> Any:
    feats = coerce_features(example, scorer.schema)
    return infer_signature(feats, scorer.score(feats), params={"threshold": scorer.threshold})


def log_pd_model(scorer: PDScorer, example: pd.DataFrame, *, name: str = "model") -> Any:
    """Log the pyfunc with signature + input example; returns MLflow's ModelInfo."""
    feats = coerce_features(example, scorer.schema)
    return mlflow.pyfunc.log_model(
        name=name,
        python_model=scorer,
        signature=build_signature(scorer, feats),
        input_example=feats.head(5),
        pip_requirements=_pinned_requirements(),
        code_paths=[str(Path(__file__).resolve().parent)],
    )


def _pinned_requirements() -> list[str]:
    import sklearn

    return [
        f"mlflow=={mlflow.__version__}",
        f"scikit-learn=={sklearn.__version__}",
        f"pandas=={pd.__version__}",
        f"numpy=={np.__version__}",
        "pydantic>=2.8",
        "pyyaml>=6.0",
    ]


def load_scorer(model_uri: str) -> Any:
    return mlflow.pyfunc.load_model(model_uri)


def score_with(model: Any, X: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    params = None if threshold is None else {"threshold": float(threshold)}
    out = model.predict(X, params=params)
    return pd.DataFrame(out)
