"""Scoring logic shared by the hosting server, batch transform and the local simulator."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from .model_io import LoadedModel, load_model

log = logging.getLogger("smdeploy.inference")


class InputError(ValueError):
    """Client-side problem with the feature frame (HTTP 400)."""


class Predictor:
    def __init__(self, loaded: LoadedModel, *, threshold: float | None = None) -> None:
        self.loaded = loaded
        self.meta = loaded.metadata
        self.threshold = float(self.meta.threshold if threshold is None else threshold)
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be in (0, 1)")

    @property
    def feature_columns(self) -> list[str]:
        return list(self.meta.feature_columns)

    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Align the request frame with the training contract; extra columns are dropped."""
        missing = [c for c in self.feature_columns if c not in frame.columns]
        if missing:
            raise InputError(f"missing feature columns: {missing}")
        if len(frame) == 0:
            raise InputError("no rows to score")
        out = pd.DataFrame(index=frame.index)
        for col in self.meta.numeric_columns:
            out[col] = pd.to_numeric(frame[col], errors="coerce").astype(np.float64)
        for col in self.meta.categorical_columns:
            values = frame[col]
            out[col] = values.where(values.isna(), values.astype(str))
        return out[self.feature_columns]

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = self.prepare(frame)
        proba = np.asarray(self.loaded.pipeline.predict_proba(X))[:, 1]
        return pd.DataFrame(
            {
                "probability": proba.astype(np.float64),
                "label": (proba >= self.threshold).astype(np.int64),
            }
        )


class ModelHandler:
    """Lazy, thread-safe model loader; `/ping` reports 503 until `load()` has succeeded."""

    def __init__(self, model_dir: Path, *, threshold: float | None = None) -> None:
        self.model_dir = Path(model_dir)
        self.threshold = threshold
        self._predictor: Predictor | None = None
        self._error: str | None = None
        self._lock = threading.Lock()

    def load(self) -> Predictor:
        with self._lock:
            if self._predictor is None:
                try:
                    loaded = load_model(self.model_dir)
                    self._predictor = Predictor(loaded, threshold=self.threshold)
                    self._error = None
                    log.info(
                        "model loaded",
                        extra={
                            "model_dir": str(self.model_dir),
                            "sha256": loaded.metadata.model_sha256[:12],
                        },
                    )
                except Exception as exc:
                    self._error = f"{type(exc).__name__}: {exc}"
                    log.error("model load failed: %s", self._error)
                    raise
            return self._predictor

    @property
    def ready(self) -> bool:
        return self._predictor is not None

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def predictor(self) -> Predictor:
        if self._predictor is None:
            raise RuntimeError("model not loaded")
        return self._predictor
