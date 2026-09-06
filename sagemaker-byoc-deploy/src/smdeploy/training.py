"""The `train` entrypoint: SageMaker training container contract for a tabular classifier.

Contract obligations implemented here and covered by tests:
  * read hyperparameters and channels from the mounted tree (or `SM_*` overrides);
  * emit metrics as `validation:<name>=<value>` lines so `MetricDefinitions` regexes capture
    them into CloudWatch and the training-job description;
  * write the model to /opt/ml/model (becomes model.tar.gz), extra outputs to /opt/ml/output/data;
  * on any failure write the reason to /opt/ml/output/failure and exit non-zero;
  * honour /opt/ml/checkpoints for managed spot training.
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .hyperparams import TrainingHyperparameters
from .model_io import ModelMetadata, save_model
from .paths import SageMakerPaths

log = logging.getLogger("smdeploy.training")

METRIC_PREFIX = "validation:"
METRIC_NAMES = ("auc", "log_loss", "brier", "accuracy", "precision", "recall")


class TrainingDataError(ValueError):
    """The channels do not contain what the hyperparameters promise."""


@dataclass
class TrainingReport:
    metrics: dict[str, float]
    metrics_on: str
    n_train: int
    n_validation: int
    model_dir: Path
    metadata: ModelMetadata


def load_channel_frame(paths: SageMakerPaths, name: str, *, required: bool) -> pd.DataFrame | None:
    files = paths.channel_files(name)
    if not files:
        if required:
            raise TrainingDataError(
                f"channel {name!r} has no CSV files under {paths.channel(name)} "
                "(check InputDataConfig and the S3 prefix)"
            )
        return None
    frames = [pd.read_csv(f) for f in files]
    frame = pd.concat(frames, ignore_index=True)
    log.info("channel %s: %d file(s), %d rows", name, len(files), len(frame))
    return frame


def infer_columns(
    frame: pd.DataFrame, target: str, id_columns: list[str]
) -> tuple[list[str], list[str]]:
    exclude = {target, *id_columns}
    numeric = [
        c for c in frame.columns if c not in exclude and pd.api.types.is_numeric_dtype(frame[c])
    ]
    categorical = [c for c in frame.columns if c not in exclude and c not in numeric]
    if not numeric and not categorical:
        raise TrainingDataError("no feature columns left after removing target and id columns")
    return numeric, categorical


def build_estimator(hp: TrainingHyperparameters) -> Any:
    if hp.model == "logistic_regression":
        return LogisticRegression(C=hp.C, max_iter=hp.max_iter, random_state=hp.seed)
    if hp.model == "random_forest":
        return RandomForestClassifier(
            n_estimators=hp.n_estimators, max_depth=hp.max_depth, random_state=hp.seed, n_jobs=1
        )
    return HistGradientBoostingClassifier(
        max_iter=hp.n_estimators, max_depth=hp.max_depth, random_state=hp.seed
    )


def build_pipeline(
    hp: TrainingHyperparameters, numeric: list[str], categorical: list[str]
) -> Pipeline:
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    return Pipeline(
        [("prep", ColumnTransformer(transformers, remainder="drop")), ("clf", build_estimator(hp))]
    )


def binary_target(frame: pd.DataFrame, target: str, positive_label: int | str) -> np.ndarray:
    if target not in frame.columns:
        raise TrainingDataError(
            f"target column {target!r} not in data columns {list(frame.columns)}"
        )
    values = frame[target]
    y = (values.astype(str) == str(positive_label)).to_numpy(dtype=np.int64)
    if y.sum() == 0 or y.sum() == len(y):
        raise TrainingDataError(
            f"target {target!r} has a single class for positive_label={positive_label!r}"
        )
    return y


def evaluate(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (p >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    both = 0 < int(y.sum()) < len(y)
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "auc": float(roc_auc_score(y, p)) if both else float("nan"),
        "log_loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
        "brier": float(np.mean((p - y) ** 2)),
        "accuracy": float((pred == y).mean()),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "n": float(len(y)),
    }


def emit_metrics(metrics: dict[str, float], stream: TextIO) -> None:
    """One `validation:<name>=<value>` line per metric — the regex-friendly contract."""
    for name in METRIC_NAMES:
        value = metrics.get(name)
        if value is not None and value == value:
            stream.write(f"{METRIC_PREFIX}{name}={value:.6f}\n")
    stream.flush()


def _write_checkpoint(paths: SageMakerPaths, payload: dict[str, Any]) -> None:
    directory = paths.checkpoints_dir
    if directory.is_dir():
        (directory / "checkpoint.json").write_text(json.dumps(payload), encoding="utf-8")


def run_training(
    paths: SageMakerPaths,
    hp: TrainingHyperparameters | None = None,
    *,
    job_name: str | None = None,
    stream: TextIO = sys.stdout,
) -> TrainingReport:
    try:
        hp = hp or TrainingHyperparameters.from_sagemaker_file(paths.hyperparameters_file)
        paths.ensure_output_dirs()
        log.info("hyperparameters: %s", hp.model_dump())
        checkpoint = paths.checkpoints_dir / "checkpoint.json"
        if checkpoint.is_file():
            log.info(
                "found %s from an interrupted run; single-shot fit restarts from scratch",
                checkpoint,
            )

        train_df = load_channel_frame(paths, "train", required=True)
        assert train_df is not None
        val_df = load_channel_frame(paths, "validation", required=False)
        if val_df is None and hp.validation_split > 0:
            y_all = binary_target(train_df, hp.target, hp.positive_label)
            train_df, val_df = train_test_split(
                train_df, test_size=hp.validation_split, random_state=hp.seed, stratify=y_all
            )
            train_df = train_df.reset_index(drop=True)
            val_df = val_df.reset_index(drop=True)

        numeric, categorical = infer_columns(train_df, hp.target, hp.id_columns)
        features = numeric + categorical
        y_train = binary_target(train_df, hp.target, hp.positive_label)
        pipeline = build_pipeline(hp, numeric, categorical)
        pipeline.fit(train_df[features], y_train)
        _write_checkpoint(paths, {"status": "fitted", "utc": datetime.now(UTC).isoformat()})

        if val_df is not None:
            y_val = binary_target(val_df, hp.target, hp.positive_label)
            p_val = np.asarray(pipeline.predict_proba(val_df[features]))[:, 1]
            metrics, metrics_on, n_val = (
                evaluate(y_val, p_val, hp.threshold),
                "validation",
                len(val_df),
            )
        else:
            p_train = np.asarray(pipeline.predict_proba(train_df[features]))[:, 1]
            metrics, metrics_on, n_val = evaluate(y_train, p_train, hp.threshold), "train", 0
            log.warning("no validation data: metrics are in-sample")
        emit_metrics(metrics, stream)

        metadata = ModelMetadata(
            target=hp.target,
            positive_label=hp.positive_label,
            threshold=hp.threshold,
            feature_columns=features,
            numeric_columns=numeric,
            categorical_columns=categorical,
            model_kind=hp.model,
            hyperparameters=hp.model_dump(mode="json"),
            metrics=metrics,
            metrics_on=metrics_on,
            n_train=len(train_df),
            n_validation=int(n_val),
            training_job_name=job_name,
        )
        stamped = save_model(pipeline, metadata, paths.model_dir)
        (paths.output_data_dir / "evaluation.json").write_text(
            json.dumps(
                {
                    "metrics": metrics,
                    "metrics_on": metrics_on,
                    "model_sha256": stamped.model_sha256,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return TrainingReport(
            metrics, metrics_on, len(train_df), int(n_val), paths.model_dir, stamped
        )
    except Exception as exc:
        paths.output_dir.mkdir(parents=True, exist_ok=True)
        paths.failure_file.write_text(f"{type(exc).__name__}: {exc}", encoding="utf-8")
        raise


def main(argv: list[str] | None = None) -> int:
    del argv
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    paths = SageMakerPaths.from_env()
    try:
        report = run_training(paths, job_name=paths.env.get("TRAINING_JOB_NAME"))
    except Exception:
        traceback.print_exc()
        return 1
    log.info(
        "training complete: %s rows, metrics on %s: %s",
        report.n_train,
        report.metrics_on,
        report.metrics,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
