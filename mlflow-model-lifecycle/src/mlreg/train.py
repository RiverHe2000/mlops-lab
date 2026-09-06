"""Tracked training: contract → split → CV (nested runs) → fit → evaluate → log model."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from . import metrics as M
from . import tracking as T
from .config import CVConfig, DataSchema, TrainingConfig
from .data import Dataset, load_frame, make_split
from .models import build_pipeline, global_importance, predict_pd
from .plots import plot_calibration, plot_roc, plot_score_distribution
from .pyfunc import PDScorer, log_pd_model
from .stats import bootstrap_ci


@dataclass
class TrainResult:
    run_id: str
    model_uri: str
    threshold: float
    metrics_test: dict[str, float]
    metrics_cv: pd.DataFrame
    slices: dict[str, pd.DataFrame]
    data_fingerprint: str
    config_hash: str
    test_ci: dict[str, dict[str, float]] = field(default_factory=dict)
    pipeline: Any = None

    def cv_mean(self) -> dict[str, float]:
        return T.series_to_metrics(self.metrics_cv.drop(columns="fold").mean())

    def cv_std(self) -> dict[str, float]:
        return T.series_to_metrics(self.metrics_cv.drop(columns="fold").std(ddof=0))

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model_uri": self.model_uri,
            "threshold": self.threshold,
            "metrics_test": self.metrics_test,
            "test_ci": self.test_ci,
            "cv_mean": self.cv_mean(),
            "cv_std": self.cv_std(),
            "data_fingerprint": self.data_fingerprint,
            "config_hash": self.config_hash,
        }


def cross_validate(
    pipeline: Any, X: pd.DataFrame, y: np.ndarray, cv: CVConfig, *, log_nested: bool = False
) -> tuple[pd.DataFrame, np.ndarray]:
    """Stratified K-fold; returns per-fold metrics and out-of-fold PDs (for threshold selection)."""
    skf = StratifiedKFold(n_splits=cv.folds, shuffle=True, random_state=cv.seed)
    oof = np.full(len(y), np.nan, dtype=np.float64)
    rows: list[dict[str, float]] = []
    for k, (tr, va) in enumerate(skf.split(X, y)):
        model = clone(pipeline)
        model.fit(X.iloc[tr], y[tr])
        p = predict_pd(model, X.iloc[va])
        oof[va] = p
        m = {
            "fold": float(k),
            "auc": M.roc_auc(y[va], p),
            "ks": M.ks_statistic(y[va], p),
            "brier": M.brier(y[va], p),
            "log_loss": M.log_loss(y[va], p),
            "ece": M.expected_calibration_error(y[va], p),
        }
        rows.append(m)
        if log_nested:
            with T.start_run(f"fold-{k}", {"mlreg.fold": str(k)}, nested=True):
                T.log_metrics_prefixed({kk: v for kk, v in m.items() if kk != "fold"}, "cv_")
    return pd.DataFrame(rows), oof


def choose_threshold(cfg: TrainingConfig, y_train: np.ndarray, oof: np.ndarray) -> float:
    """Fixed from config, or cost-optimal on *out-of-fold* PDs (never on the holdout)."""
    if cfg.decision_threshold == "cost_optimal":
        t, _ = M.optimal_threshold_by_cost(y_train, oof)
        return t
    return float(cfg.decision_threshold)


def evaluate_holdout(
    y: np.ndarray, p: np.ndarray, threshold: float, protected: pd.DataFrame, *, n_boot: int
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, pd.DataFrame]]:
    summary = M.summarize(y, p, threshold)
    cis = {
        "auc": bootstrap_ci(y, p, M.roc_auc, n_boot=n_boot, seed=1).to_dict(),
        "brier": bootstrap_ci(y, p, M.brier, n_boot=n_boot, seed=2).to_dict(),
        "ks": bootstrap_ci(y, p, M.ks_statistic, n_boot=n_boot, seed=3).to_dict(),
    }
    slices = {
        str(col): M.slice_metrics(y, p, protected[col], threshold) for col in protected.columns
    }
    return summary, cis, slices


def train(cfg: TrainingConfig, *, n_boot: int = 1000, log_nested_cv: bool = True) -> TrainResult:
    schema = DataSchema.from_yaml(cfg.schema_path)
    frame = load_frame(cfg.data_path, schema)
    ds = make_split(frame, schema, cfg.split)
    return train_on(cfg, schema, ds, n_boot=n_boot, log_nested_cv=log_nested_cv)


def train_on(
    cfg: TrainingConfig,
    schema: DataSchema,
    ds: Dataset,
    *,
    n_boot: int = 1000,
    log_nested_cv: bool = True,
) -> TrainResult:
    T.configure(cfg.tracking)
    tags = T.lineage_tags(cfg, ds.fingerprint, schema.fingerprint())
    pipeline = build_pipeline(schema, cfg.model, cfg.split.seed)

    with T.start_run(cfg.name, tags) as run:
        T.log_params_flat({"kind": cfg.model.kind, **cfg.model.params}, "model.")
        T.log_params_flat(cfg.split.model_dump(), "split.")
        T.log_params_flat(cfg.cv.model_dump(), "cv.")
        T.log_params_flat(
            {
                "n_total": ds.n_total,
                "n_train": ds.n_train,
                "n_test": ds.n_test,
                "n_features": len(schema.feature_columns),
                "protected": ",".join(schema.protected_columns) or "-",
            },
            "data.",
        )

        cv_table, oof = cross_validate(
            pipeline, ds.X_train, ds.y_train, cfg.cv, log_nested=log_nested_cv
        )
        T.log_metrics_prefixed(
            T.series_to_metrics(cv_table.drop(columns="fold").mean()), "cv_mean_"
        )
        T.log_metrics_prefixed(
            T.series_to_metrics(cv_table.drop(columns="fold").std(ddof=0)), "cv_std_"
        )
        threshold = choose_threshold(cfg, ds.y_train, oof)
        T.log_params_flat(
            {"threshold": f"{threshold:.4f}", "threshold_rule": str(cfg.decision_threshold)},
            "decision.",
        )

        pipeline.fit(ds.X_train, ds.y_train)
        p_test = predict_pd(pipeline, ds.X_test)
        summary, cis, slices = evaluate_holdout(
            ds.y_test, p_test, threshold, ds.protected_test, n_boot=n_boot
        )
        T.log_metrics_prefixed(summary, "test_")
        for name, ci in cis.items():
            T.log_metrics_prefixed(
                {f"{name}_ci_lower": ci["lower"], f"{name}_ci_upper": ci["upper"]}, "test_"
            )

        # artefacts: the evidence pack a reviewer needs
        T.log_json(cfg.model_dump(mode="json"), "config.json")
        T.log_json(schema.model_dump(mode="json"), "schema.json")
        T.log_json({"metrics": summary, "ci": cis}, "evaluation/holdout.json")
        T.log_frame(cv_table, "cv_folds.csv")
        T.log_frame(M.calibration_table(ds.y_test, p_test), "evaluation/calibration.csv")
        T.log_frame(M.decile_lift(ds.y_test, p_test), "evaluation/decile_lift.csv")
        T.log_frame(global_importance(pipeline), "evaluation/feature_importance.csv")
        for col, table in slices.items():
            T.log_frame(table, f"evaluation/slice_{col}.csv")
        T.log_frame(
            pd.DataFrame({schema.id_column: ds.ids_test, "y": ds.y_test, "pd": p_test}),
            "evaluation/predictions_test.csv",
        )
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            T.log_file(plot_roc(ds.y_test, p_test, summary["auc"], d / "roc.png"), "plots")
            T.log_file(
                plot_calibration(M.calibration_table(ds.y_test, p_test), d / "calibration.png"),
                "plots",
            )
            T.log_file(
                plot_score_distribution(ds.y_test, p_test, threshold, d / "scores.png"), "plots"
            )

        scorer = PDScorer(pipeline, schema, threshold)
        info = log_pd_model(scorer, ds.X_test.head(20))
        run_id = str(run.info.run_id)
        model_uri = str(info.model_uri)

    return TrainResult(
        run_id=run_id,
        model_uri=model_uri,
        threshold=threshold,
        metrics_test=summary,
        metrics_cv=cv_table,
        slices=slices,
        data_fingerprint=ds.fingerprint,
        config_hash=cfg.config_hash(),
        test_ci=cis,
        pipeline=pipeline,
    )
