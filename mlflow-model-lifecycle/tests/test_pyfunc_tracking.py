from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import convert_input_example_to_serving_input, validate_serving_input

from mlreg import tracking as T
from mlreg.config import DataSchema
from mlreg.models import predict_pd
from mlreg.pyfunc import OUTPUT_COLUMNS, PDScorer, score_with
from mlreg.train import choose_threshold, train

from .conftest import make_config


def test_run_has_lineage_tags_params_metrics(use_trained_store: SimpleNamespace) -> None:
    t = use_trained_store
    run = mlflow.get_run(t.result.run_id)
    tags = run.data.tags
    assert tags["data.fingerprint"] == t.result.data_fingerprint
    assert tags["mlreg.config_hash"] == t.result.config_hash
    assert tags["mlreg.model_kind"] == "logistic_regression"
    assert tags["owner"] == "tests"
    assert "data.schema_fingerprint" in tags and "created_utc" in tags
    params = run.data.params
    assert params["model.kind"] == "logistic_regression"
    assert params["cv.folds"] == "3"
    assert params["data.n_test"] == "250"
    assert float(params["decision.threshold"]) == pytest.approx(t.result.threshold, abs=1e-4)
    metrics = run.data.metrics
    assert metrics["test_auc"] == pytest.approx(t.result.metrics_test["auc"])
    assert metrics["test_auc_ci_lower"] <= metrics["test_auc"] <= metrics["test_auc_ci_upper"]
    assert "cv_mean_auc" in metrics and "cv_std_brier" in metrics


def test_nested_cv_runs(use_trained_store: SimpleNamespace) -> None:
    t = use_trained_store
    exp = mlflow.get_experiment_by_name(t.cfg.tracking.experiment)
    assert exp is not None
    children = MlflowClient().search_runs(
        [exp.experiment_id], f"tags.mlflow.parentRunId = '{t.result.run_id}'"
    )
    assert len(children) == t.cfg.cv.folds
    assert all("cv_auc" in r.data.metrics for r in children)
    assert len(t.result.metrics_cv) == t.cfg.cv.folds
    assert set(t.result.cv_mean()) == {"auc", "ks", "brier", "log_loss", "ece"}


def test_artifacts_form_an_evidence_pack(
    use_trained_store: SimpleNamespace, tmp_path: Path
) -> None:
    t = use_trained_store
    client = MlflowClient()
    top = {f.path for f in client.list_artifacts(t.result.run_id)}
    assert {"config.json", "schema.json", "cv_folds.csv", "evaluation", "plots"} <= top
    evaluation = {f.path for f in client.list_artifacts(t.result.run_id, "evaluation")}
    assert {
        "evaluation/holdout.json",
        "evaluation/predictions_test.csv",
        "evaluation/calibration.csv",
        "evaluation/decile_lift.csv",
        "evaluation/feature_importance.csv",
        "evaluation/slice_personal_status_sex.csv",
        "evaluation/slice_foreign_worker.csv",
    } <= evaluation
    plots = {f.path for f in client.list_artifacts(t.result.run_id, "plots")}
    assert {"plots/roc.png", "plots/calibration.png", "plots/scores.png"} <= plots
    local = client.download_artifacts(t.result.run_id, "evaluation/holdout.json", str(tmp_path))
    holdout = json.loads(Path(local).read_text(encoding="utf-8"))
    assert holdout["metrics"]["auc"] == pytest.approx(t.result.metrics_test["auc"])
    preds = pd.read_csv(
        client.download_artifacts(t.result.run_id, "evaluation/predictions_test.csv", str(tmp_path))
    )
    assert len(preds) == 250 and set(preds.columns) == {"application_id", "y", "pd"}


def test_pyfunc_roundtrip_matches_pipeline(
    use_trained_store: SimpleNamespace, frame: pd.DataFrame, schema: DataSchema
) -> None:
    t = use_trained_store
    model = mlflow.pyfunc.load_model(t.result.model_uri)
    X = frame[schema.feature_columns].head(50)
    out = score_with(model, X)
    assert out.columns.tolist() == OUTPUT_COLUMNS
    np.testing.assert_allclose(out["pd"].to_numpy(), predict_pd(t.result.pipeline, X), atol=1e-9)
    declines = (out["pd"] >= t.result.threshold).tolist()
    assert (out["decision"] == "decline").tolist() == declines
    assert out["reason_codes"].str.len().gt(0).any()
    # the threshold is a signature parameter: override per request
    assert (score_with(model, X, threshold=0.001)["decision"] == "decline").all()
    assert (score_with(model, X, threshold=0.999)["decision"] == "approve").all()


def test_signature_is_enforced(
    use_trained_store: SimpleNamespace, frame: pd.DataFrame, schema: DataSchema
) -> None:
    t = use_trained_store
    model = mlflow.pyfunc.load_model(t.result.model_uri)
    X = frame[schema.feature_columns].head(3)
    wrong_type = X.copy()
    wrong_type["age_years"] = "old"
    with pytest.raises(MlflowException):
        model.predict(wrong_type)
    with pytest.raises(MlflowException):
        model.predict(X.drop(columns=["purpose"]))
    info = mlflow.models.get_model_info(t.result.model_uri)
    sig = info.signature
    assert sig is not None
    assert sig.inputs.input_names() == schema.feature_columns
    assert sig.outputs.input_names() == OUTPUT_COLUMNS
    assert sig.params is not None and "threshold" in sig.params.to_dict()[0]["name"]
    payload = convert_input_example_to_serving_input(X.head(2))
    served = validate_serving_input(t.result.model_uri, payload)
    assert len(pd.DataFrame(served)) == 2


def test_pdscorer_validates_threshold(
    use_trained_store: SimpleNamespace, frame: pd.DataFrame, schema: DataSchema
) -> None:
    scorer = PDScorer(use_trained_store.result.pipeline, schema, 0.3)
    with pytest.raises(ValueError, match="threshold"):
        scorer.score(frame[schema.feature_columns].head(2), threshold=1.5)
    out = scorer.predict(None, frame[schema.feature_columns].head(2), params={"threshold": 0.5})
    assert len(out) == 2


def test_fixed_threshold_and_flat_run(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, name="fixed", threshold=0.42)
    assert choose_threshold(cfg, np.array([0, 1]), np.array([0.1, 0.9])) == 0.42
    result = train(cfg, n_boot=20, log_nested_cv=False)
    assert result.threshold == 0.42
    exp = mlflow.get_experiment_by_name(cfg.tracking.experiment)
    assert exp is not None
    children = MlflowClient().search_runs(
        [exp.experiment_id], f"tags.mlflow.parentRunId = '{result.run_id}'"
    )
    assert children == []
    summary = result.summary()
    assert {"run_id", "model_uri", "metrics_test", "cv_mean", "cv_std", "data_fingerprint"} <= set(
        summary
    )


def test_tracking_helpers(use_trained_store: SimpleNamespace, tmp_path: Path) -> None:
    assert T.git_sha(cwd=tmp_path) is None
    with mlflow.start_run(run_name="helpers") as run:
        T.log_metrics_prefixed({"a": float("nan"), "b": 1.0}, "x_")
        T.log_params_flat({"nested": {"k": [1, 2]}}, "p.")
        T.log_text("hello", "notes/hello.txt")
        assert T.active_run_id() == run.info.run_id
    data = mlflow.get_run(run.info.run_id).data
    assert {k for k in data.metrics if k.startswith("x_")} == {"x_b"}
    assert data.params["p.nested"] == '{"k": [1, 2]}'
    mlflow.end_run()
    with pytest.raises(RuntimeError):
        T.active_run_id()
