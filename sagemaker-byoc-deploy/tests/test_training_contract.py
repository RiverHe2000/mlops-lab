from __future__ import annotations

import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smdeploy import training
from smdeploy.aws.training_job import default_metric_definitions
from smdeploy.hyperparams import TrainingHyperparameters
from smdeploy.local_run import write_sagemaker_layout
from smdeploy.model_io import METADATA_FILE, MODEL_FILE, ModelIntegrityError, load_model
from smdeploy.paths import SageMakerPaths
from smdeploy.synthetic import CATEGORICAL, NUMERIC, TARGET
from smdeploy.training import TrainingDataError, TrainingReport, run_training


def test_training_writes_the_contract_outputs(trained: TrainingReport) -> None:
    model_dir = trained.model_dir
    assert (model_dir / MODEL_FILE).is_file() and (model_dir / METADATA_FILE).is_file()
    meta = trained.metadata
    assert meta.feature_columns == NUMERIC + CATEGORICAL
    assert (
        meta.target == TARGET and meta.threshold == 0.3 and meta.model_kind == "logistic_regression"
    )
    assert (
        meta.metrics_on == "validation" and trained.n_validation == 100 and trained.n_train == 400
    )
    assert 0.6 < meta.metrics["auc"] <= 1.0
    assert meta.training_job_name == "local-simulation"
    assert len(meta.model_sha256) == 64
    evaluation = json.loads((model_dir.parent / "output" / "data" / "evaluation.json").read_text())
    assert evaluation["metrics"] == meta.metrics and evaluation["model_sha256"] == meta.model_sha256
    checkpoint = model_dir.parent / "checkpoints" / "checkpoint.json"
    assert json.loads(checkpoint.read_text())["status"] == "fitted"
    assert not (model_dir.parent / "output" / "failure").exists()


def test_metric_lines_match_metric_definitions(trained: TrainingReport) -> None:
    stream = io.StringIO()
    training.emit_metrics(trained.metrics, stream)
    text = stream.getvalue()
    for definition in default_metric_definitions():
        match = re.search(definition.regex, text)
        assert match, f"{definition.name} not captured from:\n{text}"
        assert float(match.group(1)) == pytest.approx(
            trained.metrics[definition.name.split(":")[1]], abs=1e-6
        )


def test_model_roundtrip_and_integrity(trained: TrainingReport) -> None:
    loaded = load_model(trained.model_dir)
    assert loaded.metadata == trained.metadata
    tampered_dir = trained.model_dir.parent / "tampered"
    tampered_dir.mkdir(exist_ok=True)
    for name in (MODEL_FILE, METADATA_FILE):
        (tampered_dir / name).write_bytes((trained.model_dir / name).read_bytes())
    with (tampered_dir / MODEL_FILE).open("ab") as fh:
        fh.write(b"corruption")
    with pytest.raises(ModelIntegrityError):
        load_model(tampered_dir)
    assert (
        load_model(tampered_dir, verify=False).metadata.model_sha256
        == trained.metadata.model_sha256
    )
    with pytest.raises(FileNotFoundError):
        load_model(tampered_dir.parent / "nowhere")


def test_missing_channel_writes_failure_file(tmp_path: Path) -> None:
    paths = SageMakerPaths(base=tmp_path, env={})
    with pytest.raises(TrainingDataError, match="channel 'train'"):
        run_training(paths, TrainingHyperparameters())
    assert "TrainingDataError" in paths.failure_file.read_text()


def test_single_class_and_missing_target(tmp_path: Path, frame: pd.DataFrame) -> None:
    one_class = frame.copy()
    one_class[TARGET] = 0
    csv = tmp_path / "one.csv"
    one_class.to_csv(csv, index=False)
    paths = write_sagemaker_layout(
        tmp_path / "a", train_csv=csv, hyperparameters={"target": TARGET}
    )
    with pytest.raises(TrainingDataError, match="single class"):
        run_training(paths)
    paths2 = write_sagemaker_layout(
        tmp_path / "b", train_csv=csv, hyperparameters={"target": "nope"}
    )
    with pytest.raises(TrainingDataError, match="target column"):
        run_training(paths2)
    assert "TrainingDataError" in paths2.failure_file.read_text()


def test_validation_split_and_in_sample_fallback(tmp_path: Path, csv_dir: Path) -> None:
    split_paths = write_sagemaker_layout(
        tmp_path / "split",
        train_csv=csv_dir / "train.csv",
        hyperparameters={
            "target": TARGET,
            "id_columns": "application_id",
            "validation_split": 0.25,
        },
    )
    report = run_training(split_paths)
    assert (
        report.metrics_on == "validation" and report.n_validation == 100 and report.n_train == 300
    )
    insample = write_sagemaker_layout(
        tmp_path / "insample",
        train_csv=csv_dir / "train.csv",
        hyperparameters={"target": TARGET, "id_columns": "application_id", "validation_split": 0},
    )
    report2 = run_training(insample, stream=io.StringIO())
    assert report2.metrics_on == "train" and report2.n_validation == 0


@pytest.mark.parametrize("model", ["random_forest", "hist_gradient_boosting"])
def test_other_estimators(tmp_path: Path, csv_dir: Path, model: str) -> None:
    paths = write_sagemaker_layout(
        tmp_path / model,
        train_csv=csv_dir / "train.csv",
        validation_csv=csv_dir / "validation.csv",
        hyperparameters={
            "target": TARGET,
            "id_columns": "application_id",
            "model": model,
            "n_estimators": 30,
            "max_depth": 4,
        },
    )
    report = run_training(paths, stream=io.StringIO())
    assert report.metadata.model_kind == model and report.metrics["auc"] > 0.6


def test_infer_columns_and_evaluate() -> None:
    frame = pd.DataFrame({"id": ["a", "b"], "x": [1.0, 2.0], "c": ["u", "v"], "y": [0, 1]})
    assert training.infer_columns(frame, "y", ["id"]) == (["x"], ["c"])
    with pytest.raises(TrainingDataError, match="no feature columns"):
        training.infer_columns(frame[["id", "y"]], "y", ["id"])
    y = np.array([0, 0, 1, 1])
    m = training.evaluate(y, np.array([0.1, 0.6, 0.7, 0.9]), 0.5)
    assert (
        m["auc"] == 1.0
        and m["accuracy"] == 0.75
        and m["precision"] == pytest.approx(2 / 3)
        and m["recall"] == 1.0
    )
    degenerate = training.evaluate(np.array([1, 1]), np.array([0.2, 0.3]), 0.5)
    assert degenerate["auc"] != degenerate["auc"]  # nan when a single class is present
    assert degenerate["precision"] == 0.0


def test_main_entrypoint(tmp_path: Path, csv_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = write_sagemaker_layout(
        tmp_path / "main",
        train_csv=csv_dir / "train.csv",
        validation_csv=csv_dir / "validation.csv",
        hyperparameters={"target": TARGET, "id_columns": "application_id"},
    )
    monkeypatch.setenv("SMDEPLOY_BASE_DIR", str(paths.base))
    monkeypatch.setenv("TRAINING_JOB_NAME", "job-42")
    assert training.main() == 0
    assert load_model(paths.model_dir).metadata.training_job_name == "job-42"
    monkeypatch.setenv("SMDEPLOY_BASE_DIR", str(tmp_path / "empty"))
    assert training.main() == 1
    assert (tmp_path / "empty" / "output" / "failure").is_file()
