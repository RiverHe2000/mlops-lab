from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from smdeploy.hyperparams import TrainingHyperparameters
from smdeploy.paths import DEFAULT_BASE, SageMakerPaths


def test_default_layout_is_opt_ml() -> None:
    p = SageMakerPaths(env={})
    assert p.base == DEFAULT_BASE
    assert p.hyperparameters_file == DEFAULT_BASE / "input" / "config" / "hyperparameters.json"
    assert p.channel("train") == DEFAULT_BASE / "input" / "data" / "train"
    assert p.model_dir == DEFAULT_BASE / "model"
    assert p.failure_file == DEFAULT_BASE / "output" / "failure"
    assert p.output_data_dir == DEFAULT_BASE / "output" / "data"
    assert p.checkpoints_dir == DEFAULT_BASE / "checkpoints"


def test_env_overrides(tmp_path: Path) -> None:
    env = {
        "SMDEPLOY_BASE_DIR": str(tmp_path),
        "SM_CHANNEL_TRAIN": str(tmp_path / "elsewhere"),
        "SM_MODEL_DIR": str(tmp_path / "m"),
        "SM_OUTPUT_DATA_DIR": str(tmp_path / "od"),
        "SM_NUM_CPUS": "3",
    }
    p = SageMakerPaths.from_env(env)
    assert p.base == tmp_path
    assert p.channel("train") == tmp_path / "elsewhere"
    assert p.channel("validation") == tmp_path / "input" / "data" / "validation"
    assert p.model_dir == tmp_path / "m"
    assert p.output_data_dir == tmp_path / "od"
    assert p.num_cpus == 3
    assert SageMakerPaths(env={}).num_cpus >= 1


def test_channel_files_and_configs(tmp_path: Path) -> None:
    p = SageMakerPaths(base=tmp_path, env={})
    assert p.channel_files("train") == []
    (p.channel("train") / "part").mkdir(parents=True)
    (p.channel("train") / "b.csv").write_text("x\n1\n")
    (p.channel("train") / "part" / "a.csv").write_text("x\n2\n")
    (p.channel("train") / "ignore.txt").write_text("nope")
    names = [f.name for f in p.channel_files("train")]
    assert names == [
        "b.csv",
        "a.csv",
    ]  # deterministic: sorted by full path, sub-directories included
    assert p.read_resource_config() == {} and p.current_host == "algo-1"
    p.input_config.mkdir(parents=True)
    p.resource_config_file.write_text(
        json.dumps({"current_host": "algo-2", "hosts": ["algo-1", "algo-2"]})
    )
    p.input_data_config_file.write_text("[]")
    assert p.current_host == "algo-2"
    assert p.read_input_data_config() == {}  # non-object JSON is treated as empty
    p.ensure_output_dirs()
    assert p.model_dir.is_dir() and p.output_data_dir.is_dir()


def test_hyperparameters_decode_sagemaker_strings() -> None:
    raw = {
        "target": "default",
        "C": "0.5",
        "max_depth": "null",
        "n_estimators": "50",
        "id_columns": '["a", "b"]',
        "threshold": "0.3",
        "sagemaker_program": "ignored.py",
        "model": "random_forest",
    }
    hp = TrainingHyperparameters.from_sagemaker_dict(raw)
    assert hp.C == 0.5 and hp.max_depth is None and hp.n_estimators == 50
    assert hp.id_columns == ["a", "b"] and hp.model == "random_forest" and hp.threshold == 0.3
    assert TrainingHyperparameters.from_sagemaker_dict({"id_columns": "a, b ,c"}).id_columns == [
        "a",
        "b",
        "c",
    ]
    assert TrainingHyperparameters.from_sagemaker_dict({"id_columns": ""}).id_columns == []
    assert TrainingHyperparameters.from_sagemaker_dict({"max_depth": "None"}).max_depth is None


def test_hyperparameters_roundtrip_and_validation(tmp_path: Path) -> None:
    hp = TrainingHyperparameters(model="hist_gradient_boosting", max_depth=4, id_columns=["id"])
    transported = hp.to_sagemaker_dict()
    assert all(isinstance(v, str) for v in transported.values())
    assert TrainingHyperparameters.from_sagemaker_dict(transported) == hp
    with pytest.raises(ValidationError):
        TrainingHyperparameters.from_sagemaker_dict({"threshold": "1.5"})
    with pytest.raises(ValidationError):
        TrainingHyperparameters.from_sagemaker_dict({"model": "svm"})
    missing = tmp_path / "none.json"
    assert TrainingHyperparameters.from_sagemaker_file(missing) == TrainingHyperparameters()
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2]")
    with pytest.raises(ValueError, match="JSON object"):
        TrainingHyperparameters.from_sagemaker_file(bad)
    good = tmp_path / "hp.json"
    good.write_text(json.dumps({"C": "2"}))
    assert TrainingHyperparameters.from_sagemaker_file(good).C == 2.0
