from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mlreg.config import DataSchema, ModelConfig, SchemaColumn, TrackingConfig, TrainingConfig

from .conftest import SCHEMA, make_config


def test_schema_loads_and_partitions_columns(schema: DataSchema) -> None:
    assert schema.target == "default"
    assert schema.id_column == "application_id"
    assert len(schema.feature_columns) == 18
    assert set(schema.protected_columns) == {"personal_status_sex", "foreign_worker"}
    assert len(schema.numeric_features) == 7
    assert len(schema.categorical_features) == 11
    purpose = schema.column("purpose")
    assert purpose.categories is not None and "car_new" in purpose.categories
    with pytest.raises(KeyError):
        schema.column("nope")


def test_schema_fingerprint_tracks_content(schema: DataSchema) -> None:
    assert schema.fingerprint() == DataSchema.from_yaml(SCHEMA).fingerprint()
    assert schema.model_copy(update={"version": "1.1"}).fingerprint() != schema.fingerprint()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "x", "kind": "numeric", "categories": ["a"]},
        {"name": "x", "kind": "categorical", "min": 0},
        {"name": "x", "kind": "numeric", "min": 5, "max": 1},
        {"name": "x", "kind": "categorical", "categories": ["a", "a"]},
    ],
)
def test_schema_column_consistency(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        SchemaColumn(**kwargs)


def _cols(**over: dict[str, Any]) -> list[dict[str, Any]]:
    base: dict[str, dict[str, Any]] = {
        "id": {"name": "id", "kind": "categorical", "role": "id"},
        "f": {"name": "f", "kind": "numeric"},
        "y": {"name": "y", "kind": "numeric", "role": "target"},
    }
    base.update(over)
    return list(base.values())


def test_schema_role_rules() -> None:
    ok = DataSchema(name="s", version="1", target="y", id_column="id", columns=_cols())
    assert ok.feature_columns == ["f"]
    bad: list[dict[str, Any]] = [
        {"target": "f", "id_column": "id", "columns": _cols()},
        {"target": "y", "id_column": "zz", "columns": _cols()},
        {
            "target": "y",
            "id_column": "id",
            "columns": _cols(f={"name": "f", "kind": "numeric", "role": "protected"}),
        },
        {"target": "y", "id_column": "id", "columns": [*_cols(), {"name": "f", "kind": "numeric"}]},
        {
            "target": "y",
            "id_column": "id",
            "columns": [*_cols(), {"name": "y2", "kind": "numeric", "role": "target"}],
        },
    ]
    for case in bad:
        with pytest.raises(ValidationError):
            DataSchema.model_validate({"name": "s", "version": "1", **case})


def test_training_config_from_yaml_resolves_paths_and_hash(root: Path, tmp_path: Path) -> None:
    path = root / "configs" / "baseline_logreg.yaml"
    cfg = TrainingConfig.from_yaml(path, base_dir=root)
    assert cfg.data_path.is_absolute() and cfg.data_path.exists()
    assert cfg.schema_path.exists()
    h = cfg.config_hash()
    # paths and tracking do not influence the hash; the model does
    assert TrainingConfig.from_yaml(path, base_dir=tmp_path).config_hash() == h
    assert cfg.model_copy(update={"tracking": TrackingConfig(experiment="zzz")}).config_hash() == h
    assert cfg.model_copy(update={"model": ModelConfig(kind="random_forest")}).config_hash() != h


def test_all_shipped_configs_validate(root: Path) -> None:
    paths = sorted((root / "configs").glob("*.yaml"))
    assert len(paths) == 3
    for path in paths:
        cfg = TrainingConfig.from_yaml(path, base_dir=root)
        assert cfg.tracking.registered_model == "credit-pd"
        assert cfg.decision_threshold == "cost_optimal"


def test_threshold_validation(tmp_path: Path) -> None:
    assert make_config(tmp_path, threshold=0.3).decision_threshold == 0.3
    with pytest.raises(ValidationError):
        make_config(tmp_path, threshold=1.5)
    with pytest.raises(ValidationError):
        make_config(tmp_path, threshold="bogus")


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    t = TrackingConfig(tracking_uri="sqlite:///a.db", artifact_location="/x")
    assert t.resolved_uri() == "sqlite:///a.db"
    assert t.resolved_artifact_location() == "/x"
    monkeypatch.setenv("MLREG_TRACKING_URI", "http://mlflow:5000")
    monkeypatch.setenv("MLREG_ARTIFACT_LOCATION", "s3://bucket/x")
    assert t.resolved_uri() == "http://mlflow:5000"
    assert t.resolved_artifact_location() == "s3://bucket/x"


def test_unknown_keys_rejected() -> None:
    with pytest.raises(ValidationError):
        TrackingConfig.model_validate({"bogus": 1})
