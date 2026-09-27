from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
import yaml

from mlreg import holdout
from mlreg import registry as R
from mlreg.cli import main
from mlreg.config import DataSchema, SplitConfig
from mlreg.data import make_split
from mlreg.holdout import TAG, HoldoutMismatchError, SplitManifest, verify_run_holdout


def test_manifest_records_actual_membership(frame: pd.DataFrame, schema: DataSchema) -> None:
    ds = make_split(frame, schema, SplitConfig(seed=42))
    manifest = SplitManifest.from_dataset(ds)
    manifest.check(manifest)
    assert len(manifest.train_ids) == ds.n_train
    assert len(manifest.test_ids) == ds.n_test
    assert not set(manifest.train_ids) & set(manifest.test_ids)
    assert set(manifest.test_ids).isdisjoint(ds.ids_test)
    same = SplitManifest.model_validate_json(manifest.model_dump_json())
    assert same.fingerprint() == manifest.fingerprint()
    # Row order within a partition must not change its identity.
    ds.ids_train = ds.ids_train.iloc[::-1]
    ds.ids_test = ds.ids_test.iloc[::-1]
    assert SplitManifest.from_dataset(ds) == manifest
    # The whole CSV digest stays constant when a split changes, but this guard catches it.
    other = SplitManifest.from_dataset(make_split(frame, schema, SplitConfig(seed=7)))
    assert other.data_fingerprint == manifest.data_fingerprint
    with pytest.raises(HoldoutMismatchError, match="used to train"):
        manifest.check(other)


@pytest.mark.parametrize(
    ("saved", "current", "message"),
    [
        ({"train_ids": ()}, {}, "empty"),
        ({"train_ids": ("a", "a")}, {}, "duplicated"),
        ({"test_ids": ()}, {}, "empty"),
        ({"test_ids": ("b", "b")}, {}, "duplicated"),
        ({"train_ids": ("b",)}, {}, "overlap"),
        ({}, {"data_fingerprint": "changed"}, "snapshot"),
        ({}, {"test_ids": ("a",)}, "used to train"),
        ({}, {"test_ids": ("c",)}, "membership differs"),
        ({}, {"train_ids": ("c",)}, "membership differs"),
    ],
)
def test_invalid_membership_is_rejected(
    saved: dict[str, Any], current: dict[str, Any], message: str
) -> None:
    manifest = SplitManifest(data_fingerprint="snapshot", train_ids=("a",), test_ids=("b",))
    with pytest.raises(HoldoutMismatchError, match=message):
        manifest.model_copy(update=saved).check(manifest.model_copy(update=current))


@pytest.mark.parametrize("defect", ["none", "no_run", "no_tag", "missing", "invalid", "tampered"])
def test_run_evidence_fails_closed(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = SplitManifest(data_fingerprint="snapshot", train_ids=("a",), test_ids=("b",))
    path = tmp_path / "manifest.json"
    if defect != "missing":
        path.write_text(
            "{}" if defect == "invalid" else manifest.model_dump_json(), encoding="utf-8"
        )
    tags = {} if defect == "no_tag" else {TAG: manifest.fingerprint()}
    if defect == "tampered":
        tags[TAG] = "wrong"
    fake = SimpleNamespace(
        get_run=lambda _: SimpleNamespace(data=SimpleNamespace(tags=tags)),
        download_artifacts=lambda *_: str(path),
    )
    monkeypatch.setattr(holdout, "MlflowClient", lambda: fake)
    if defect == "none":
        verify_run_holdout("run", manifest)
    else:
        with pytest.raises(HoldoutMismatchError):
            verify_run_holdout(None if defect == "no_run" else "run", manifest)


@pytest.mark.parametrize(("seed", "test_size"), [(7, 0.25), (42, 0.4)])
def test_cli_rejects_split_change_before_scoring_or_alias_move(
    seed: int,
    test_size: float,
    use_trained_store: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    t = use_trained_store
    model_name = f"holdout-{seed}-{test_size}"
    candidate = R.register(t.result.model_uri, model_name)
    cfg = t.cfg.model_copy(
        update={
            "split": SplitConfig(seed=seed, test_size=test_size),
            "tracking": t.cfg.tracking.model_copy(update={"registered_model": model_name}),
        }
    )
    config_path = tmp_path / "changed.yaml"
    config_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8")
    policy = tmp_path / "policy.yaml"
    # Disabling the optional same-CSV check cannot bypass the mandatory membership guard.
    policy.write_text("require_same_data_snapshot: false\n", encoding="utf-8")

    def never_score(*_: object) -> None:
        pytest.fail("holdout leakage must be rejected before a model is loaded")

    monkeypatch.setattr(R, "load_version", never_score)
    out = tmp_path / "gate"
    assert (
        main(
            [
                "promote",
                "--config",
                str(config_path),
                "--policy",
                str(policy),
                "--candidate-version",
                str(candidate.version),
                "--out",
                str(out),
            ]
        )
        == 2
    )
    report = json.loads((out / "gate_report.json").read_text(encoding="utf-8"))
    assert report["checks"][0]["name"] == "holdout_identity"
    assert report["challenger"] == {}  # no misleading performance numbers on leaked rows
    assert R.get_alias(model_name, "champion") is None
    assert main(["card", "--config", str(config_path), "--version", "1"]) == 2
    assert "HOLD:" in capsys.readouterr().err
    assert main(["compare", "--config", str(config_path), "--versions", "1", "1"]) == 2
