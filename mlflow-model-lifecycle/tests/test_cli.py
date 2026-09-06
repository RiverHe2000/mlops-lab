from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
import yaml

from mlreg import registry as R
from mlreg import tracking as T
from mlreg.cli import _threshold_for, main
from mlreg.config import DataSchema, TrainingConfig

from .conftest import DATA, SCHEMA, log_quick_model, make_config


def _write_cfg(tmp: Path, name: str, params: dict[str, Any]) -> Path:
    cfg = make_config(tmp, name=name, params=params, registered_model="cli-model")
    path = tmp / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8")
    return path


def _write_policy(tmp: Path, name: str, **absolute: float) -> Path:
    policy = {
        "absolute": absolute,
        "relative": {
            "auc_noninferiority_margin": 0.2,
            "brier_max_degradation": 0.2,
            "max_score_psi": 10.0,
            "n_boot": 150,
        },
        "slices": {"min_auc": 0.5, "min_n": 30},
    }
    path = tmp / f"{name}.yaml"
    path.write_text(yaml.safe_dump(policy), encoding="utf-8")
    return path


def test_validate_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate", "--data", str(DATA), "--schema", str(SCHEMA)]) == 0
    assert "OK" in capsys.readouterr().out
    bad = pd.read_csv(DATA, dtype=str)
    bad.loc[0, "purpose"] = "yacht"
    bad_path = tmp_path / "bad.csv"
    bad.to_csv(bad_path, index=False)
    assert main(["validate", "--data", str(bad_path), "--schema", str(SCHEMA)]) == 1
    assert "unknown_category" in capsys.readouterr().out


def test_full_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    base = _write_cfg(tmp_path, "base", {"C": 1.0, "max_iter": 500})
    chal = _write_cfg(tmp_path, "chal", {"C": 0.3, "max_iter": 500})
    permissive = _write_policy(
        tmp_path, "permissive", min_auc=0.6, max_brier=0.3, max_ece=0.2, min_n_holdout=100
    )
    strict = _write_policy(
        tmp_path, "strict", min_auc=0.99, max_brier=0.3, max_ece=0.2, min_n_holdout=100
    )
    name = "cli-model"
    tracking = TrainingConfig.from_yaml(base).tracking

    rc = main(
        [
            "train",
            "--config",
            str(base),
            "--n-boot",
            "30",
            "--register",
            "--alias",
            "challenger",
            "--out",
            "out1",
        ]
    )
    assert rc == 0
    summary = json.loads((tmp_path / "out1" / "train_summary.json").read_text(encoding="utf-8"))
    assert summary["registered"]["version"] == 1
    assert "challenger" in summary["registered"]["aliases"]

    # first promotion: no champion yet → absolute checks only
    rc = main(
        [
            "promote",
            "--config",
            str(base),
            "--policy",
            str(permissive),
            "--candidate-version",
            "1",
            "--out",
            "gate1",
            "--approved-by",
            "tester",
        ]
    )
    assert rc == 0
    assert "PROMOTE" in (tmp_path / "gate1" / "gate_report.md").read_text(encoding="utf-8")
    card = (tmp_path / "gate1" / "model_card.md").read_text(encoding="utf-8")
    assert "# Model card" in card and "Approver: tester" in card
    T.configure(tracking)
    champion = R.get_alias(name, "champion")
    assert champion is not None and champion.version == 1
    assert champion.tags["gate.decision"] == "PROMOTE" and champion.tags["promoted_by"] == "tester"
    assert R.get_alias(name, "challenger") is None  # consumed by the promotion

    capsys.readouterr()
    assert (
        main(
            [
                "promote",
                "--config",
                str(base),
                "--policy",
                str(permissive),
                "--candidate-version",
                "1",
            ]
        )
        == 0
    )
    assert "already holds" in capsys.readouterr().out

    assert main(["train", "--config", str(chal), "--n-boot", "30", "--register"]) == 0

    # strict policy → HOLD, exit code 2, champion untouched
    rc = main(
        [
            "promote",
            "--config",
            str(chal),
            "--policy",
            str(strict),
            "--candidate-version",
            "2",
            "--out",
            "gate2",
        ]
    )
    assert rc == 2
    T.configure(tracking)
    assert R.get_alias(name, "champion").version == 1  # type: ignore[union-attr]
    assert R.get_version(name, 2).tags["gate.decision"] == "HOLD"
    report = json.loads((tmp_path / "gate2" / "gate_report.json").read_text(encoding="utf-8"))
    assert report["decision"] == "HOLD" and report["champion"] is not None
    assert any(c["name"] == "same_data_snapshot" and c["passed"] for c in report["checks"])

    # dry run passes but does not move the alias
    rc = main(
        [
            "promote",
            "--config",
            str(chal),
            "--policy",
            str(permissive),
            "--candidate-version",
            "2",
            "--dry-run",
        ]
    )
    assert rc == 0
    T.configure(tracking)
    assert R.get_alias(name, "champion").version == 1  # type: ignore[union-attr]

    rc = main(
        [
            "promote",
            "--config",
            str(chal),
            "--policy",
            str(permissive),
            "--candidate-version",
            "2",
            "--approved-by",
            "reviewer",
        ]
    )
    assert rc == 0
    T.configure(tracking)
    assert R.get_alias(name, "champion").version == 2  # type: ignore[union-attr]
    assert R.get_alias(name, "previous").version == 1  # type: ignore[union-attr]

    capsys.readouterr()
    assert main(["compare", "--config", str(base), "--versions", "1", "2", "--n-boot", "100"]) == 0
    out = capsys.readouterr().out
    assert "auc" in out and "ci_lower" in out

    assert main(["card", "--config", str(base), "--version", "1", "--out", "cards/v1.md"]) == 0
    assert "Model card" in (tmp_path / "cards" / "v1.md").read_text(encoding="utf-8")

    capsys.readouterr()
    assert main(["serve-check", "--config", str(base), "--alias", "champion"]) == 0
    assert "accepted a serving payload" in capsys.readouterr().out


def test_register_command(
    tmp_path: Path, schema: DataSchema, frame: pd.DataFrame, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg_path = _write_cfg(tmp_path, "reg", {"C": 1.0, "max_iter": 300})
    tracking = TrainingConfig.from_yaml(cfg_path).tracking
    uri = log_quick_model(schema, frame, tracking)
    rc = main(
        [
            "register",
            "--config",
            str(cfg_path),
            "--model-uri",
            uri,
            "--alias",
            "challenger",
            "--tag",
            "a=b",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert (
        payload["version"] == 1
        and payload["tags"]["a"] == "b"
        and payload["aliases"] == ["challenger"]
    )


def test_threshold_lookup_falls_back(use_trained_store: SimpleNamespace) -> None:
    t = use_trained_store
    rv = R.register(t.result.model_uri, "fallback-model")
    assert _threshold_for(rv) == pytest.approx(t.result.threshold, abs=1e-4)  # from the run params
    assert _threshold_for(R.RegisteredVersion("x", 1, None, "src")) == 0.5
    tagged = R.RegisteredVersion("x", 1, None, "src", tags={"decision.threshold": "0.2"})
    assert _threshold_for(tagged) == 0.2
