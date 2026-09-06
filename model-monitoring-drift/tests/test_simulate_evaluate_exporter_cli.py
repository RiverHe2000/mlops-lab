from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import uvicorn
from fastapi.testclient import TestClient

from mlwatch import cli
from mlwatch.baseline import Baseline
from mlwatch.config import AlertPolicy
from mlwatch.evaluate import EvalRow, _target_hit, evaluate_monitor, to_markdown
from mlwatch.exporter import Exporter, create_app
from mlwatch.monitor import run_window
from mlwatch.policy import Finding, MonitorState
from mlwatch.schema import captures_to_frame
from mlwatch.simulate import SCENARIOS, WindowSpec, generate_window, model_score, simulate_dataset

from .conftest import POLICY_PATH, make_window

START = datetime(2026, 4, 1, tzinfo=UTC)


def test_generate_window_is_deterministic_and_scenarios_bite() -> None:
    a, la = generate_window(WindowSpec("none", 0.0, 200, 5, START))
    b, _ = generate_window(WindowSpec("none", 0.0, 200, 5, START))
    assert [r.model_dump() for r in a] == [r.model_dump() for r in b]
    assert len(la) == 200 and la[0].label_ts is not None and la[0].label_ts > a[0].ts
    frame = captures_to_frame(a)
    assert frame["ts"].is_monotonic_increasing and set(frame["decision"]) <= {"approve", "decline"}
    missing, _ = generate_window(WindowSpec("quality_missing", 0.3, 2000, 5, START))
    assert 0.25 < captures_to_frame(missing)["x_income"].isna().mean() < 0.35
    unseen, _ = generate_window(WindowSpec("quality_unseen", 0.1, 2000, 5, START))
    assert 0.07 < (captures_to_frame(unseen)["c_region"] == "TAS").mean() < 0.13
    shifted, _ = generate_window(WindowSpec("score_shift", 0.5, 2000, 5, START))
    clean, _ = generate_window(WindowSpec("none", 0.0, 2000, 5, START))
    assert captures_to_frame(shifted)["score"].mean() == pytest.approx(
        captures_to_frame(clean)["score"].mean() * 0.5, rel=0.02
    )
    scores = model_score(captures_to_frame(clean))
    assert np.all((scores > 0) & (scores < 1))


def test_simulate_dataset_writes_files(tmp_path: Path) -> None:
    manifest = simulate_dataset(
        tmp_path / "sim",
        scenario="covariate_shift",
        magnitude=0.5,
        windows=2,
        n_baseline=300,
        n_window=100,
        seed=1,
    )
    assert (
        manifest.windows == ["window_01.jsonl", "window_02.jsonl"]
        and manifest.target == "x_utilisation"
    )
    for name in ("baseline.jsonl", "labels.jsonl", "manifest.json", *manifest.windows):
        assert (tmp_path / "sim" / name).is_file()
    labels = (tmp_path / "sim" / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(labels) == 300 + 2 * 100
    assert (
        json.loads((tmp_path / "sim" / "manifest.json").read_text(encoding="utf-8"))["scenario"]
        == "covariate_shift"
    )


def test_evaluate_monitor_small_grid(policy: AlertPolicy) -> None:
    rows = evaluate_monitor(
        policy,
        grid={"none": [0.0], "covariate_shift": [1.0]},
        seeds=2,
        n_baseline=1500,
        n_window=600,
        n_boot=10,
    )
    assert [r.scenario for r in rows] == ["none", "covariate_shift"]
    assert (
        rows[0].flagged_rate <= 0.5
        and rows[1].target_hit_rate == 1.0
        and rows[1].mean_target_psi > 0.25
    )
    table = to_markdown(rows)
    assert table.startswith("| Scenario") and "covariate_shift" in table
    no_labels = evaluate_monitor(
        policy,
        grid={"prior_shift": [1.0]},
        seeds=1,
        n_baseline=1500,
        n_window=600,
        n_boot=10,
        with_labels=False,
    )
    assert no_labels[0].mean_auc_drop != no_labels[0].mean_auc_drop  # nan without labels
    assert set(rows[0].to_dict()) >= {"scenario", "flagged_rate", "target_hit_rate"}
    assert isinstance(rows[0], EvalRow)


def test_target_hit_logic() -> None:
    findings = [
        Finding("data", "x_utilisation", "WARN", "m"),
        Finding("performance", "auc", "OK", "m"),
    ]
    assert (
        _target_hit(findings, "x_utilisation")
        and not _target_hit(findings, "performance")
        and not _target_hit(findings, "")
    )
    assert _target_hit([Finding("prediction", "score", "CRITICAL", "m")], "score")
    assert _target_hit([Finding("performance", "auc", "CRITICAL", "m")], "performance")


def test_exporter_and_app(baseline: Baseline, policy: AlertPolicy, tmp_path: Path) -> None:
    frame, labels = make_window("covariate_shift", 1.0, seed=21)
    report = run_window(baseline, frame, labels, policy, MonitorState(), window_id="w", n_boot=20)
    exporter = Exporter()
    exporter.update(report)
    text = exporter.render().decode()
    assert (
        'mlwatch_feature_psi{feature="x_utilisation"}' in text
        and "mlwatch_window_status 2.0" in text
    )
    assert 'mlwatch_runs_total{status="CRITICAL"} 1.0' in text

    root = tmp_path / "data"
    root.mkdir()
    baseline.save(root / "baseline.json")
    (root / "policy.yaml").write_text(POLICY_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    simulate_dataset(
        root / "sim",
        scenario="none",
        magnitude=0.0,
        windows=1,
        n_baseline=100,
        n_window=400,
        seed=3,
    )
    (tmp_path / "outside.jsonl").write_text("")
    app = create_app(
        root / "baseline.json",
        root / "policy.yaml",
        state_path=root / "state.json",
        allowed_root=root,
    )
    with TestClient(app) as client:
        assert client.get("/health").json()["baseline_model_version"] == "v1"
        assert client.get("/report").status_code == 404
        assert client.get("/metrics").status_code == 200
        r = client.post(
            "/run",
            json={
                "capture_path": str(root / "sim" / "window_01.jsonl"),
                "labels_path": str(root / "sim" / "labels.jsonl"),
                "n_boot": 10,
            },
        )
        assert r.status_code == 200 and r.json()["window_id"] == "window_01"
        assert client.get("/report").json()["n_rows"] == 400
        metrics = client.get("/metrics").text
        assert "mlwatch_window_rows 400.0" in metrics
        assert (
            client.post("/run", json={"capture_path": str(tmp_path / "outside.jsonl")}).status_code
            == 400
        )
        assert (
            client.post("/run", json={"capture_path": str(root / "missing.jsonl")}).status_code
            == 404
        )
    assert json.loads((root / "state.json").read_text(encoding="utf-8"))["windows_seen"] == 1


def test_cli_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sim = tmp_path / "sim"
    assert (
        cli.main(
            [
                "simulate",
                "--scenario",
                "covariate_shift",
                "--magnitude",
                "1.0",
                "--windows",
                "2",
                "--n-baseline",
                "2000",
                "--n-window",
                "600",
                "--out",
                str(sim),
            ]
        )
        == 0
    )
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["windows"] == ["window_01.jsonl", "window_02.jsonl"]
    rc = cli.main(
        [
            "baseline",
            "--capture",
            str(sim / "baseline.jsonl"),
            "--labels",
            str(sim / "labels.jsonl"),
            "--out",
            str(sim / "baseline.json"),
        ]
    )
    assert rc == 0 and Baseline.load(sim / "baseline.json").performance is not None
    common = [
        "run",
        "--baseline",
        str(sim / "baseline.json"),
        "--labels",
        str(sim / "labels.jsonl"),
        "--policy",
        str(POLICY_PATH),
        "--state",
        str(sim / "state.json"),
        "--n-boot",
        "10",
    ]
    first = cli.main(
        [
            *common,
            "--capture",
            str(sim / "window_01.jsonl"),
            "--out",
            str(sim / "r1"),
            "--format",
            "json",
        ]
    )
    second = cli.main(
        [*common, "--capture", str(sim / "window_02.jsonl"), "--out", str(sim / "r2")]
    )
    assert (
        first in (2, 3) and second == 3
    )  # sustained CRITICAL drift → retrain on the second window
    assert (sim / "r2" / "report.md").is_file() and json.loads(
        (sim / "state.json").read_text(encoding="utf-8")
    )["windows_seen"] == 2
    capsys.readouterr()
    as_of = cli.main(
        [
            *common,
            "--capture",
            str(sim / "window_01.jsonl"),
            "--as-of",
            "2026-01-01T00:00:00+00:00",
            "--window-id",
            "early",
        ]
    )
    assert (
        as_of in (0, 2, 3) and "not evaluated" in capsys.readouterr().out
    )  # labels not yet observed
    quiet = tmp_path / "quiet"
    cli.main(
        [
            "simulate",
            "--scenario",
            "none",
            "--windows",
            "1",
            "--n-baseline",
            "2000",
            "--n-window",
            "600",
            "--out",
            str(quiet),
        ]
    )
    cli.main(
        [
            "baseline",
            "--capture",
            str(quiet / "baseline.jsonl"),
            "--out",
            str(quiet / "baseline.json"),
            "--features",
            "x_income,x_utilisation",
        ]
    )
    capsys.readouterr()
    rc = cli.main(
        [
            "run",
            "--baseline",
            str(quiet / "baseline.json"),
            "--capture",
            str(quiet / "window_01.jsonl"),
            "--policy",
            str(POLICY_PATH),
            "--n-boot",
            "10",
        ]
    )
    assert rc == 0
    rc = cli.main(
        [
            "evaluate",
            "--policy",
            str(POLICY_PATH),
            "--scenario",
            "none",
            "--seeds",
            "1",
            "--n-baseline",
            "1000",
            "--n-window",
            "400",
            "--n-boot",
            "10",
            "--out",
            str(tmp_path / "eval.md"),
        ]
    )
    assert rc == 0 and (tmp_path / "eval.json").is_file()

    captured: dict[str, Any] = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: captured.update(kw, app=app))
    assert (
        cli.main(
            [
                "serve",
                "--baseline",
                str(sim / "baseline.json"),
                "--policy",
                str(POLICY_PATH),
                "--port",
                "9999",
                "--allowed-root",
                str(sim),
            ]
        )
        == 0
    )
    assert captured["port"] == 9999 and captured["app"].title == "mlwatch exporter"
    assert len(SCENARIOS) == 7
