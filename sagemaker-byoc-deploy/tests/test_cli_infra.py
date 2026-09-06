from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import cfnlint.api
import pandas as pd
import pytest
import yaml
from cfnlint.config import ManualArgs
from dockerfile_parse import DockerfileParser

from smdeploy import cli
from smdeploy.aws.clients import AwsClients
from smdeploy.aws.endpoint import EndpointDeployer
from smdeploy.aws.operations import SmokeResult
from smdeploy.aws.registry import RegistrationSpec, register_model_package
from smdeploy.synthetic import FEATURES, TARGET
from smdeploy.training import TrainingReport

from .conftest import ROLE, ROOT

IMAGE = "123456789012.dkr.ecr.ap-southeast-2.amazonaws.com/credit-pd:sha1"
ENV = {
    "PROJECT_NAME": "credit-pd",
    "SAGEMAKER_EXECUTION_ROLE_ARN": ROLE,
    "ARTIFACT_BUCKET": "bkt",
    "IMAGE_URI": IMAGE,
}


@pytest.fixture
def cli_aws(aws: AwsClients, monkeypatch: pytest.MonkeyPatch) -> AwsClients:
    monkeypatch.setattr(AwsClients, "from_session", classmethod(lambda _cls, **_kw: aws))
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    return aws


def test_local_train_and_invoke(
    tmp_path: Path, csv_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    hp_file = tmp_path / "hp.json"
    hp_file.write_text(json.dumps({"target": TARGET, "id_columns": "application_id", "C": "0.5"}))
    rc = cli.main(
        [
            "local-train",
            "--train-csv",
            str(csv_dir / "train.csv"),
            "--validation-csv",
            str(csv_dir / "validation.csv"),
            "--hyperparameters",
            str(hp_file),
            "--base-dir",
            str(tmp_path / "ml"),
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["metrics_on"] == "validation" and out["metrics"]["auc"] > 0.6
    model_dir = str(tmp_path / "ml" / "model")
    smoke = str(csv_dir / "smoke.csv")
    rc = cli.main(
        [
            "local-invoke",
            "--model-dir",
            model_dir,
            "--payload",
            smoke,
            "--content-type",
            "text/csv; header=present",
            "--accept",
            "application/json",
        ]
    )
    assert rc == 0 and "predictions" in capsys.readouterr().out
    rc = cli.main(
        [
            "local-invoke",
            "--model-dir",
            model_dir,
            "--payload",
            smoke,
            "--content-type",
            "application/xml",
        ]
    )
    assert rc == 1


def test_plan_renders_requests(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("TRAINING_JOB_NAME", "credit-pd-plan")
    training = str(ROOT / "configs" / "training.yaml")
    stage = str(ROOT / "configs" / "endpoint.prod.yaml")
    assert cli.main(["plan", "--training", training, "--stage", stage]) == 0
    plan = json.loads(capsys.readouterr().out)
    job = plan["create_training_job"]
    assert job["TrainingJobName"] == "credit-pd-plan" and job["EnableManagedSpotTraining"] is True
    assert (
        job["HyperParameters"]["C"] == "0.5"
        and job["AlgorithmSpecification"]["TrainingImage"] == IMAGE
    )
    routing = plan["stage"]["deployment_config"]["BlueGreenUpdatePolicy"][
        "TrafficRoutingConfiguration"
    ]
    assert routing["Type"] == "CANARY"
    assert plan["stage"]["autoscaling"]["max_capacity"] == 6


def test_push_dry_run_and_registry_commands(
    cli_aws: AwsClients, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        cli.main(
            ["push", "--repository", "credit-pd", "--tag", "sha1", "--tag", "run-1", "--dry-run"]
        )
        == 0
    )
    out = json.loads(capsys.readouterr().out)
    assert out["repository"].endswith("/credit-pd") and out["tags"] == ["sha1", "run-1"]
    assert out["login"][0] == "docker"

    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text(json.dumps({"metrics": {"validation:auc": 0.65}}))
    args = [
        "register",
        "--group",
        "credit-pd",
        "--image-uri",
        IMAGE,
        "--model-data-url",
        "s3://b/m.tar.gz",
        "--metrics-json",
        str(metrics_file),
        "--min-auc",
        "0.7",
    ]
    assert cli.main(args) == 2  # below the bar → not registered
    capsys.readouterr()
    metrics_file.write_text(json.dumps({"metrics": {"validation:auc": 0.81}}))
    assert cli.main([*args, "--metadata", "git_sha=abc"]) == 0
    registered = json.loads(capsys.readouterr().out)
    assert registered["approval_status"] == "PendingManualApproval"
    assert registered["metadata"]["git_sha"] == "abc"
    assert registered["metadata"]["metric_validation:auc"] == "0.810000"

    assert cli.main(["latest", "--group", "credit-pd"]) == 1  # nothing approved yet
    capsys.readouterr()
    assert cli.main(["latest", "--group", "credit-pd", "--any-status"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["arn"] == registered["arn"]
    assert (
        cli.main(["approve", "--package-arn", info["arn"], "--by", "reviewer", "--note", "ok"]) == 0
    )
    assert json.loads(capsys.readouterr().out)["approval_status"] == "Approved"
    assert cli.main(["latest", "--group", "credit-pd"]) == 0
    capsys.readouterr()
    assert cli.main(["approve", "--package-arn", info["arn"], "--by", "reviewer", "--reject"]) == 0
    assert json.loads(capsys.readouterr().out)["approval_status"] == "Rejected"


def test_deploy_status_rollback_teardown(
    cli_aws: AwsClients,
    tmp_path: Path,
    csv_dir: Path,
    trained: TrainingReport,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    package = register_model_package(
        cli_aws.sagemaker,
        RegistrationSpec(
            group="credit-pd",
            image_uri=IMAGE,
            model_data_url="s3://b/m.tar.gz",
            approval_status="Approved",
        ),
    )
    stage_file = str(ROOT / "configs" / "endpoint.staging.yaml")
    smoke_csv = str(csv_dir / "smoke.csv")
    # moto's runtime returns an opaque body → the smoke test fails; a freshly created endpoint has nothing to roll back to
    rc = cli.main(
        [
            "deploy",
            "--stage",
            stage_file,
            "--smoke-csv",
            smoke_csv,
            "--model-dir",
            str(trained.model_dir),
        ]
    )
    assert rc == 2 and "RELEASE FAILED" in capsys.readouterr().out
    # with a passing smoke test the release succeeds and the report is written
    monkeypatch.setattr(
        "smdeploy.release.smoke_test", lambda *_a, **_k: SmokeResult(True, 200, 5.0, 5, "ok")
    )
    out_file = tmp_path / "release.json"
    rc = cli.main(
        [
            "deploy",
            "--stage",
            stage_file,
            "--package-arn",
            package.arn,
            "--smoke-csv",
            smoke_csv,
            "--out",
            str(out_file),
        ]
    )
    assert rc == 0
    report = json.loads(out_file.read_text())
    assert report["deploy"]["action"] == "unchanged" and report["stage"] == "staging"
    capsys.readouterr()

    assert cli.main(["status", "--endpoint-name", "credit-pd-staging"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "InService"
    assert cli.main(["status", "--endpoint-name", "nope"]) == 1
    capsys.readouterr()
    assert cli.main(["teardown", "--endpoint-name", "credit-pd-staging"]) == 0
    assert len(json.loads(capsys.readouterr().out)["removed"]) == 3
    assert cli.main(["latest", "--group", "missing-group"]) == 1


def test_rollback_command(
    cli_aws: AwsClients, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_rollback(
        self: EndpointDeployer, endpoint_name: str, to_config_name: str, *, timeout_s: float = 0
    ) -> dict[str, Any]:
        calls.append((endpoint_name, to_config_name))
        return {"EndpointConfigName": to_config_name, "EndpointStatus": "InService"}

    monkeypatch.setattr(EndpointDeployer, "rollback", fake_rollback)
    assert cli.main(["rollback", "--endpoint-name", "e", "--to-config", "e-old"]) == 0
    assert calls == [("e", "e-old")] and json.loads(capsys.readouterr().out)["config"] == "e-old"


# --- infrastructure and packaging files -------------------------------------------------------
def test_cloudformation_template_lints_clean() -> None:
    template = ROOT / "infra" / "cloudformation" / "platform.yaml"
    matches = cfnlint.api.lint(
        template.read_text(encoding="utf-8"),
        config=ManualArgs(regions=["ap-southeast-2", "us-east-1"]),
    )
    assert matches == [], chr(10).join(str(m) for m in matches)

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_multi_constructor("!", lambda _ldr, _suffix, _node: None)
    doc = yaml.load(template.read_text(encoding="utf-8"), Loader=Loader)
    resources = doc["Resources"]
    expected = {
        "ArtifactBucket",
        "ImageRepository",
        "ModelPackageGroup",
        "SageMakerExecutionRole",
        "GitHubOidcProvider",
        "GitHubDeployRole",
    }
    assert expected <= set(resources)
    assert resources["ImageRepository"]["Properties"]["ImageTagMutability"] == "IMMUTABLE"
    assert (
        resources["ArtifactBucket"]["Properties"]["PublicAccessBlockConfiguration"][
            "BlockPublicPolicy"
        ]
        is True
    )
    assert set(doc["Outputs"]) >= {
        "ArtifactBucketName",
        "ImageRepositoryUri",
        "SageMakerExecutionRoleArn",
        "GitHubDeployRoleArn",
    }


def test_dockerfile_follows_the_byoc_contract() -> None:
    parser = DockerfileParser(path=str(ROOT / "container" / "Dockerfile"))
    instructions = [(i["instruction"], i["value"]) for i in parser.structure]
    froms = [v for k, v in instructions if k == "FROM"]
    assert froms and all(":latest" not in f and ":" in f.split(" ")[0] for f in froms), froms
    assert any(k == "USER" and v.strip() == "sagemaker" for k, v in instructions)
    assert any(k == "EXPOSE" and v.strip() == "8080" for k, v in instructions)
    assert any(
        k == "COPY" and "container/train" in v and "container/serve" in v for k, v in instructions
    )
    assert any(k == "HEALTHCHECK" for k, _ in instructions)
    assert not any(
        k == "ENTRYPOINT" for k, _ in instructions
    )  # SageMaker passes `train` / `serve` as the command
    assert any(k == "CMD" and "serve" in v for k, v in instructions)
    for script, target in (("train", "smdeploy-train"), ("serve", "smdeploy-serve")):
        text = (ROOT / "container" / script).read_text(encoding="utf-8")
        assert text.startswith("#!/bin/sh") and f"exec {target}" in text


def workflow_path(name: str) -> Path:
    """Locate a workflow whether the project is checked out standalone or as a directory of
    the `mlops-lab` repository, where GitHub requires the file at the repository root."""
    candidates = (
        ROOT / ".github" / "workflows" / name,
        ROOT.parent / ".github" / "workflows" / f"{ROOT.name}-{name}",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    msg = f"workflow {name} not found in {[str(c) for c in candidates]}"
    raise FileNotFoundError(msg)


def test_workflows_use_oidc_and_environments() -> None:
    cd_path = workflow_path("cd.yml")
    cd = yaml.safe_load(cd_path.read_text(encoding="utf-8"))
    assert cd["permissions"]["id-token"] == "write"
    text = cd_path.read_text(encoding="utf-8")
    assert "aws-access-key-id" not in text and "role-to-assume" in text
    jobs = cd["jobs"]
    assert jobs["deploy-staging"]["environment"] == "staging"
    assert jobs["approve"]["environment"] == "production"
    assert jobs["deploy-production"]["environment"] == "production"
    assert (
        "approve" in jobs["deploy-production"]["needs"]
        and "deploy-staging" in jobs["approve"]["needs"]
    )
    ci = yaml.safe_load(workflow_path("ci.yml").read_text(encoding="utf-8"))
    steps = " ".join(str(s.get("run", "")) for s in ci["jobs"]["container"]["steps"])
    assert "docker build" in steps and "smdeploy:ci train" in steps and "smdeploy:ci serve" in steps


def test_example_data_generator(tmp_path: Path) -> None:
    script = ROOT / "scripts" / "make_example_data.py"
    result = subprocess.run(
        [sys.executable, str(script), "--out", str(tmp_path), "--rows", "100"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    train = pd.read_csv(tmp_path / "credit_train.csv")
    smoke = pd.read_csv(tmp_path / "smoke_rows.csv")
    assert len(train) == 80 and smoke.columns.tolist() == FEATURES and len(smoke) == 5
    hp = json.loads((tmp_path / "hyperparameters.json").read_text())
    assert all(isinstance(v, str) for v in hp.values())
