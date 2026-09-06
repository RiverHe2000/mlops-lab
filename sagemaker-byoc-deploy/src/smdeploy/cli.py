"""Command line for the whole path: local contract checks, image, training job, registry,
release, rollback, teardown. `plan` renders the exact AWS requests without calling AWS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .aws.clients import AwsClients
from .aws.ecr import build_and_push, ensure_repository, login_command
from .aws.endpoint import EndpointDeployer
from .aws.operations import describe_autoscaling
from .aws.registry import (
    ApprovalStatus,
    RegistrationSpec,
    describe_package,
    latest_package,
    register_model_package,
    set_approval,
)
from .aws.training_job import (
    TrainingJobSpec,
    start_training_job,
    training_outputs,
    wait_for_training_job,
)
from .hyperparams import TrainingHyperparameters
from .local_run import simulate_invocation, simulate_training
from .model_io import load_metadata
from .release import ReleaseError, StageConfig, load_yaml, release


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _clients(args: argparse.Namespace) -> AwsClients:
    return AwsClients.from_session(region=args.region, profile=args.profile)


# --- local ---------------------------------------------------------------------------------
def cmd_local_train(args: argparse.Namespace) -> int:
    hp: dict[str, Any] = (
        json.loads(Path(args.hyperparameters).read_text(encoding="utf-8"))
        if args.hyperparameters
        else {}
    )
    report = simulate_training(
        Path(args.base_dir),
        train_csv=Path(args.train_csv),
        hyperparameters=TrainingHyperparameters.from_sagemaker_dict(hp),
        validation_csv=Path(args.validation_csv) if args.validation_csv else None,
    )
    _print(
        {
            "model_dir": str(report.model_dir),
            "metrics_on": report.metrics_on,
            "metrics": report.metrics,
        }
    )
    return 0


def cmd_local_invoke(args: argparse.Namespace) -> int:
    body = Path(args.payload).read_bytes()
    status, content, headers = simulate_invocation(
        Path(args.model_dir), body, content_type=args.content_type, accept=args.accept
    )
    print(f"HTTP {status} {headers.get('content-type', '')}")
    print(content.decode("utf-8", errors="replace"))
    return 0 if status == 200 else 1


def cmd_plan(args: argparse.Namespace) -> int:
    """Render the requests the other commands would send, for review in a pull request."""
    plan: dict[str, Any] = {}
    if args.training:
        plan["create_training_job"] = TrainingJobSpec.model_validate(
            load_yaml(Path(args.training))
        ).to_request()
    if args.stage:
        stage = StageConfig.from_yaml(Path(args.stage))
        endpoint = {
            **stage.endpoint,
            "image_uri": "<from model package>",
            "model_data_url": "s3://<from model package>",
        }
        plan["stage"] = {
            "stage": stage.stage,
            "endpoint": endpoint,
            "deployment_config": stage.rollout.to_deployment_config(),
            "alarms": stage.alarms.model_dump() if stage.alarms else None,
            "autoscaling": stage.autoscaling.model_dump() if stage.autoscaling else None,
        }
    _print(plan)
    return 0


# --- image ---------------------------------------------------------------------------------
def cmd_push(args: argparse.Namespace) -> int:
    clients = _clients(args)
    repo = ensure_repository(clients.ecr, args.repository)
    login = login_command(clients.ecr)
    if args.dry_run:
        _print({"repository": repo.uri, "tags": args.tag, "login": login[0]})
        return 0
    executed = build_and_push(
        repo, args.tag, context=Path(args.context), dockerfile=Path(args.dockerfile), login=login
    )
    _print({"image": repo.image_uri(args.tag[0]), "commands": [" ".join(c) for c in executed]})
    return 0


# --- training job --------------------------------------------------------------------------
def cmd_train_job(args: argparse.Namespace) -> int:
    clients = _clients(args)
    spec = TrainingJobSpec.model_validate(load_yaml(Path(args.config)))
    if args.job_name:
        spec = spec.model_copy(update={"job_name": args.job_name})
    if args.image_uri:
        spec = spec.model_copy(update={"image_uri": args.image_uri})
    arn = start_training_job(clients.sagemaker, spec)
    print(f"started {arn}")
    if args.no_wait:
        return 0
    desc = wait_for_training_job(clients.sagemaker, spec.job_name, poll_s=args.poll_seconds)
    outputs = training_outputs(desc)
    _print(outputs.__dict__)
    if args.out:
        Path(args.out).write_text(
            json.dumps(outputs.__dict__, indent=2, default=str), encoding="utf-8"
        )
    return 0


# --- registry ------------------------------------------------------------------------------
def cmd_register(args: argparse.Namespace) -> int:
    clients = _clients(args)
    metrics: dict[str, float] = {}
    if args.metrics_json:
        raw = json.loads(Path(args.metrics_json).read_text(encoding="utf-8"))
        metrics = {
            str(k): float(v)
            for k, v in (raw.get("metrics", raw)).items()
            if isinstance(v, int | float)
        }
    if (
        args.min_auc is not None
        and metrics.get("validation:auc", metrics.get("auc", 0.0)) < args.min_auc
    ):
        print(f"validation AUC {metrics} below --min-auc {args.min_auc}; not registering")
        return 2
    spec = RegistrationSpec(
        group=args.group,
        image_uri=args.image_uri,
        model_data_url=args.model_data_url,
        metrics=metrics,
        evaluation_s3_uri=args.evaluation_s3_uri,
        description=args.description,
        approval_status="Approved" if args.approve else "PendingManualApproval",
        customer_metadata=dict(kv.split("=", 1) for kv in args.metadata),
    )
    info = register_model_package(clients.sagemaker, spec)
    _print(info.__dict__)
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    clients = _clients(args)
    status: ApprovalStatus = "Rejected" if args.reject else "Approved"
    info = set_approval(clients.sagemaker, args.package_arn, status, note=f"{args.by}: {args.note}")
    _print(info.__dict__)
    return 0


def cmd_latest(args: argparse.Namespace) -> int:
    clients = _clients(args)
    info = latest_package(
        clients.sagemaker, args.group, approval=None if args.any_status else "Approved"
    )
    if info is None:
        print("no matching model package")
        return 1
    _print(info.__dict__)
    return 0


# --- release -------------------------------------------------------------------------------
def cmd_deploy(args: argparse.Namespace) -> int:
    clients = _clients(args)
    stage = StageConfig.from_yaml(Path(args.stage))
    package = (
        describe_package(clients.sagemaker, args.package_arn)
        if args.package_arn
        else latest_package(
            clients.sagemaker,
            stage.model_package_group,
            approval="Approved" if stage.require_approved else None,
        )
    )
    if package is None:
        print(f"no eligible package in {stage.model_package_group}")
        return 1
    rows = pd.read_csv(args.smoke_csv)
    columns = _feature_columns(args, rows)
    try:
        report = release(clients, stage, package, rows, columns, wait_timeout_s=args.wait_timeout)
    except ReleaseError as exc:
        print(f"RELEASE FAILED: {exc}")
        _print(exc.report.to_dict())
        return 2
    _print(report.to_dict())
    if args.out:
        Path(args.out).write_text(
            json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8"
        )
    return 0


def _feature_columns(args: argparse.Namespace, rows: pd.DataFrame) -> list[str]:
    if args.model_dir:
        return list(load_metadata(Path(args.model_dir)).feature_columns)
    drop = set(args.drop_columns or [])
    return [str(c) for c in rows.columns if c not in drop]


def cmd_rollback(args: argparse.Namespace) -> int:
    clients = _clients(args)
    deployer = EndpointDeployer(clients.sagemaker)
    desc = deployer.rollback(args.endpoint_name, args.to_config)
    _print(
        {
            "endpoint": args.endpoint_name,
            "config": desc.get("EndpointConfigName"),
            "status": desc.get("EndpointStatus"),
        }
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    clients = _clients(args)
    desc = EndpointDeployer(clients.sagemaker).describe(args.endpoint_name)
    if desc is None:
        print(f"{args.endpoint_name}: not found")
        return 1
    out: dict[str, Any] = {
        "endpoint": args.endpoint_name,
        "status": desc.get("EndpointStatus"),
        "config": desc.get("EndpointConfigName"),
        "variants": desc.get("ProductionVariants"),
        "failure_reason": desc.get("FailureReason"),
    }
    if args.autoscaling:
        out["autoscaling"] = describe_autoscaling(
            clients.application_autoscaling, args.endpoint_name, args.variant
        )
    _print(out)
    return 0


def cmd_teardown(args: argparse.Namespace) -> int:
    clients = _clients(args)
    removed = EndpointDeployer(clients.sagemaker).delete(
        args.endpoint_name, delete_configs=True, delete_models=True
    )
    _print({"removed": removed})
    return 0


# --- parser --------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="smdeploy", description="SageMaker BYOC: build, train, register, release."
    )
    sub = ap.add_subparsers(dest="command", required=True)

    def aws(p: argparse.ArgumentParser) -> None:
        p.add_argument("--region", default=None)
        p.add_argument("--profile", default=None)

    p = sub.add_parser("local-train", help="run the training contract in a local /opt/ml layout")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--validation-csv", default=None)
    p.add_argument(
        "--hyperparameters", default=None, help="JSON file (values as SageMaker would pass them)"
    )
    p.add_argument("--base-dir", default="runs/opt-ml")
    p.set_defaults(func=cmd_local_train)

    p = sub.add_parser("local-invoke", help="POST a payload to the hosting app in-process")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--payload", required=True)
    p.add_argument("--content-type", default="text/csv")
    p.add_argument("--accept", default="application/json")
    p.set_defaults(func=cmd_local_invoke)

    p = sub.add_parser("plan", help="print the AWS requests a config would produce")
    p.add_argument("--training", default=None)
    p.add_argument("--stage", default=None)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("push", help="ensure the ECR repository, build and push the image")
    aws(p)
    p.add_argument("--repository", required=True)
    p.add_argument(
        "--tag",
        action="append",
        required=True,
        help="repeatable; the first is built, the rest re-tagged",
    )
    p.add_argument("--context", default=".")
    p.add_argument("--dockerfile", default="container/Dockerfile")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_push)

    p = sub.add_parser("train-job", help="start a SageMaker training job from a YAML spec and wait")
    aws(p)
    p.add_argument("--config", required=True)
    p.add_argument("--job-name", default=None)
    p.add_argument("--image-uri", default=None)
    p.add_argument("--no-wait", action="store_true")
    p.add_argument("--poll-seconds", type=float, default=30.0)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_train_job)

    p = sub.add_parser(
        "register", help="create a model package version (pending approval by default)"
    )
    aws(p)
    p.add_argument("--group", required=True)
    p.add_argument("--image-uri", required=True)
    p.add_argument("--model-data-url", required=True)
    p.add_argument("--metrics-json", default=None)
    p.add_argument("--evaluation-s3-uri", default=None)
    p.add_argument(
        "--min-auc", type=float, default=None, help="refuse to register below this validation AUC"
    )
    p.add_argument("--description", default="")
    p.add_argument("--approve", action="store_true")
    p.add_argument("--metadata", action="append", default=[], help="key=value (repeatable)")
    p.set_defaults(func=cmd_register)

    p = sub.add_parser("approve", help="approve or reject a model package")
    aws(p)
    p.add_argument("--package-arn", required=True)
    p.add_argument("--by", required=True)
    p.add_argument("--note", default="")
    p.add_argument("--reject", action="store_true")
    p.set_defaults(func=cmd_approve)

    p = sub.add_parser("latest", help="show the newest (approved) package of a group")
    aws(p)
    p.add_argument("--group", required=True)
    p.add_argument("--any-status", action="store_true")
    p.set_defaults(func=cmd_latest)

    p = sub.add_parser(
        "deploy", help="release a package to a stage: alarms, blue/green, smoke, autoscaling"
    )
    aws(p)
    p.add_argument("--stage", required=True, help="stage YAML (configs/endpoint.*.yaml)")
    p.add_argument("--package-arn", default=None, help="default: latest approved in the group")
    p.add_argument("--smoke-csv", required=True)
    p.add_argument("--model-dir", default=None, help="local model dir to read feature columns from")
    p.add_argument(
        "--drop-columns", nargs="*", default=None, help="columns to drop from the smoke CSV"
    )
    p.add_argument("--wait-timeout", type=float, default=3600.0)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_deploy)

    p = sub.add_parser("rollback", help="update the endpoint back to a previous config")
    aws(p)
    p.add_argument("--endpoint-name", required=True)
    p.add_argument("--to-config", required=True)
    p.set_defaults(func=cmd_rollback)

    p = sub.add_parser("status", help="endpoint status")
    aws(p)
    p.add_argument("--endpoint-name", required=True)
    p.add_argument("--autoscaling", action="store_true")
    p.add_argument("--variant", default="AllTraffic")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("teardown", help="delete endpoint, its config and models")
    aws(p)
    p.add_argument("--endpoint-name", required=True)
    p.set_defaults(func=cmd_teardown)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
