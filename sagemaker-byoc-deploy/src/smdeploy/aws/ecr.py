"""ECR: repository hygiene and the image push handshake.

Docker itself is invoked through an injectable `runner`, so the command sequence is unit-tested
while the real push only happens where a Docker daemon exists (CI, a Linux box)."""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from botocore.exceptions import ClientError

from .clients import is_not_found

if TYPE_CHECKING:
    from mypy_boto3_ecr import ECRClient

Runner = Callable[[Sequence[str], str | None], None]


class DockerUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class EcrRepository:
    name: str
    uri: str
    arn: str

    @property
    def registry(self) -> str:
        return self.uri.split("/", 1)[0]

    def image_uri(self, tag: str) -> str:
        return f"{self.uri}:{tag}"


def lifecycle_policy(keep_last: int, untagged_days: int = 7) -> str:
    return json.dumps(
        {
            "rules": [
                {
                    "rulePriority": 1,
                    "description": f"expire untagged images after {untagged_days} days",
                    "selection": {
                        "tagStatus": "untagged",
                        "countType": "sinceImagePushed",
                        "countUnit": "days",
                        "countNumber": untagged_days,
                    },
                    "action": {"type": "expire"},
                },
                {
                    "rulePriority": 2,
                    "description": f"keep the last {keep_last} tagged images",
                    "selection": {
                        "tagStatus": "any",
                        "countType": "imageCountMoreThan",
                        "countNumber": keep_last,
                    },
                    "action": {"type": "expire"},
                },
            ]
        }
    )


def ensure_repository(
    ecr: ECRClient,
    name: str,
    *,
    scan_on_push: bool = True,
    immutable_tags: bool = True,
    keep_last: int = 20,
) -> EcrRepository:
    """Idempotent: describe, create if missing, always (re)apply the lifecycle policy."""
    try:
        desc = ecr.describe_repositories(repositoryNames=[name])["repositories"][0]
    except ClientError as exc:
        if not is_not_found(exc):
            raise
        desc = ecr.create_repository(
            repositoryName=name,
            imageScanningConfiguration={"scanOnPush": scan_on_push},
            imageTagMutability="IMMUTABLE" if immutable_tags else "MUTABLE",
            encryptionConfiguration={"encryptionType": "AES256"},
        )["repository"]
    ecr.put_lifecycle_policy(repositoryName=name, lifecyclePolicyText=lifecycle_policy(keep_last))
    return EcrRepository(
        name=str(desc["repositoryName"]),
        uri=str(desc["repositoryUri"]),
        arn=str(desc["repositoryArn"]),
    )


def login_command(ecr: ECRClient) -> tuple[list[str], str]:
    """`docker login` arguments plus the password to feed on stdin (never on the command line)."""
    data = ecr.get_authorization_token()["authorizationData"][0]
    token = base64.b64decode(str(data["authorizationToken"])).decode("utf-8")
    user, _, password = token.partition(":")
    endpoint = str(data["proxyEndpoint"]).removeprefix("https://")
    return ["docker", "login", "--username", user, "--password-stdin", endpoint], password


def default_runner(cmd: Sequence[str], stdin: str | None) -> None:
    if shutil.which(cmd[0]) is None:
        raise DockerUnavailableError(f"{cmd[0]!r} is not installed or not on PATH")
    subprocess.run(list(cmd), input=stdin, text=True, check=True)


def build_and_push(
    repo: EcrRepository,
    tags: Sequence[str],
    *,
    context: Path,
    dockerfile: Path,
    login: tuple[list[str], str] | None = None,
    runner: Runner = default_runner,
    platform: str = "linux/amd64",
) -> list[list[str]]:
    """Build once, tag for every requested tag, push each; returns the executed commands."""
    if not tags:
        raise ValueError("at least one tag is required")
    executed: list[list[str]] = []

    def run(cmd: list[str], stdin: str | None = None) -> None:
        runner(cmd, stdin)
        executed.append(cmd)

    if login is not None:
        run(login[0], login[1])
    primary = repo.image_uri(tags[0])
    run(
        [
            "docker",
            "build",
            "--platform",
            platform,
            "-t",
            primary,
            "-f",
            str(dockerfile),
            str(context),
        ]
    )
    for tag in tags[1:]:
        run(["docker", "tag", primary, repo.image_uri(tag)])
    for tag in tags:
        run(["docker", "push", repo.image_uri(tag)])
    return executed
