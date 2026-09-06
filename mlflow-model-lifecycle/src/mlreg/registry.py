"""Model Registry operations built on aliases (`@champion`, `@challenger`, `@previous`).

Aliases are the MLflow 2.9+/3.x replacement for stages: a mutable pointer per model name,
so deployments load `models:/credit-pd@champion` and promotion is one alias move that is
recorded on the version's tags with who approved it and which gate report backed it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

CHAMPION = "champion"
CHALLENGER = "challenger"
PREVIOUS = "previous"

_NOT_FOUND = {"RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"}


@dataclass
class RegisteredVersion:
    name: str
    version: int
    run_id: str | None
    source: str
    aliases: list[str] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def uri(self) -> str:
        return f"models:/{self.name}/{self.version}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "run_id": self.run_id,
            "source": self.source,
            "aliases": list(self.aliases),
            "tags": dict(self.tags),
            "uri": self.uri,
        }


def _wrap(mv: Any) -> RegisteredVersion:
    return RegisteredVersion(
        name=str(mv.name),
        version=int(mv.version),
        run_id=str(mv.run_id) if mv.run_id else None,
        source=str(mv.source),
        aliases=[str(a) for a in (mv.aliases or [])],
        tags={str(k): str(v) for k, v in (mv.tags or {}).items()},
    )


def register(model_uri: str, name: str, *, tags: dict[str, str] | None = None) -> RegisteredVersion:
    """Create a new registry version from a logged model and stamp lineage tags on it."""
    mv = mlflow.register_model(model_uri, name, tags=tags or {})
    client = MlflowClient()
    return _wrap(client.get_model_version(name, mv.version))


def get_version(name: str, version: int) -> RegisteredVersion:
    return _wrap(MlflowClient().get_model_version(name, str(version)))


def get_alias(name: str, alias: str) -> RegisteredVersion | None:
    try:
        return _wrap(MlflowClient().get_model_version_by_alias(name, alias))
    except MlflowException as exc:
        if exc.error_code in _NOT_FOUND:
            return None
        raise


def set_alias(name: str, alias: str, version: int) -> None:
    MlflowClient().set_registered_model_alias(name, alias, str(version))


def delete_alias(name: str, alias: str) -> None:
    try:
        MlflowClient().delete_registered_model_alias(name, alias)
    except MlflowException as exc:  # pragma: no cover - backend dependent
        if exc.error_code not in _NOT_FOUND:
            raise


def set_tags(name: str, version: int, tags: dict[str, str]) -> None:
    client = MlflowClient()
    for k, v in tags.items():
        client.set_model_version_tag(name, str(version), k, v)


def list_versions(name: str) -> list[RegisteredVersion]:
    client = MlflowClient()
    try:
        found = client.search_model_versions(f"name = '{name}'")
    except MlflowException as exc:
        if exc.error_code in _NOT_FOUND:
            return []
        raise
    versions = [_wrap(client.get_model_version(name, mv.version)) for mv in found]
    return sorted(versions, key=lambda v: v.version)


def promote(
    name: str,
    version: int,
    *,
    approved_by: str,
    evidence: dict[str, str] | None = None,
    alias: str = CHAMPION,
) -> tuple[RegisteredVersion, RegisteredVersion | None]:
    """Move `alias` to `version`; the displaced version gets `@previous` and a retirement tag."""
    now = datetime.now(UTC).isoformat(timespec="seconds")
    current = get_alias(name, alias)
    if current is not None and current.version == version:
        return current, None
    if current is not None:
        set_alias(name, PREVIOUS, current.version)
        set_tags(name, current.version, {"retired_utc": now, "retired_by_version": str(version)})
    set_alias(name, alias, version)
    tags = {"promoted_utc": now, "promoted_by": approved_by, "promotion_alias": alias}
    tags.update(evidence or {})
    set_tags(name, version, tags)
    challenger = get_alias(name, CHALLENGER)
    if challenger is not None and challenger.version == version:
        delete_alias(name, CHALLENGER)
    return get_version(name, version), current


def load_by_alias(name: str, alias: str) -> Any:
    return mlflow.pyfunc.load_model(f"models:/{name}@{alias}")


def load_version(name: str, version: int) -> Any:
    return mlflow.pyfunc.load_model(f"models:/{name}/{version}")
