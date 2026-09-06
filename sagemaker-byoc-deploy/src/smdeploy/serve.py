"""The `serve` entrypoint: reads SageMaker's hosting environment variables and runs uvicorn.

SAGEMAKER_BIND_TO_PORT          port to listen on (8080 unless SageMaker says otherwise)
SAGEMAKER_MODEL_SERVER_WORKERS  worker processes (default: min(cpu count, 4))
SAGEMAKER_MAX_PAYLOAD_IN_MB     request size cap (6 MB real-time default)
SMDEPLOY_THRESHOLD              optional decision-threshold override
SMDEPLOY_BASE_DIR               /opt/ml unless simulating locally
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from .inference import ModelHandler
from .paths import SageMakerPaths
from .server import configure_logging, create_app


@dataclass(frozen=True)
class ServeSettings:
    model_dir: Path
    port: int = 8080
    workers: int = 1
    max_payload_mb: int = 6
    threshold: float | None = None
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ServeSettings:
        e = dict(os.environ) if env is None else dict(env)
        paths = SageMakerPaths.from_env(e)
        cpu = paths.num_cpus
        threshold = e.get("SMDEPLOY_THRESHOLD")
        return cls(
            model_dir=paths.model_dir,
            port=int(e.get("SAGEMAKER_BIND_TO_PORT", "8080")),
            workers=int(e.get("SAGEMAKER_MODEL_SERVER_WORKERS", str(min(cpu, 4)))),
            max_payload_mb=int(e.get("SAGEMAKER_MAX_PAYLOAD_IN_MB", "6")),
            threshold=float(threshold) if threshold else None,
            log_level=e.get("SMDEPLOY_LOG_LEVEL", "INFO"),
        )


def app_factory() -> FastAPI:
    """uvicorn factory (import string `smdeploy.serve:app_factory`) so multi-worker mode works."""
    settings = ServeSettings.from_env()
    configure_logging(settings.log_level)
    handler = ModelHandler(settings.model_dir, threshold=settings.threshold)
    return create_app(handler, max_payload_mb=settings.max_payload_mb, workers=settings.workers)


def main(argv: list[str] | None = None) -> int:
    del argv
    settings = ServeSettings.from_env()
    uvicorn.run(
        "smdeploy.serve:app_factory",
        factory=True,
        host="0.0.0.0",
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level.lower(),
        access_log=False,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
