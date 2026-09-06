"""The SageMaker container filesystem contract.

SageMaker mounts one tree into every training and hosting container:

    /opt/ml/input/config/hyperparameters.json   all values are JSON *strings*
    /opt/ml/input/config/inputdataconfig.json   channel → content type / distribution
    /opt/ml/input/config/resourceconfig.json    hosts, current_host, network interface
    /opt/ml/input/data/<channel>/               one directory per channel (File mode)
    /opt/ml/model/                              training writes here → model.tar.gz in S3;
                                                hosting extracts model.tar.gz here
    /opt/ml/output/failure                      free-text failure reason surfaced in the console
    /opt/ml/output/data/                        extra outputs → output.tar.gz in S3
    /opt/ml/checkpoints/                        synced with CheckpointConfig.S3Uri (spot training)

The base directory is injectable (`SMDEPLOY_BASE_DIR`) so the exact same code runs in a
temporary directory on a laptop and in CI without Docker; the `SM_*` variables that the
SageMaker training toolkit sets are honoured when present.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_BASE = Path("/opt/ml")
ENV_BASE_DIR = "SMDEPLOY_BASE_DIR"


@dataclass(frozen=True)
class SageMakerPaths:
    base: Path = DEFAULT_BASE
    env: Mapping[str, str] = field(
        default_factory=lambda: dict(os.environ), compare=False, repr=False
    )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> SageMakerPaths:
        e = dict(os.environ) if env is None else dict(env)
        base = Path(e.get(ENV_BASE_DIR) or DEFAULT_BASE)
        return cls(base=base, env=e)

    # --- input -------------------------------------------------------------------------
    @property
    def input_config(self) -> Path:
        return self.base / "input" / "config"

    @property
    def hyperparameters_file(self) -> Path:
        return self.input_config / "hyperparameters.json"

    @property
    def input_data_config_file(self) -> Path:
        return self.input_config / "inputdataconfig.json"

    @property
    def resource_config_file(self) -> Path:
        return self.input_config / "resourceconfig.json"

    @property
    def input_data(self) -> Path:
        return self.base / "input" / "data"

    def channel(self, name: str) -> Path:
        override = self.env.get(f"SM_CHANNEL_{name.upper()}")
        return Path(override) if override else self.input_data / name

    def channel_files(self, name: str, suffix: str = ".csv") -> list[Path]:
        directory = self.channel(name)
        if not directory.is_dir():
            return []
        return sorted(p for p in directory.rglob(f"*{suffix}") if p.is_file())

    # --- output ------------------------------------------------------------------------
    @property
    def model_dir(self) -> Path:
        override = self.env.get("SM_MODEL_DIR")
        return Path(override) if override else self.base / "model"

    @property
    def output_dir(self) -> Path:
        return self.base / "output"

    @property
    def failure_file(self) -> Path:
        return self.output_dir / "failure"

    @property
    def output_data_dir(self) -> Path:
        override = self.env.get("SM_OUTPUT_DATA_DIR")
        return Path(override) if override else self.output_dir / "data"

    @property
    def checkpoints_dir(self) -> Path:
        return self.base / "checkpoints"

    def ensure_output_dirs(self) -> None:
        for d in (self.model_dir, self.output_dir, self.output_data_dir):
            d.mkdir(parents=True, exist_ok=True)

    # --- config files ------------------------------------------------------------------
    def read_resource_config(self) -> dict[str, Any]:
        return _read_json(self.resource_config_file)

    def read_input_data_config(self) -> dict[str, Any]:
        return _read_json(self.input_data_config_file)

    @property
    def current_host(self) -> str:
        return str(self.read_resource_config().get("current_host", "algo-1"))

    @property
    def num_cpus(self) -> int:
        override = self.env.get("SM_NUM_CPUS")
        return int(override) if override else (os.cpu_count() or 1)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return dict(loaded) if isinstance(loaded, dict) else {}
