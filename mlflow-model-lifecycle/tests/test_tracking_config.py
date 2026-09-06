"""Client defaults for an HTTP tracking server.

Found against `deploy/docker-compose.yml`: the server advertises presigned multipart downloads
whose URLs point at `minio:9000`; from the host every model load stalled in retries.
"""

from __future__ import annotations

import pytest

from mlreg.tracking import PROXY_MULTIPART_ENV, prefer_proxied_downloads


@pytest.mark.parametrize("uri", ["http://localhost:5000", "https://mlflow.example.com"])
def test_http_server_defaults_to_proxied_downloads(uri: str) -> None:
    env: dict[str, str] = {}
    assert prefer_proxied_downloads(uri, env) is True
    assert env == {PROXY_MULTIPART_ENV: "false"}


def test_explicit_setting_is_respected() -> None:
    env = {PROXY_MULTIPART_ENV: "true"}
    assert prefer_proxied_downloads("http://localhost:5000", env) is False
    assert env[PROXY_MULTIPART_ENV] == "true"


@pytest.mark.parametrize("uri", ["sqlite:///mlruns/mlflow.db", "file:./mlruns", "databricks"])
def test_local_and_non_http_stores_are_left_alone(uri: str) -> None:
    env: dict[str, str] = {}
    assert prefer_proxied_downloads(uri, env) is False
    assert env == {}
