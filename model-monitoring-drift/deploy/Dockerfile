# mlwatch exporter image: runs the Prometheus endpoint and the on-demand monitor.
FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[serve]"

FROM python:3.12-slim
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY --from=build /opt/venv /opt/venv
RUN useradd --create-home --uid 10001 mlwatch && mkdir -p /data && chown mlwatch /data
USER mlwatch
WORKDIR /data
EXPOSE 9108
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:9108/health', timeout=3).status == 200 else 1)"
CMD ["mlwatch", "serve", "--baseline", "/data/baseline.json", "--policy", "/data/policy.yaml", "--state", "/data/monitor.state.json", "--allowed-root", "/data", "--port", "9108"]
