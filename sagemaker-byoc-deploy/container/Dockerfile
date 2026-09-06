# syntax=docker/dockerfile:1.7
# Bring-your-own-container image for SageMaker training AND hosting.
# SageMaker starts it as `docker run <image> train` or `docker run <image> serve`, so the two
# executables live on PATH and there is deliberately no ENTRYPOINT.
FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

FROM python:3.12-slim
LABEL org.opencontainers.image.title="smdeploy" \
      org.opencontainers.image.description="SageMaker BYOC training + inference container" \
      org.opencontainers.image.licenses="MIT"
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SAGEMAKER_BIND_TO_PORT=8080 \
    SAGEMAKER_MAX_PAYLOAD_IN_MB=6 \
    SMDEPLOY_LOG_LEVEL=INFO
COPY --from=build /opt/venv /opt/venv
COPY container/train container/serve /usr/local/bin/
RUN chmod 0755 /usr/local/bin/train /usr/local/bin/serve \
 && useradd --create-home --uid 10001 sagemaker \
 && mkdir -p /opt/ml/model /opt/ml/input /opt/ml/output /opt/ml/checkpoints \
 && chown -R sagemaker:sagemaker /opt/ml
USER sagemaker
WORKDIR /home/sagemaker
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/ping', timeout=3).status == 200 else 1)"
CMD ["serve"]
