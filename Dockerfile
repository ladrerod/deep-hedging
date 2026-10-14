# syntax=docker/dockerfile:1
FROM python:3.12-slim
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_DEV=1 PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp \
    DEEP_HEDGING_MODELS=/app/models DEEP_HEDGING_OUT=/data DEEP_HEDGING_HOST=0.0.0.0
WORKDIR /app
RUN --mount=from=ghcr.io/astral-sh/uv:0.13,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=from=ghcr.io/astral-sh/uv:0.13,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable
COPY models ./models
RUN mkdir /data && chown 10001:10001 /data
USER 10001:10001
EXPOSE 8000
ENTRYPOINT ["deep-hedging"]
CMD ["serve"]
