# syntax=docker/dockerfile:1.7

# =============================================================================
# Stage 1: Lightweight Runtime Dependency Builder (CPU-only PaaS / 512MB RAM)
# Uses official astral-sh/uv image; excludes heavy GPU/eval deps (torch,
# sentence-transformers, fastembed, ragas) for fast builds & low memory usage.
# =============================================================================
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1

WORKDIR /build

RUN uv venv /opt/venv

RUN uv pip install --python /opt/venv/bin/python --no-cache \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.32.0" \
    "sse-starlette>=2.1.3" \
    "pydantic>=2.10.0" \
    "pydantic-settings>=2.6.0" \
    "langgraph>=0.2.55" \
    "langchain>=0.3.0" \
    "langchain-core>=0.3.0" \
    "langchain-openai>=0.2.0" \
    "langchain-google-genai>=2.0.0" \
    "langgraph-checkpoint-sqlite>=2.0.0" \
    "qdrant-client>=1.12.0" \
    "redis[hiredis]>=5.2.0" \
    "structlog>=24.4.0" \
    "httpx>=0.28.0" \
    "tenacity>=9.0.0" \
    "tiktoken>=0.8.0"

# =============================================================================
# Stage 2: Minimal Runtime Image
# =============================================================================
FROM python:3.11-slim-bookworm AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    RUNTIME_MODE=auto \
    PIPELINE_VERSION=v3 \
    LOG_LEVEL=INFO

WORKDIR /app

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

COPY --from=builder /opt/venv /opt/venv
COPY app ./app
COPY tests/sse_viewer.html ./tests/sse_viewer.html
COPY README.md README.en.md README.ko.md GEMINI.md ./

RUN mkdir -p /app/data && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/api/v1/health', timeout=3)"

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
