# Pinned deterministic reproduction container for Enterprise World Model Engine (T09)
FROM python:3.11-slim-bookworm

LABEL org.opencontainers.image.title="Enterprise World Model Engine" \
      org.opencontainers.image.description="Reproducible scientific world model kernel with certified uncertainty" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.source="https://github.com/vfcarida/Enterprise-World-Model-Engine"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for deterministic dependency resolution
COPY --from=ghcr.io/astral-sh/uv:0.5.21 /uv /bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY scripts ./scripts
COPY conformance ./conformance
COPY Makefile ./

# Install dependencies using pinned pyproject and uv
RUN uv pip install --system -e ".[all,dsl]"

# Default execution runs full scientific reproduction
CMD ["python", "scripts/reproduce.py", "--manifest", "/app/reproduction_manifest.json"]
