FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspaces/Enterprise-World-Model-Engine

# Install uv for deterministic, fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:0.5.21 /uv /uvx /bin/

RUN python -m pip install --upgrade pip setuptools wheel
