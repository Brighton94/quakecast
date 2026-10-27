FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgeos-dev libproj-dev proj-data proj-bin \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/home/vscode/.venv \
    PYTHONUNBUFFERED=1
