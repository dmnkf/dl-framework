FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    RYE_HOME=/app/.rye \
    LC_ALL=C.UTF-8

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        git \
        curl \
        build-essential && \
    rm -rf /var/lib/apt/lists/* && \
    curl -sSf https://rye.astral.sh/get | RYE_INSTALL_OPTION="--yes" bash

# https://fortuitous.com/guides/rye-global-guide/
COPY pyproject.toml /app/.rye/global/pyproject.toml
COPY requirements.lock /app/.rye/global/requirements.lock
RUN bash -c 'source "/app/.rye/env" && cd /app/.rye/global && rye sync --no-lock --no-dev'

WORKDIR /workspace

ENTRYPOINT [ "bash", "-c", "source /app/.rye/env && source /app/.rye/global/.venv/bin/activate && exec \"$@\"", "--" ]