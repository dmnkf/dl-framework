# https://github.com/aws/deep-learning-containers/blob/master/available_images.md
FROM 763104351884.dkr.ecr.sa-east-1.amazonaws.com/pytorch-training:2.4.0-gpu-py311-cu124-ubuntu22.04-sagemaker

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

# AWS setup
RUN mkdir /opt/ml /opt/ml/input /opt/ml/input/config /opt/ml/input/data /opt/ml/input/data/training /opt/ml/model /opt/ml/output /opt/program

ENV PYTHONPATH=/opt/program
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set NVIDIA mount environments
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
ENV NVIDIA_VISIBLE_DEVICES="all"
ENV NVIDIA_DRIVER_CAPABILITIES="compute,utility"
ENV NVIDIA_REQUIRE_CUDA "cuda>=9.0"




# https://fortuitous.com/guides/rye-global-guide/
COPY pyproject.toml /app/.rye/global/pyproject.toml
COPY requirements.lock /app/.rye/global/requirements.lock

ENV PATH="/opt/ml/code:${PATH}"
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code

WORKDIR /opt/ml/code

COPY src /opt/ml/code/
COPY scripts/aws/. /opt/ml/code/

ENV SAGEMAKER_PROGRAM=entrypoint.sh
