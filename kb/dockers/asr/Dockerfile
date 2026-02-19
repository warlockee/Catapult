# =============================================================================
# ASR Tools Docker Image
# =============================================================================
# Lightweight container with ASR evaluation and client tools.
# For the full vLLM serving image, use custom-vllm Dockerfile with:
#
#   cd custom-vllm
#   DOCKER_BUILDKIT=1 docker build \
#       --build-arg SETUPTOOLS_SCM_PRETEND_VERSION="0.10.2" \
#       --build-arg RUN_WHEEL_CHECK=false \
#       --build-arg VLLM_USE_PRECOMPILED=1 \
#       --build-arg VLLM_PRECOMPILED_WHEEL_LOCATION=<wheel-file> \
#       --network=host \
#       --tag asr-model:v0.10.2 \
#       --target vllm-base \
#       --file docker/Dockerfile .
#
# This Dockerfile creates a tools-only image for evaluation/testing.
#
# Usage:
#   docker build -t asr-model-tools:latest -f Dockerfile .
#   docker run -v /fsx:/fsx asr-model-tools:latest \
#       python eval_asr.py /fsx/data/common_voice.arrow --url http://host:26007/v1
# =============================================================================

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/asr

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ASR tools
COPY audio_utils.py asr_client.py eval_asr.py test_asr_docker.py __init__.py ./

# Default: show help
CMD ["python", "eval_asr.py", "--help"]
