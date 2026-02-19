# =============================================================================
# Generic vLLM Docker Image for Model Registry
# =============================================================================
# This Dockerfile creates a generic vLLM inference container for any model.
# It uses the official vLLM OpenAI-compatible image and adds model files.
#
# Build Args:
#   MODEL_NAME     - Name of the model (used for paths and served-model-name)
#   VLLM_VERSION   - vLLM image version tag (default: latest)
#
# Environment Variables (runtime configurable):
#   MODEL_PATH         - Path to model inside container (default: /models/${MODEL_NAME})
#   SERVED_MODEL_NAME  - Model name for API (default: ${MODEL_NAME})
#   PORT               - API server port (default: 8000)
#   MAX_MODEL_LEN      - Maximum sequence length (default: auto)
#   GPU_MEMORY_UTIL    - GPU memory utilization (default: 0.9)
#   TENSOR_PARALLEL    - Tensor parallel size (default: 1)
#   DTYPE              - Data type: auto, half, bfloat16, float16 (default: auto)
#   EXTRA_ARGS         - Additional vLLM server arguments
#
# Usage:
#   docker build -t my-model:v1.0 --build-arg MODEL_NAME=Qwen2.5-7B .
#   docker run --gpus all -p 8000:8000 my-model:v1.0
# =============================================================================

ARG VLLM_VERSION=latest
FROM vllm/vllm-openai:${VLLM_VERSION}

# Build arguments
ARG MODEL_NAME=model

# Set model name as environment variable for runtime use
ENV MODEL_NAME=${MODEL_NAME}

# Create directories
WORKDIR /models
RUN mkdir -p /tmp/staging

# Copy all staged content from build context
# The workspace service creates: models/, requirements/, third_party/
COPY . /tmp/staging/

# Move model files to /models
RUN if [ -d "/tmp/staging/models/${MODEL_NAME}" ]; then \
        mv "/tmp/staging/models/${MODEL_NAME}" "/models/${MODEL_NAME}"; \
    elif [ -d "/tmp/staging/models" ]; then \
        # If MODEL_NAME subdir doesn't exist but models does, move all contents
        mv /tmp/staging/models/* /models/ 2>/dev/null || true; \
    fi

# Install staged wheel artifacts (vLLM wheels, custom packages, etc.)
# Wheels are staged directly in the build root by workspace_service.stage_artifacts()
# Skip dummy_wheel.whl (placeholder file created by workspace service)
RUN for whl in /tmp/staging/*.whl; do \
        if [ -f "$whl" ] && [ "$(basename $whl)" != "dummy_wheel.whl" ]; then \
            echo "Installing wheel: $whl"; \
            pip install --no-cache-dir --force-reinstall "$whl" || true; \
        fi; \
    done

# Install additional requirements if present
RUN if [ -f /tmp/staging/requirements/extra.txt ]; then \
        pip install --no-cache-dir -r /tmp/staging/requirements/extra.txt 2>/dev/null || true; \
    fi && \
    if [ -f /tmp/staging/requirements/requirements.txt ]; then \
        pip install --no-cache-dir -r /tmp/staging/requirements/requirements.txt 2>/dev/null || true; \
    fi

# Install third_party packages if present
RUN for pkg in /tmp/staging/third_party/*/; do \
        if [ -d "${pkg}" ]; then \
            if [ -f "${pkg}setup.py" ] || [ -f "${pkg}pyproject.toml" ]; then \
                pip install --no-cache-dir "${pkg}" --no-deps 2>/dev/null || true; \
            fi; \
        fi \
    done 2>/dev/null || true

# Cleanup staging area
RUN rm -rf /tmp/staging

# Default environment variables (can be overridden at runtime)
ENV PORT=8000
ENV GPU_MEMORY_UTIL=0.9
ENV TENSOR_PARALLEL=1
ENV DTYPE=auto
ENV MAX_MODEL_LEN=""
ENV EXTRA_ARGS=""

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose the API port
EXPOSE ${PORT}

# Flexible entrypoint script that uses environment variables
# This allows runtime configuration without rebuilding the image
ENTRYPOINT ["/bin/bash", "-c", "\
    # Find the model path - check if MODEL_NAME directory exists\n\
    if [ -d \"/models/${MODEL_NAME}\" ]; then\n\
        MODEL_DIR=\"/models/${MODEL_NAME}\";\n\
    else\n\
        # Find first directory in /models\n\
        MODEL_DIR=$(find /models -mindepth 1 -maxdepth 1 -type d | head -1);\n\
    fi;\n\
    \n\
    if [ -z \"$MODEL_DIR\" ]; then\n\
        echo \"ERROR: No model found in /models/\";\n\
        exit 1;\n\
    fi;\n\
    \n\
    echo \"Starting vLLM server with model: $MODEL_DIR\";\n\
    \n\
    ARGS=\"--model ${MODEL_DIR}\";\n\
    ARGS=\"${ARGS} --served-model-name ${SERVED_MODEL_NAME:-${MODEL_NAME}}\";\n\
    ARGS=\"${ARGS} --port ${PORT}\";\n\
    ARGS=\"${ARGS} --gpu-memory-utilization ${GPU_MEMORY_UTIL}\";\n\
    ARGS=\"${ARGS} --tensor-parallel-size ${TENSOR_PARALLEL}\";\n\
    ARGS=\"${ARGS} --dtype ${DTYPE}\";\n\
    if [ -n \"${MAX_MODEL_LEN}\" ]; then ARGS=\"${ARGS} --max-model-len ${MAX_MODEL_LEN}\"; fi;\n\
    if [ -n \"${EXTRA_ARGS}\" ]; then ARGS=\"${ARGS} ${EXTRA_ARGS}\"; fi;\n\
    \n\
    echo \"Running: python3 -m vllm.entrypoints.openai.api_server ${ARGS}\";\n\
    exec python3 -m vllm.entrypoints.openai.api_server ${ARGS}\n\
"]
