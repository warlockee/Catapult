#!/bin/bash
# =============================================================================
# Build ASR Docker Image
# =============================================================================
# This script builds the ASR Docker image from custom-vllm.
#
# Prerequisites:
#   - custom-vllm repository cloned
#   - Pre-compiled vLLM wheel available
#
# Usage:
#   ./build.sh [OPTIONS]
#
# Options:
#   --vllm-dir PATH       Path to custom-vllm directory (default: ./custom-vllm)
#   --wheel PATH          Path to vLLM wheel (required)
#   --tag TAG             Docker image tag (default: asr-model:latest)
#   --version VERSION     vLLM version (default: 0.10.2)
#   --push                Push to registry after build
#   --registry REGISTRY   Registry URL for push
#
# Example:
#   ./build.sh --vllm-dir /path/to/custom-vllm \
#       --wheel vllm-0.10.2+asr-cp312-cp312-linux_x86_64.whl \
#       --tag asr-model:v0.10.2
# =============================================================================

set -e

# Defaults
VLLM_DIR="./custom-vllm"
VLLM_WHEEL=""
IMAGE_TAG="asr-model:latest"
VLLM_VERSION="0.10.2"
PUSH=false
REGISTRY=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --vllm-dir)
            VLLM_DIR="$2"
            shift 2
            ;;
        --wheel)
            VLLM_WHEEL="$2"
            shift 2
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --version)
            VLLM_VERSION="$2"
            shift 2
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        -h|--help)
            head -30 "$0" | tail -25
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate
if [ -z "$VLLM_WHEEL" ]; then
    echo "Error: --wheel is required"
    echo "Usage: ./build.sh --wheel <path-to-wheel>"
    exit 1
fi

if [ ! -d "$VLLM_DIR" ]; then
    echo "Error: custom-vllm directory not found: $VLLM_DIR"
    echo "Clone it first: git clone --recursive -b main git@github.com:your-org/custom-vllm.git"
    exit 1
fi

if [ ! -f "$VLLM_WHEEL" ]; then
    echo "Error: vLLM wheel not found: $VLLM_WHEEL"
    exit 1
fi

# Get absolute paths
VLLM_DIR=$(realpath "$VLLM_DIR")
VLLM_WHEEL=$(realpath "$VLLM_WHEEL")
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Copy wheel to vllm directory
WHEEL_NAME=$(basename "$VLLM_WHEEL")
cp "$VLLM_WHEEL" "$VLLM_DIR/$WHEEL_NAME"

# Copy ASR tools to vllm directory for build context
mkdir -p "$VLLM_DIR/kb/dockers/asr"
cp "$SCRIPT_DIR"/*.py "$VLLM_DIR/kb/dockers/asr/"

echo "=============================================="
echo "Building ASR Docker Image"
echo "=============================================="
echo "vLLM directory: $VLLM_DIR"
echo "vLLM wheel: $WHEEL_NAME"
echo "Image tag: $IMAGE_TAG"
echo "vLLM version: $VLLM_VERSION"
echo "=============================================="

cd "$VLLM_DIR"

# Build image
DOCKER_BUILDKIT=1 docker build \
    --build-arg SETUPTOOLS_SCM_PRETEND_VERSION="$VLLM_VERSION" \
    --build-arg RUN_WHEEL_CHECK=false \
    --build-arg VLLM_USE_PRECOMPILED=1 \
    --build-arg VLLM_PRECOMPILED_WHEEL_LOCATION="$WHEEL_NAME" \
    --network=host \
    --tag "$IMAGE_TAG" \
    --target vllm-base \
    --file docker/Dockerfile \
    .

echo "=============================================="
echo "Build complete: $IMAGE_TAG"
echo "=============================================="

# Push if requested
if [ "$PUSH" = true ]; then
    if [ -n "$REGISTRY" ]; then
        FULL_TAG="$REGISTRY/$IMAGE_TAG"
        docker tag "$IMAGE_TAG" "$FULL_TAG"
        docker push "$FULL_TAG"
        echo "Pushed: $FULL_TAG"
    else
        docker push "$IMAGE_TAG"
        echo "Pushed: $IMAGE_TAG"
    fi
fi

# Cleanup
rm -f "$VLLM_DIR/$WHEEL_NAME"
rm -rf "$VLLM_DIR/kb"

echo "Done!"
