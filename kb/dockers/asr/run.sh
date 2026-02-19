#!/bin/bash
# =============================================================================
# Run ASR vLLM Server
# =============================================================================
# Launches the vLLM OpenAI-compatible API server for ASR.
#
# Usage:
#   ./run.sh [OPTIONS]
#
# Options:
#   --model PATH          Path to model (required)
#   --model-name NAME     Served model name (default: asr-model)
#   --port PORT           API port (default: 26007)
#   --tensor-parallel N   Tensor parallel size (default: 1)
#   --max-seqs N          Max concurrent sequences (default: 174)
#   --max-model-len N     Max model length (default: 8192)
#
# Example:
#   ./run.sh --model /models/your-org/audio-model-v3-2b-svad
# =============================================================================

set -e

# Defaults
MODEL_PATH=""
MODEL_NAME="asr-model"
PORT=26007
TENSOR_PARALLEL_SIZE=1
MAX_NUM_SEQS=174
MAX_MODEL_LEN=8192

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --tensor-parallel)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --max-seqs)
            MAX_NUM_SEQS="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        -h|--help)
            head -25 "$0" | tail -20
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate
if [ -z "$MODEL_PATH" ]; then
    echo "Error: --model is required"
    echo "Usage: ./run.sh --model <path-to-model>"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path not found: $MODEL_PATH"
    exit 1
fi

echo "=============================================="
echo "Starting ASR Server"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Model name: $MODEL_NAME"
echo "Port: $PORT"
echo "Tensor parallel: $TENSOR_PARALLEL_SIZE"
echo "Max sequences: $MAX_NUM_SEQS"
echo "Max model length: $MAX_MODEL_LEN"
echo "=============================================="

python -m vllm.entrypoints.openai.api_server \
    --served-model-name "$MODEL_NAME" \
    --model "$MODEL_PATH" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --port "$PORT" \
    --trust-remote-code \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --enable-prefix-caching \
    --max-model-len "$MAX_MODEL_LEN"
