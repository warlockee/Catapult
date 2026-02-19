#!/bin/bash
# =============================================================================
# ASR All-in-One Entrypoint
# =============================================================================
# Starts both vLLM server and ASR FastAPI service
#
# Usage:
#   ./entrypoint.sh --model /path/to/model [OPTIONS]
#
# Options:
#   --model PATH          Path to model (required)
#   --model-name NAME     Served model name (default: asr-model)
#   --vllm-port PORT      vLLM internal port (default: 26007)
#   --api-port PORT       ASR API port (default: 8000)
#   --max-model-len N     Max model length (default: 4096)
# =============================================================================

set -e

# Defaults
MODEL_PATH=""
MODEL_NAME="asr-model"
VLLM_PORT=26007
API_PORT=8000
MAX_MODEL_LEN=4096
MAX_NUM_SEQS=100
GPU_MEMORY_UTILIZATION=0.85

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
        --vllm-port)
            VLLM_PORT="$2"
            shift 2
            ;;
        --api-port)
            API_PORT="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --max-num-seqs)
            MAX_NUM_SEQS="$2"
            shift 2
            ;;
        --gpu-memory-utilization)
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        -h|--help)
            head -20 "$0" | tail -15
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
    echo "Usage: ./entrypoint.sh --model <path-to-model>"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path not found: $MODEL_PATH"
    exit 1
fi

echo "=============================================="
echo "ASR All-in-One Service"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Model name: $MODEL_NAME"
echo "vLLM port: $VLLM_PORT (internal)"
echo "API port: $API_PORT (external)"
echo "Max model length: $MAX_MODEL_LEN"
echo "=============================================="

# Start vLLM server in background
echo "Starting vLLM server..."
python3 -m vllm.entrypoints.openai.api_server \
    --served-model-name "$MODEL_NAME" \
    --model "$MODEL_PATH" \
    --port "$VLLM_PORT" \
    --trust-remote-code \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    &

VLLM_PID=$!

# Wait for vLLM to be ready
echo "Waiting for vLLM to be ready..."
for i in {1..120}; do
    if curl -s "http://localhost:$VLLM_PORT/v1/models" > /dev/null 2>&1; then
        echo "vLLM is ready!"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "Error: vLLM process died"
        exit 1
    fi
    sleep 1
done

# Check if vLLM is ready
if ! curl -s "http://localhost:$VLLM_PORT/v1/models" > /dev/null 2>&1; then
    echo "Error: vLLM failed to start within 120 seconds"
    exit 1
fi

# Start ASR service
echo "Starting ASR service on port $API_PORT..."
export VLLM_URL="http://localhost:$VLLM_PORT/v1"
export MODEL_NAME="$MODEL_NAME"

python3 /app/asr/asr_service.py \
    --host 0.0.0.0 \
    --port "$API_PORT" \
    --vllm-url "http://localhost:$VLLM_PORT/v1" \
    --model "$MODEL_NAME"
