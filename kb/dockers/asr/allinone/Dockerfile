# =============================================================================
# ASR All-in-One Docker Image
# =============================================================================
# Complete ASR service: Audio → VAD → 4s Cut → Inference → Text
#
# Build:
#   docker build -t audio-1.7b-asr-allinone:latest -f Dockerfile .
#
# Run:
#   docker run --gpus '"device=0"' -p 8000:8000 -v /fsx:/fsx \
#       audio-1.7b-asr-allinone:latest \
#       --model /models/your-org/audio-model-v3-1.7b
# =============================================================================

FROM audio-model-v3:asr-vllm-v10

# Install additional dependencies for VAD preprocessing
RUN pip install --no-cache-dir \
    fastapi>=0.100.0 \
    uvicorn>=0.23.0 \
    python-multipart>=0.0.6 \
    soundfile>=0.12.0 \
    torchaudio>=2.0.0 \
    packaging>=21.0

# Create app directory
WORKDIR /app/asr

# Copy ASR service files
COPY asr_service.py .
COPY entrypoint.sh .

RUN chmod +x entrypoint.sh

# Expose ports: 8000 for ASR API, 26007 for internal vLLM
EXPOSE 8000 26007

ENTRYPOINT ["/app/asr/entrypoint.sh"]
