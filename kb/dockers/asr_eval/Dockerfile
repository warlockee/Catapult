# ASR Evaluation Docker
# Usage: docker run <image> --endpoint_url http://host:port/v1
#
# Evaluates ASR model against Common Voice subset and compares WER to baseline.

FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    numpy \
    soundfile \
    pyarrow \
    openai \
    requests \
    jiwer \
    whisper-normalizer \
    packaging

# Pre-download Silero VAD model during build
RUN python3 -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)"

# Copy evaluation scripts and data
COPY audio_utils.py /app/
COPY run_eval.py /app/
COPY baseline.json /app/
COPY data/eval_subset.arrow /app/data/

# Set entrypoint
ENTRYPOINT ["python3", "/app/run_eval.py"]
