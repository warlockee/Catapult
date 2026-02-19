# ASR Docker

Complete package for building, running, and evaluating Audio Understanding (ASR) service.

## Contents

```
kb/dockers/asr/
├── Dockerfile           # Docker build file (alternative to custom-vllm Dockerfile)
├── build.sh             # Build script using custom-vllm
├── run.sh               # Run vLLM server
├── requirements.txt     # Python dependencies for tools
├── audio_utils.py       # VAD, segmentation, encoding utilities
├── asr_client.py        # CLI: transcribe single audio files
├── eval_asr.py          # WER/CER evaluation against Arrow datasets
├── test_asr_docker.py   # Smoke test for vLLM endpoint
└── README.md            # This file
```

## Quick Start

### 1. Build Docker Image

```bash
# Clone your custom vLLM fork
git clone --recursive -b your-branch git@github.com:your-org/custom-vllm.git
cd custom-vllm

# Copy pre-compiled wheel
cp /workspace/vllm-builds/vllm-0.10.2+asr-cp312-cp312-linux_x86_64.whl .

# Build image
DOCKER_BUILDKIT=1 docker build \
    --build-arg SETUPTOOLS_SCM_PRETEND_VERSION="0.10.2" \
    --build-arg RUN_WHEEL_CHECK=false \
    --build-arg VLLM_USE_PRECOMPILED=1 \
    --build-arg VLLM_PRECOMPILED_WHEEL_LOCATION=vllm-0.10.2+asr-cp312-cp312-linux_x86_64.whl \
    --network=host \
    --tag asr-model:v0.10.2 \
    --target vllm-base \
    --file docker/Dockerfile \
    .
```

### 2. Run Container

```bash
# Start container
docker run --gpus '"device=0"' \
    --ipc=host --network=host --shm-size=20gb \
    -v /fsx:/fsx \
    --name asr-model \
    -itd asr-model:v0.10.2

# Enter container
docker exec -it asr-model /bin/bash
```

### 3. Launch vLLM Server (inside container)

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --served-model-name "asr-model" \
    --model /models/your-org/audio-model-v3-1.7b \
    --tensor-parallel-size 1 \
    --port 26007 \
    --trust-remote-code \
    --max-num-seqs 100 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85
```

### 4. Test & Evaluate

```bash
# Smoke test
python test_asr_docker.py --port 26007 --model asr-model

# Transcribe a file
python asr_client.py audio.wav --url http://localhost:26007/v1 --model asr-model

# Run WER evaluation
python eval_asr.py /data/audio_eval/asr/common_voice_15_en.arrow \
    --url http://localhost:26007/v1 \
    --model asr-model \
    --limit 100 \
    --output results.json
```

## Pipeline Details

### Audio Processing Pipeline

1. **Load audio** → resample to 16kHz mono
2. **VAD (Voice Activity Detection)** → detect speech segments using Silero VAD
3. **Segmentation** → split into ≤4s chunks
4. **Base64 encoding** → encode WAV chunks as base64
5. **vLLM API** → send to chat completions endpoint
6. **Post-processing** → combine transcriptions

### VAD Parameters

Default parameters match reference evaluation:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `threshold` | 0.55 | Speech probability threshold |
| `min_speech_duration_ms` | 125 | Minimum speech chunk duration |
| `min_silence_duration_ms` | 200 | Minimum silence between chunks |
| `speech_pad_ms` | 300 | Padding added to speech chunks |

### API Format

vLLM OpenAI-compatible chat completions with audio:

```python
{
    "model": "asr-model",
    "messages": [
        {"role": "system", "content": "You are an automatic speech recognition (ASR) system."},
        {"role": "user", "content": [
            {"type": "text", "text": "Your task is to listen to audio input and output the exact spoken words as plain text in English."},
            {"type": "audio_url", "audio_url": {"url": "data:audio/wav_0;base64,..."}}
        ]}
    ],
    "max_completion_tokens": 256,
    "temperature": 0.0,
    "stop": ["<|eot_id|>", "<|endoftext|>", "<|audio_eos|>", "<|im_end|>"]
}
```

## Evaluation Output

`eval_asr.py --output results.json` produces:

```json
{
    "wer": 0.1230,
    "cer": 0.0456,
    "samples_evaluated": 100,
    "samples_with_errors": 0,
    "no_speech_count": 2,
    "dataset_path": "/data/audio_eval/asr/common_voice_15_en.arrow",
    "elapsed_seconds": 123.4,
    "model": "asr-model",
    "language": "English",
    "results": [
        {
            "unique_id": 0,
            "reference": "The quick brown fox...",
            "hypothesis": "The quick brown fox...",
            "reference_normalized": "the quick brown fox",
            "hypothesis_normalized": "the quick brown fox",
            "num_segments": 2,
            "audio_duration_seconds": 3.5,
            "error": null
        }
    ]
}
```

## Models

Available models:

| Model | Path | Description |
|-------|------|-------------|
| 1.7B | `/models/your-org/audio-model-v3-1.7b` | 1.7B parameters |
| 1.7B (SVAD) | `/models/your-org/audio-model-v3-2b-svad` | 1.7B parameters, server-side VAD variant |

## Expected Results

Common Voice 15 English:

| Samples | Our WER | Reference WER | Difference |
|---------|---------|---------------|------------|
| 100 | 16.56% | - | - |
| 500 | **13.12%** | **13.19%** | **0.07%** |
| Full (16,386) | - | 6.80% | - |

**Note**: "2B" is shorthand for 1.7B model.
