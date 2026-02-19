# ASR All-in-One Docker

Complete ASR service in a single container: **Audio → VAD → 4s Cut → Inference → Text**

## Overview

This Docker image provides a complete ASR pipeline that handles:
1. Audio loading and resampling to 16kHz mono
2. Voice Activity Detection (Silero VAD)
3. Segmentation into ≤4s chunks
4. Inference via internal vLLM server
5. Text output

No client-side preprocessing required - just send audio files and get transcriptions.

## Validation Results

| Configuration | WER | Samples | Notes |
|--------------|-----|---------|-------|
| H100-31 (reference) | 13.12% | 500 | Tools eval against production server |
| All-in-one Docker | 12.88% | 500 | Validated alignment |
| H100-31 (partial) | ~10.9% | ~6800 | Partial full dataset eval |
| All-in-one (partial) | ~11.7% | ~6800 | Partial full dataset eval |

### Key Fixes for Alignment

1. **`extra_body={"skip_special_tokens": False}`** - Required for proper vLLM output
2. **`MAX_CHUNKS_PER_REQUEST=4`** - Batch audio chunks (≤16s per API call) to match reference

## Quick Start

### Build

```bash
cd kb/dockers/asr/allinone
docker build -t audio-1.7b-asr-allinone:latest .
```

### Run

```bash
docker run --gpus '"device=0"' \
    --ipc=host --shm-size=20gb \
    -p 8000:8000 \
    -v /fsx:/fsx \
    -v ~/.cache/torch/hub:/root/.cache/torch/hub:ro \
    audio-1.7b-asr-allinone:latest \
    --model /models/your-org/audio-model-v3-1.7b
```

Note: Mount torch hub cache for Silero VAD model (avoids download at startup).

### Test

```bash
# Health check
curl http://localhost:8000/health

# Transcribe audio file
curl -X POST http://localhost:8000/transcribe \
    -F "file=@audio.wav" \
    -F "language=English"
```

## API Endpoints

### `GET /health`
Health check endpoint.

**Response:**
```json
{
    "status": "healthy",
    "model": "asr-model",
    "vllm_url": "http://localhost:26007/v1"
}
```

### `POST /transcribe`
Transcribe a single audio file.

**Parameters:**
- `file` (required): Audio file (wav, mp3, flac, etc.)
- `language` (optional): Target language (default: "English", use "auto" for auto-detect)

**Response:**
```json
{
    "transcription": "Hello, how are you?",
    "language": "English",
    "audio_duration_seconds": 2.5,
    "num_segments": 1,
    "processing_time_seconds": 0.8,
    "realtime_factor": 3.1
}
```

### `POST /transcribe/batch`
Transcribe multiple audio files.

**Parameters:**
- `files` (required): List of audio files
- `language` (optional): Target language

**Response:**
```json
{
    "results": [
        {"filename": "audio1.wav", "transcription": "Hello", "status": "success"},
        {"filename": "audio2.wav", "transcription": "World", "status": "success"}
    ]
}
```

## Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | (required) | Path to model |
| `--model-name` | asr-model | Served model name |
| `--api-port` | 8000 | External API port |
| `--vllm-port` | 26007 | Internal vLLM port |
| `--max-model-len` | 4096 | Max model length |
| `--max-num-seqs` | 100 | Max concurrent sequences |
| `--gpu-memory-utilization` | 0.85 | GPU memory utilization |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   All-in-One Container                   │
│                                                         │
│  ┌─────────────────┐      ┌─────────────────────────┐  │
│  │   FastAPI       │      │        vLLM             │  │
│  │   (port 8000)   │─────▶│    (port 26007)        │  │
│  │                 │      │                         │  │
│  │ • /transcribe   │      │ • Model inference       │  │
│  │ • VAD           │      │ • OpenAI-compatible API │  │
│  │ • 4s cutting    │      │                         │  │
│  └─────────────────┘      └─────────────────────────┘  │
│           ▲                                             │
│           │                                             │
└───────────┼─────────────────────────────────────────────┘
            │
      Audio input
```

## Pipeline Details

1. **Audio Loading**: Accepts any format, resamples to 16kHz mono
2. **VAD**: Silero VAD with parameters:
   - threshold: 0.55
   - min_speech_duration_ms: 125
   - min_silence_duration_ms: 200
   - speech_pad_ms: 300
3. **Segmentation**: Splits into ≤4s chunks
4. **Batching**: Groups ≤4 chunks per vLLM request (~16s max)
5. **Inference**: vLLM with Audio model
6. **Output**: Plain text transcription

## Python Client Example

```python
import requests

# Single file
with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/transcribe",
        files={"file": f},
        data={"language": "English"}
    )
    print(response.json()["transcription"])

# Batch
files = [
    ("files", open("audio1.wav", "rb")),
    ("files", open("audio2.wav", "rb")),
]
response = requests.post(
    "http://localhost:8000/transcribe/batch",
    files=files,
    data={"language": "English"}
)
for result in response.json()["results"]:
    print(f"{result['filename']}: {result['transcription']}")
```
