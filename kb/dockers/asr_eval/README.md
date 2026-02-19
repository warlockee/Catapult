# ASR Evaluation Docker

All-in-one Docker for evaluating ASR endpoints against Common Voice dataset.

## Quick Start

```bash
# Pull from Azure Container Registry
docker pull your-registry.example.com/catapult/asr-eval:v1

# Run evaluation
docker run --rm --network=host \
    your-registry.example.com/catapult/asr-eval:v1 \
    --endpoint_url http://host:port/v1
```

## Usage Examples

```bash
# Basic usage (vLLM V1 engine with Chat Completions API)
docker run --rm --network=host \
    your-registry.example.com/catapult/asr-eval:v1 \
    --endpoint_url http://host:port/v1

# For vLLM V0 engine (Audio Transcriptions API)
docker run --rm --network=host \
    your-registry.example.com/catapult/asr-eval:v1 \
    --endpoint_url http://host:port/v1 \
    --api-mode transcriptions \
    --no-baseline

# Quick test with fewer samples
docker run --rm --network=host \
    your-registry.example.com/catapult/asr-eval:v1 \
    --endpoint_url http://host:port/v1 \
    --limit 100

# Save detailed results to JSON
docker run --rm --network=host -v $(pwd):/output \
    your-registry.example.com/catapult/asr-eval:v1 \
    --endpoint_url http://host:port/v1 \
    --output /output/results.json

# Measurement only (no pass/fail)
docker run --rm --network=host \
    your-registry.example.com/catapult/asr-eval:v1 \
    --endpoint_url http://host:port/v1 \
    --no-baseline
```

## API Mode

The eval Docker supports two API modes for different vLLM engine configurations:

| Mode | API Endpoint | vLLM Engine | Use When |
|------|--------------|-------------|----------|
| `chat` (default) | `/v1/chat/completions` | V1 (`VLLM_USE_V1=1`) | Standard deployment |
| `transcriptions` | `/v1/audio/transcriptions` | V0 (`VLLM_USE_V1=0`) | Stability-focused deployment |

**Important**: The baseline WER (13.55%) was established using `chat` mode. When using `transcriptions` mode, use `--no-baseline` for initial testing.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--endpoint_url` | (required) | ASR endpoint URL (e.g., http://host:port/v1) |
| `--model` | asr-model | Model name at the endpoint |
| `--api-mode` | chat | API mode: `chat` or `transcriptions` |
| `--limit` | 500 | Maximum samples to evaluate |
| `--offset` | 0 | Start index in dataset |
| `--language` | English | Language for transcription |
| `--output` | - | Path to save JSON results |
| `--expected-wer` | 0.1355 | Override expected WER (e.g., 0.15 for 15%) |
| `--tolerance` | 0.01 | Override WER tolerance (e.g., 0.02 for 2%) |
| `--no-baseline` | - | Just measure WER without baseline comparison |
| `--repetition-penalty` | 1.1 | Repetition penalty (chat mode only) |
| `--verbose` | - | Show per-sample output |

## Baseline

- **Expected WER**: 13.55% (tolerance: +/- 1%)
- **API Mode**: `chat` (Chat Completions API)
- **Model**: `audio-model-v3-1.7b-chk-65000`
- **Dataset**: Common Voice 15 English (500 samples)

## Exit Codes

- `0`: PASS - WER within tolerance or better than baseline
- `1`: FAIL - WER exceeds tolerance

## Dataset

Contains 500 samples from Common Voice 15 English dataset embedded in the Docker image.

## Building

```bash
# Build locally
docker build -t asr-eval .

# Build with network access (if needed)
docker build --network=host -t asr-eval .
```

## Troubleshooting

### "Model does not support Chat Completions API"
Your endpoint is running vLLM V0 engine. Use `--api-mode transcriptions`:
```bash
docker run --rm --network=host \
    your-registry.example.com/catapult/asr-eval:v1 \
    --endpoint_url http://host:port/v1 \
    --api-mode transcriptions \
    --no-baseline
```

### WER higher than expected
1. Verify the endpoint is running the correct model checkpoint
2. Check vLLM server has `--max-model-len 8192`
3. For chat mode, ensure `repetition_penalty=1.1` is applied

### Connection refused
1. Verify the endpoint URL is correct and includes `/v1`
2. Use `--network=host` when running Docker
3. Check the ASR service is running and healthy
