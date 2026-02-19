# WER Alignment Investigation

## Problem Statement
ASR evaluation through Catapult showed inconsistent WER results compared to reference servers and between different evaluation methods.

## Investigation Date: 2026-01-31

---

## Trial 1: Benchmark Format Issue (Frontend)

**Problem:** 100% benchmark failure rate for asr-allinone deployments.

**Root Cause:** Frontend's `getRequestBodyForEndpoint()` in `benchmark-utils.ts` was returning chat completions JSON for `/transcribe` endpoint, but `/transcribe` expects file upload, not JSON.

**Fix Applied:**
```typescript
// frontend/src/components/shared/benchmark-utils.ts
} else if (endpointPath === '/transcribe' || endpointPath === '/health') {
  // ASR all-in-one /transcribe expects file upload, not JSON
  // /health endpoints don't need a request body
  return {};
}
```

**Result:** Benchmarks no longer fail with format errors.

---

## Trial 2: Routing Issue (Backend)

**Problem:** asr-allinone deployments not routing to inference benchmark path.

**Root Cause:** `/transcribe` didn't match the `/v1/` check for `is_inference_benchmark` in `benchmark_service.py`.

**Fix Applied:**
```python
# backend/app/services/benchmark_service.py (line ~1367)
is_inference_benchmark = (
    benchmark.method.upper() == "POST" and
    ("/v1/" in benchmark.endpoint_path or benchmark.endpoint_path == "/transcribe")
)

# Force inference path for asr-allinone
if server_type and server_type.lower() == 'asr-allinone':
    logger.info("asr-allinone deployment: routing to inference benchmark path")
    is_inference_benchmark = True
```

**Result:** asr-allinone correctly routes to ASR evaluation.

---

## Trial 3: Health Endpoint for Load Tests

**Problem:** Load tests were trying to use `/transcribe` with JSON body.

**Root Cause:** For asr-allinone, API benchmarks should use `/health` (GET) instead of `/transcribe` (file upload).

**Fix Applied:**
```python
# backend/app/services/benchmark_service.py (line ~1532)
if server_type and server_type.lower() == 'asr-allinone':
    selected_inference_endpoint = "/health"
    selected_inference_payload = None
    logger.info("asr-allinone: using /health for API benchmark")

# Also fixed method to be GET for /health (line ~1598)
benchmark_method = "GET" if selected_inference_endpoint == "/health" else "POST"
```

**Result:** Load tests pass using health checks.

---

## Trial 4: TCP Connection Exhaustion

**Problem:** `Errno 99: Cannot assign requested address` during evaluation.

**Root Cause:** Previous evaluation opened too many TCP connections without proper cleanup.

**Fix:** Restarted the deployment container.

**Result:** Connections recovered.

---

## Trial 5: Sync vs Async WER Discrepancy

**Problem:** Different WER results between sync and async benchmarks.

| Benchmark ID | Type | Time | Samples | WER |
|-------------|------|------|---------|-----|
| 363d9a21 | sync | 20:27 | 500 | 12.88% |
| 9342cf22 | async/UI | 21:36 | 500 | 17.42% |
| e33c3900 | async | later | 100 | 38% |
| 202c417b | sync | later | 100 | 38% |

**Investigation:** Both sync and async eventually returned 38% WER for 100 samples. The 12.88% result from earlier could not be reproduced.

**Finding:** WER varies dramatically depending on which samples are evaluated.

---

## Trial 6: Sample Range WER Analysis

**Problem:** Understanding why WER varies so much.

**Test:** Evaluated different sample ranges from CommonVoice dataset.

**Results:**
| Sample Range | WER (asr-allinone) |
|--------------|-------------------|
| 0-49         | 17.98%            |
| 50-99        | 55.94%            |
| 100-149      | 10.33%            |
| 200-249      | 14.95%            |

**Finding:** Samples 50-99 have extremely high WER (55.94%), which explains why:
- 100 samples (0-99) → ~38% WER (average of 18% and 56%)
- 500 samples → varies based on sample distribution

---

## Trial 7: Reference Server Comparison

**Problem:** Is the high WER on samples 50-99 expected?

**Test:** Compare asr-allinone with reference server h100-31.

**Results:**
| Sample Range | asr-allinone (1.7B) | h100-31 (2B) |
|--------------|---------------------|--------------|
| 0-49         | 17.98%              | 17.98%       |
| 50-99        | 55.94%              | 13.88%       |

**Finding:** 4x WER difference on samples 50-99! The models produce different results.

---

## Trial 8: Model Identity Investigation

**Problem:** Why do the models perform so differently?

**Discovery 1 - Model Mislabeling:**
```
Model name: audio-model-v3-2b-svad (suggests 2B)
Actual path: /models/your-org/audio-model-v3-1.7b (1.7B!)
```

**Discovery 2 - Internal vLLM Check:**
```bash
$ docker exec deployment-xxx curl -s http://localhost:26007/v1/models
# Returns: "root": "/models/audio-model-v3-1.7b"
```

**Discovery 3 - Reference Server:**
```
h100-31 model: audio-model-v3-2b-svad (actual 2B model)
```

**Root Cause:** The model registry entry was pointing to the wrong checkpoint.

---

## Trial 9: Attempted 2B Model Fix

**Action:** Updated model's storage_path to 2B checkpoint.
```bash
curl -X PUT "http://localhost:8000/api/v1/models/..." \
  -d '{"storage_path": "/models/your-org/audio-model-v3-2b"}'
```

**Action:** Triggered new Docker build.

**Result:** FAILED - Container crashes on startup.

**Root Cause:** The 2B checkpoint directory is EMPTY:
```bash
$ ls /models/your-org/audio-model-v3-2b/
# (empty)

$ ls /models/your-org/audio-model-v3-1.7b/
config.json  model-00001-of-00002.safetensors  model-00002-of-00002.safetensors  ...
```

---

## Trial 10: Repetition Loop Analysis

**Problem:** Why does 1.7B get 55.94% WER while 2B gets 13.88% on samples 50-99?

**Test:** Detailed comparison of sample 65 outputs.

**Results:**
```
Reference text: "Other well-known relatives in Handsome Lake's family included
                Governor Blacksnake, Red Jacket and Half-Town."

1.7B output: "I'm going to have a look at it. I'm going to have a look at it.
             I'm going to have a look at it..." (repeated ~25 times)
WER: 1352.94%

2B output: "Other women will look good. And how can you look? Family includes
           both dogs and cats. Black men, yellow jackets, and black men."
WER: 117.65%
```

**Root Cause:** The 1.7B model gets stuck in **repetition loops** on difficult audio samples, producing gibberish that inflates WER dramatically.

---

## Summary of Root Causes

| Issue | Root Cause | Status |
|-------|-----------|--------|
| Benchmark failures | Wrong request format for /transcribe | FIXED |
| Routing issues | /transcribe not detected as inference | FIXED |
| Load test failures | Using /transcribe instead of /health | FIXED |
| TCP exhaustion | Connection leak | FIXED (restart) |
| WER inconsistency | Sample range variation | UNDERSTOOD |
| 4x WER difference | 1.7B vs 2B model | IDENTIFIED |
| Model mislabeling | Wrong checkpoint path | IDENTIFIED |
| 2B model missing | Empty checkpoint directory | **BLOCKER** |
| Repetition loops | 1.7B model limitation | IDENTIFIED |

---

## Key Lessons Learned

### 1. Never Trust Model Names
The model `audio-model-v3-2b-svad` was named "2B" but actually contained a 1.7B checkpoint. **Always verify the actual model files.**

### 2. Verify Checkpoint Directories Have Files
```bash
# ALWAYS check before building Docker images
ls -la /models/your-org/<checkpoint>/config.json
```

### 3. WER Varies Dramatically by Sample
CommonVoice dataset difficulty is not uniform:
- Some ranges: 10% WER
- Some ranges: 56% WER
- Always use consistent sample sets for comparison

### 4. Repetition Loops Inflate WER
A single sample with a repetition loop can have >1000% WER, dramatically affecting overall metrics.

### 5. Check Internal vLLM Model
```bash
docker exec <container> curl -s http://localhost:26007/v1/models | jq '.data[0].root'
```

---

## Action Items

1. **Find actual 2B model files** - h100-31 is serving them, so they exist somewhere
2. **Populate 2B checkpoint directory** with correct files
3. **Rebuild Docker image** with 2B model
4. **Add Docker build validation** - check model files exist before building
5. **Consider repetition penalty** for 1.7B as fallback
6. **Document model→checkpoint mapping** to prevent future mislabeling

---

## Technical Reference

### asr-allinone vLLM Configuration
```python
max_completion_tokens=256
temperature=0.0
stop=["<|eot_id|>", "<|endoftext|>", "<|audio_eos|>", "<|im_end|>"]
extra_body={"skip_special_tokens": False}
```

### VAD Configuration
```python
threshold: float = 0.55
min_speech_duration_ms: int = 125
min_silence_duration_ms: int = 200
speech_pad_ms: int = 300
```

### Container Images
| Tag | Size | Model | Status |
|-----|------|-------|--------|
| vllm-0.10.2-allinone-v2 | 43GB | 1.7B | Working |
| vllm-0.10.2-allinone-v3 | 31GB | Empty | Broken |

### Key Endpoints
- `/transcribe` - File upload, POST, ASR evaluation
- `/health` - No body, GET, load testing
- `/v1/models` - Internal vLLM model info
