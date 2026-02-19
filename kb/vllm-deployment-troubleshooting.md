# vLLM Deployment Troubleshooting Guide

## Overview
This document captures lessons learned from deploying a 70B LLaMA model with vLLM, including GPU selection, tensor parallelism, and various runtime issues.

---

## Issue 1: Tokenizer TypeError

### Problem
vLLM fails with `TypeError: not a string` when loading tokenizer.

### Root Cause
Missing `tokenizer.model` file in the model directory. When the workspace service couldn't find the model source, it created dummy config files with empty `{}` content, causing tokenizer to fail because `vocab_file` was `None`.

### Solution
1. Ensure model directory contains all required tokenizer files (`tokenizer.model`, `tokenizer.json`, `tokenizer_config.json`)
2. For LLaMA models, copy tokenizer from a compatible model if missing:
   ```bash
   cp /ceph/models/Llama-3.3-70B-Instruct/original/tokenizer.model /path/to/your/model/
   ```

### Code Fix
Modified `workspace_service.py` to fail-fast instead of creating dummy files when model source doesn't exist.

---

## Issue 2: CEPH Path Resolution

### Problem
Model source not found at `/data/ceph/...` or `/fsx/models/...` paths.

### Root Cause
1. Worker container's `CEPH_MOUNT_PATH` was set to `/data/ceph` instead of `/fsx`
2. CEPH models directory wasn't mounted in containers

### Solution
1. Added `normalize_ceph_path()` function to translate `/ceph/` paths to `CEPH_MOUNT_PATH`:
   ```python
   def normalize_ceph_path(path: str) -> str:
       if path.startswith('/ceph/'):
           suffix = path[len('/ceph/'):]
           return os.path.join(settings.CEPH_MOUNT_PATH, suffix)
       return path
   ```

2. Updated `docker-compose.yml`:
   ```yaml
   # Worker CEPH_MOUNT_PATH
   CEPH_MOUNT_PATH: /fsx  # Changed from /data/ceph

   # Added volume mount
   - /ceph/models:/fsx/models:ro
   ```

---

## Issue 3: GPU Memory - Model Too Large

### Problem
CUDA out of memory (OOM) when loading 70B model with tensor parallelism.

### Root Cause
70B model in BF16 requires ~140GB GPU memory:
- TP=1: 140GB needed, 40GB available → OOM
- TP=2: 70GB per GPU needed, 40GB available → OOM
- TP=4: 35GB per GPU needed, 40GB available → Works

### Solution
For 70B BF16 models on A100 40GB GPUs, use minimum TP=4:
```json
{
  "gpu_count": 4,
  "environment_vars": {
    "TENSOR_PARALLEL": "4"
  }
}
```

### Memory Formula
```
GPU memory per device = (model_params × bytes_per_param) / tensor_parallel_size
70B × 2 bytes (BF16) / 4 GPUs = 35GB per GPU
```

---

## Issue 4: Docker GPU Flag Format

### Problem
Docker error: `cannot set both Count and DeviceIDs on device request`

### Root Cause
When using `subprocess.create_subprocess_exec()` (no shell), the `--gpus` flag value needs embedded quotes for Docker CLI to parse correctly.

### Solution
```python
# Wrong - no quotes
cmd.extend(["--gpus", f"device={device_ids}"])

# Correct - embedded quotes
cmd.extend(["--gpus", f'"device={device_ids}"'])
```

### Verification
```bash
# Test from command line
docker run --rm --gpus '"device=4,5"' alpine echo "test"  # Works
docker run --rm --gpus 'device=4,5' alpine echo "test"    # Fails
```

---

## Issue 5: Smart GPU Selection

### Problem
Deployment was selecting GPUs 0,1 which were already in use by other processes.

### Root Cause
1. `find_available_gpus()` lacked Docker fallback for nvidia-smi
2. Backend/worker containers don't have nvidia-smi installed
3. Fallback returned `list(range(count))` → always [0, 1, ...]

### Solution
Added Docker fallback to query GPU info:
```python
async def find_available_gpus(self, count: int) -> List[int]:
    stdout = None

    # Try direct nvidia-smi first
    try:
        process = await asyncio.create_subprocess_exec(
            "nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, _ = await asyncio.wait_for(process.communicate(), timeout=10)
        if process.returncode == 0 and out:
            stdout = out
    except Exception:
        pass

    # Fallback: run nvidia-smi via Docker container
    if not stdout:
        process = await asyncio.create_subprocess_exec(
            "docker", "run", "--rm", "--gpus", "all",
            "nvidia/cuda:12.1.0-base-ubuntu22.04",
            "nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        # ... parse and select least-loaded GPUs
```

---

## Issue 6: vLLM V1 Engine Crash

### Problem
vLLM V1 engine crashes with `FileNotFoundError: [Errno 2] No such file or directory: '<frozen os>'` during torchdynamo compilation.

### Root Cause
Bug in vLLM V1 engine's torchdynamo compilation backend when initializing KV cache with tensor parallelism.

### Solution
Use V0 engine by setting environment variable:
```bash
docker run -e VLLM_USE_V1=0 ...
```

Or in deployment config:
```json
{
  "environment_vars": {
    "VLLM_USE_V1": "0",
    "TENSOR_PARALLEL": "4"
  }
}
```

---

## Issue 7: Healthcheck Restart Loop

### Problem
Container keeps restarting during model loading because healthcheck fails.

### Root Cause
Docker image healthcheck has 120s start-period, but 70B model takes 10-15 minutes to load.

### Solution
Override healthcheck when starting container:
```bash
docker run \
  --health-start-period=900s \
  --health-interval=60s \
  --health-retries=10 \
  ...
```

Or update Dockerfile:
```dockerfile
HEALTHCHECK --interval=60s --timeout=10s --start-period=900s --retries=10 \
    CMD curl -f http://localhost:${PORT}/health || exit 1
```

---

## Quick Reference: 70B Model Deployment

### Minimum Requirements
- 4x A100 40GB GPUs (or 2x A100 80GB)
- TP=4 for 40GB GPUs
- ~15 minutes startup time

### Recommended Configuration
```json
{
  "gpu_count": 4,
  "gpu_enabled": true,
  "environment_vars": {
    "TENSOR_PARALLEL": "4",
    "VLLM_USE_V1": "0",
    "GPU_MEMORY_UTIL": "0.9"
  }
}
```

### Docker Run Command
```bash
docker run -d \
  --name deployment-xxx \
  --gpus '"device=4,5,6,7"' \
  --health-start-period=900s \
  -p 8000:8000 \
  -e TENSOR_PARALLEL=4 \
  -e VLLM_USE_V1=0 \
  model-registry/your-model:tag
```

---

## Files Modified

| File | Change |
|------|--------|
| `backend/app/services/deployment/local_executor.py` | Added `find_available_gpus()`, fixed GPU flag format |
| `backend/app/services/docker/workspace_service.py` | Added `normalize_ceph_path()`, fail-fast on missing model |
| `backend/app/services/filesystem_sync_service.py` | Applied `normalize_ceph_path()` to all model/version path assignments |
| `backend/app/services/garbage_collector.py` | Converted shell commands to `subprocess_exec` for security |
| `docker-compose.yml` | Fixed `CEPH_MOUNT_PATH`, added `/ceph/models` mount |

---

## Systematic Issues Found & Fixed

After the initial troubleshooting, a codebase audit found similar issues:

### Path Normalization in filesystem_sync_service.py
**Problem:** Model paths from external sources (e.g., `/ceph/models/...`) were stored directly in database without normalization, causing path resolution failures when containers use different mount points.

**Lines Fixed:** 224, 231, 262, 279

**Pattern:**
```python
# Before
model.storage_path = config.model_path

# After
model.storage_path = normalize_ceph_path(config.model_path)
```

### Shell Commands in garbage_collector.py
**Problem:** Docker commands used `create_subprocess_shell()` instead of `create_subprocess_exec()`.

**Lines Fixed:** 69-73, 78-82

**Pattern:**
```python
# Before (shell injection risk)
proc = await asyncio.create_subprocess_shell(
    "docker builder prune -f --filter until=24h",
    ...
)

# After (secure)
proc = await asyncio.create_subprocess_exec(
    "docker", "builder", "prune", "-f", "--filter", "until=24h",
    ...
)
```

### GPU Flag Format Verification
**Note:** The embedded quotes in GPU flag (`f'"device={device_ids}"'`) are **correct** for `subprocess.exec`. Docker CLI requires quotes around device specifications when not using shell interpretation.

```python
# Correct for subprocess.exec (no shell)
cmd.extend(["--gpus", f'"device={device_ids}"'])

# This passes literally: --gpus "device=4,5"
# Docker parses correctly
```
