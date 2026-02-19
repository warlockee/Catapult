# ASR All-in-One Azure Deployment Tracking

This document tracks the deployment of ASR All-in-One models on Azure, including configuration adjustments and troubleshooting fixes.

## Configuration Fixes

### 1. Disable vLLM V1 Engine
**Configuration**: Set `VLLM_USE_V1=0` (Default is `1`).

**Reasoning**:
The vLLM V1 engine (Isomorphic architecture) is enabled by default. However, for ASR (Audio-Language) models like `Qwen2Audio`, the V1 engine may inherently lack full support for the required multimodal processing pipeline or audio encoder integration.

Enabling the legacy engine (V0) via `VLLM_USE_V1=0` ensures:
-   **Stability**: Leverages the mature, proven execution path for complex multimodal models.
-   **Compatibility**: Bypasses potentially experimental features in V1 that might conflict with the audio encoder's attention patterns or tensor shapes.

**Status**: ✅ Fix verified.
