# Changelog

## 1.0.0 — Initial Open Source Release

The first public release of Catapult, an open-source MLOps platform for managing the full lifecycle of ML models — from registration through deployment to production.

### Features

- **Model & Version Management** — Register models, track versions, promote releases with flexible JSONB metadata
- **Docker Build System** — 25+ specialized Dockerfile templates covering vLLM, ASR, TTS, embedding, and multimodal models across CUDA, CPU, ROCm, ARM, Neuron, TPU, and Gaudi platforms
- **One-Click Deployments** — GPU-aware container deployments with health monitoring and log tailing
- **Performance Benchmarking** — Latency and throughput benchmarks with TTFT, p50/p95/p99, tokens/second tracking
- **Quality Evaluation** — Pluggable evaluation framework with ASR evaluation (WER, CER) out of the box
- **Artifact Management** — Upload wheels, checkpoints, configs with SHA256 verification and filesystem browsing
- **GPU Fleet Visibility** — See which models run on which machines and GPUs
- **MLflow Bridge** — Link versions to MLflow runs and experiments
- **API Key Authentication** — Role-based access with operator and viewer roles
- **Audit Logging** — Complete operation audit trail
- **Real-time Log Streaming** — SSE-based live log streaming for builds and deployments
