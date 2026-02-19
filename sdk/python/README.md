# Catapult SDK

Python SDK for Catapult - manage ML models, releases, and deployments.

## Installation

```bash
pip install bso
```

## Quick Start

```python
from bso import Registry

# Initialize client
registry = Registry(
    base_url="https://your-registry.example.com/api",
    api_key="your-api-key"
)

# Or use environment variables
# REGISTRY_URL=https://your-registry.example.com/api
# REGISTRY_API_KEY=your-api-key
registry = Registry.from_env()

# List models
models = registry.list_models()

# Create a model
model = registry.create_model(
    name="my-model",
    storage_path="s3://bucket/models/my-model",
    description="My ML model"
)

# Create a release
release = registry.create_release(
    model_name="my-model",
    version="1.0.0",
    tag="v1.0.0",
    digest="sha256:abc123...",
    metadata={"accuracy": 0.95}
)

# Deploy a release
deployment = registry.deploy(
    release_id=release.id,
    environment="production"
)

# Get latest release for an environment
latest = registry.get_latest_release(
    model_name="my-model",
    environment="production"
)
```

## Features

- Model management (create, list, get, delete)
- Release versioning with metadata
- Deployment tracking across environments
- Docker build triggering and status
- Artifact management
- Audit logging

## Requirements

- Python 3.8+
- httpx
- pydantic

## License

MIT
