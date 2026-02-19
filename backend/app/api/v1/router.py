"""
API v1 router that includes all endpoint routers.
"""
from fastapi import APIRouter

from app.api.v1.endpoints import (
    models,
    versions,
    deployments,
    api_keys,
    audit,
    artifacts,
    system,
    docker,
    benchmarks,
    evaluations,
    internal_benchmark,
    # Note: asr endpoint disabled - requires heavy dependencies (numpy, torch, etc.)
    # WER evaluation now in separate /evaluations endpoint
)

# Create main API router
api_router = APIRouter()

# Include endpoint routers
api_router.include_router(
    models.router,
    prefix="/models",
    tags=["models"],
)

api_router.include_router(
    versions.router,
    prefix="/versions",
    tags=["versions"],
)

api_router.include_router(
    deployments.router,
    prefix="/deployments",
    tags=["deployments"],
)

api_router.include_router(
    api_keys.router,
    prefix="/api-keys",
    tags=["api-keys"],
)

api_router.include_router(
    audit.router,
    prefix="/audit-logs",
    tags=["audit"],
)

api_router.include_router(
    artifacts.router,
    prefix="/artifacts",
    tags=["artifacts"],
)

api_router.include_router(
    system.router,
    prefix="/system",
    tags=["system"],
)

api_router.include_router(
    docker.router,
    prefix="/docker",
    tags=["docker"],
)

api_router.include_router(
    benchmarks.router,
    prefix="/benchmarks",
    tags=["benchmarks"],
)

api_router.include_router(
    evaluations.router,
    prefix="/evaluations",
    tags=["evaluations"],
)

# Internal benchmark callbacks (called by benchmarker container)
api_router.include_router(
    internal_benchmark.router,
    prefix="/internal/benchmarks",
    tags=["internal"],
)

# ASR endpoint disabled - requires heavy dependencies (numpy, torch, etc.)
# api_router.include_router(
#     asr.router,
#     prefix="/asr",
#     tags=["asr"],
# )
