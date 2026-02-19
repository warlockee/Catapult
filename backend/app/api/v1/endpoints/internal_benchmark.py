"""
Internal API endpoints for benchmark progress callbacks.

These endpoints are called by the benchmarker Docker container to report
progress and results. They are not authenticated since they're only
accessible from the internal network.
"""
from typing import Any, Dict
from uuid import UUID

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.repositories.benchmark_repository import BenchmarkRepository

router = APIRouter()


class BenchmarkProgressUpdate(BaseModel):
    """Progress update from benchmarker container."""
    stage: str = Field(..., description="Current stage name")
    progress: str | None = Field(None, description="Progress within stage (e.g., '3/5')")
    data: Dict[str, Any] = Field(default_factory=dict, description="Additional stage data")
    timestamp: str | None = Field(None, description="ISO timestamp")


class BenchmarkStageComplete(BaseModel):
    """Stage completion notification from benchmarker container."""
    stage: str = Field(..., description="Stage name that completed")
    success: bool = Field(..., description="Whether the stage succeeded")
    data: Dict[str, Any] = Field(default_factory=dict, description="Stage result data")
    timestamp: str | None = Field(None, description="ISO timestamp")


class BenchmarkCompletePayload(BaseModel):
    """Final benchmark results from benchmarker container."""
    success: bool = Field(..., description="Whether the benchmark succeeded")
    results: Dict[str, Any] = Field(..., description="Final benchmark results")
    timestamp: str | None = Field(None, description="ISO timestamp")


class BenchmarkErrorPayload(BaseModel):
    """Error notification from benchmarker container."""
    error: str = Field(..., description="Error message")
    stage: str | None = Field(None, description="Stage where error occurred")
    timestamp: str | None = Field(None, description="ISO timestamp")


@router.post("/{benchmark_id}/progress", status_code=status.HTTP_200_OK)
async def update_benchmark_progress(
    benchmark_id: UUID,
    payload: BenchmarkProgressUpdate,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, str]:
    """
    Receive progress update from benchmarker container.

    Updates the benchmark's current_stage and stage_progress fields.
    """
    repo = BenchmarkRepository(db)
    await repo.update_progress(benchmark_id, payload.stage, payload.progress)
    return {"status": "ok"}


@router.post("/{benchmark_id}/stage_complete", status_code=status.HTTP_200_OK)
async def complete_benchmark_stage(
    benchmark_id: UUID,
    payload: BenchmarkStageComplete,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, str]:
    """
    Receive stage completion notification from benchmarker container.

    Adds the completed stage to stages_completed array.
    """
    repo = BenchmarkRepository(db)
    await repo.update_stage_complete(
        benchmark_id,
        payload.stage,
        payload.success,
        payload.data,
    )
    return {"status": "ok"}


@router.post("/{benchmark_id}/complete", status_code=status.HTTP_200_OK)
async def complete_benchmark(
    benchmark_id: UUID,
    payload: BenchmarkCompletePayload,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, str]:
    """
    Receive final benchmark results from benchmarker container.

    Updates all benchmark metrics and sets status to completed/failed.
    """
    repo = BenchmarkRepository(db)
    await repo.mark_completed(benchmark_id, payload.results, payload.success)
    return {"status": "ok"}


@router.post("/{benchmark_id}/error", status_code=status.HTTP_200_OK)
async def report_benchmark_error(
    benchmark_id: UUID,
    payload: BenchmarkErrorPayload,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, str]:
    """
    Receive error notification from benchmarker container.

    Marks the benchmark as failed with the error message.
    """
    repo = BenchmarkRepository(db)
    try:
        await repo.mark_failed(benchmark_id, payload.error, payload.stage)
        return {"status": "ok"}
    except Exception:
        # Benchmark may already be in terminal state
        return {"status": "ignored", "reason": "Benchmark already in terminal state"}
