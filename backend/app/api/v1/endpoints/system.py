"""
API endpoints for system information.
"""
from typing import Dict

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.core.security import verify_api_key
from app.models.api_key import ApiKey
from app.services.deployment.local_executor import local_executor
from app.services.storage_service import storage_service

router = APIRouter()


class GpuInfo(BaseModel):
    """GPU availability information."""
    available: bool
    count: int


@router.get("/gpu", response_model=GpuInfo)
async def get_gpu_info(
    api_key: ApiKey = Depends(verify_api_key),
) -> GpuInfo:
    """
    Get GPU availability information.

    Detects NVIDIA GPUs on the host using nvidia-smi.

    Returns:
        GpuInfo with available flag and GPU count
    """
    available, count = await local_executor.detect_gpu()
    return GpuInfo(available=available, count=count)


@router.get("/storage", response_model=Dict[str, int])
async def get_storage_stats(
    api_key: ApiKey = Depends(verify_api_key),
) -> Dict[str, int]:
    """
    Get storage usage statistics.

    Args:
        api_key: Verified API key

    Returns:
        Dictionary with total, used, and free space in bytes
    """
    return storage_service.get_storage_usage()


@router.get("/files", response_model=list[dict])
async def list_files(
    path: str = "/",
    api_key: ApiKey = Depends(verify_api_key),
) -> list[dict]:
    """
    List files and directories in storage.

    Args:
        path: Relative path from storage root
        api_key: Verified API key

    Returns:
        List of items with details
    """
    # Sanitize path to prevent traversal if service doesn't catch it fully
    # storage_service checks it, but good to be safe.
    if path.startswith("/"):
        path = path.lstrip("/")
    
    return storage_service.list_items(path)
