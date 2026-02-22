"""Schemas for model preprocessing."""
from typing import Dict, List, Optional

from pydantic import BaseModel


class PreprocessingRequest(BaseModel):
    """Request to run a preprocessing step on a model."""
    image: str  # Docker image to run
    command: List[str]  # Command + args
    mounts: Dict[str, str] = {}  # host_path: container_path
    gpu: bool = False


class PreprocessingResponse(BaseModel):
    """Response from a preprocessing request."""
    task_id: Optional[str] = None  # Celery task ID
    status: str  # "dispatched" or "failed"
    message: str
