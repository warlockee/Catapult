"""
Pydantic schemas for MLflow metadata.
"""
from typing import Dict, List, Optional

from pydantic import BaseModel


class MlflowModelVersionInfo(BaseModel):
    """MLflow registered model version summary."""
    version: Optional[str] = None
    current_stage: Optional[str] = None
    status: Optional[str] = None
    source: Optional[str] = None
    run_id: Optional[str] = None


class MlflowMetadataResponse(BaseModel):
    """
    Response schema for MLflow metadata endpoint.

    The shape varies by resource_type (run / experiment / registered_model).
    All types share: resource_type, url, fetched_at, tags.
    """
    resource_type: str
    url: str
    fetched_at: Optional[str] = None
    tags: Dict[str, str] = {}

    # Run fields
    run_id: Optional[str] = None
    experiment_id: Optional[str] = None
    run_name: Optional[str] = None
    status: Optional[str] = None
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    artifact_uri: Optional[str] = None
    params: Optional[Dict[str, str]] = None
    metrics: Optional[Dict[str, float]] = None

    # Experiment fields
    experiment_name: Optional[str] = None
    artifact_location: Optional[str] = None
    lifecycle_stage: Optional[str] = None

    # Registered model fields
    model_name: Optional[str] = None
    description: Optional[str] = None
    creation_timestamp: Optional[int] = None
    last_updated_timestamp: Optional[int] = None
    latest_versions: Optional[List[MlflowModelVersionInfo]] = None
    requested_version: Optional[str] = None

    model_config = {"extra": "allow"}
