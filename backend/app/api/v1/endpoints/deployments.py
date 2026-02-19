"""
API endpoints for deployments.

Uses DeploymentRepository for data access and domain exceptions for error handling.
"""
import json
import re
from dataclasses import asdict
from typing import Optional
from uuid import UUID
from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import verify_api_key, require_operator
from app.models.api_key import ApiKey
from app.repositories.deployment_repository import DeploymentRepository
from app.schemas.deployment import (
    DeploymentCreate,
    DeploymentResponse,
    DeploymentWithRelease,
    DeploymentExecuteCreate,
    DeploymentExecuteResponse,
    ContainerStatusResponse,
    DeploymentLogsResponse,
    DeploymentType,
)
from app.schemas.pagination import PaginatedResponse
from app.services.audit_service import create_audit_log
from app.services.deployment_sync_service import deployment_sync_service
from app.services.deployment.deployment_service import deployment_service
from app.services.deployment.executor_base import DeploymentConfig
from app.services.deployment.local_executor import local_executor
from app.repositories.version_repository import VersionRepository
from app.repositories.model_repository import ModelRepository
from app.services.task_dispatcher import task_dispatcher

router = APIRouter()


def calculate_tensor_parallel(parameter_count: Optional[str], model_name: Optional[str] = None) -> int:
    """
    Calculate tensor parallelism based on model parameter count or model name.

    For BF16 models on 40GB GPUs:
    - 70B+ models need ~140GB+ VRAM → tp=4
    - 30B-69B models need ~60-140GB VRAM → tp=2
    - <30B models fit on single GPU → tp=1
    """
    params = None

    # Try to get params from parameter_count field first
    if parameter_count:
        param_str = parameter_count.lower().replace("b", "").strip()
        try:
            params = float(param_str)
        except ValueError:
            pass

    # If not found, try to extract from model name (e.g., "llama-3-70b-240629")
    if params is None and model_name:
        # Match patterns like "70b", "70B", "7b", "72b", "14b" in model name
        match = re.search(r'(\d+)[bB]', model_name)
        if match:
            try:
                params = float(match.group(1))
            except ValueError:
                pass

    if params is None:
        return 1

    if params >= 70:
        return 4
    elif params >= 30:
        return 2
    else:
        return 1


@router.post("", response_model=DeploymentResponse, status_code=status.HTTP_201_CREATED)
async def create_deployment(
    deployment_data: DeploymentCreate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_operator),
) -> DeploymentResponse:
    """
    Record a new deployment. Requires operator role.

    Raises:
        ReleaseNotFoundError: If release not found (404)
    """
    repo = DeploymentRepository(db)

    deployment = await repo.create_deployment(
        release_id=deployment_data.release_id,
        environment=deployment_data.environment,
        deployed_by=api_key.name,
        api_key_id=api_key.id,
        status=deployment_data.status,
        metadata=deployment_data.metadata,
    )

    await create_audit_log(
        db=db,
        action="create_deployment",
        resource_type="deployment",
        resource_id=deployment.id,
        api_key_name=api_key.name,
        api_key_id=api_key.id,
        details={
            "environment": deployment.environment,
            "release_id": str(deployment_data.release_id),
            "status": deployment.status,
        },
        ip_address=request.client.host if request.client else None,
    )

    # Materialize deployment to filesystem for recovery
    if deployment.version and deployment.version.model:
        await deployment_sync_service.materialize_deployment(
            deployment=deployment,
            model_name=deployment.version.model.name,
            release_version=deployment.version.version,
        )

    return DeploymentResponse.from_deployment(deployment)


@router.get("", response_model=PaginatedResponse[DeploymentWithRelease])
async def list_deployments(
    environment: Optional[str] = Query(None, description="Filter by environment"),
    status: Optional[str] = Query(None, description="Filter by status"),
    release_id: Optional[UUID] = Query(None, description="Filter by release ID"),
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(10, ge=1, le=100, description="Page size"),
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(verify_api_key),
) -> PaginatedResponse[DeploymentWithRelease]:
    """List deployments with optional filters (paginated)."""
    repo = DeploymentRepository(db)
    deployments, total = await repo.list_deployments(
        environment=environment,
        status=status,
        release_id=release_id,
        page=page,
        size=size,
    )

    items = [DeploymentWithRelease.from_deployment(deployment) for deployment in deployments]

    return PaginatedResponse.create(items=items, total=total, page=page, size=size)


@router.get("/{deployment_id}", response_model=DeploymentWithRelease)
async def get_deployment(
    deployment_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(verify_api_key),
) -> DeploymentWithRelease:
    """
    Get a deployment by ID.

    Raises:
        DeploymentNotFoundError: If deployment not found (404)
    """
    repo = DeploymentRepository(db)
    deployment = await repo.get_by_id_or_raise(deployment_id)

    return DeploymentWithRelease.from_deployment(deployment)


# =============================================================================
# Deployment Execution Endpoints
# =============================================================================

@router.post("/execute", response_model=DeploymentExecuteResponse, status_code=status.HTTP_201_CREATED)
async def create_and_execute_deployment(
    deployment_data: DeploymentExecuteCreate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_operator),
) -> DeploymentExecuteResponse:
    """
    Create and execute a deployment. Requires operator role.

    This will:
    1. Create a deployment record
    2. Get the Docker image tag for the version
    3. Auto-detect GPU and enable if model requires it
    4. Start a container via Celery task

    Raises:
        VersionNotFoundError: If version not found (404)
        DockerImageNotFoundError: If no Docker image exists for version (404)
    """
    # Get the Docker image tag
    image_tag = await deployment_service.get_docker_image_tag(db, deployment_data.release_id)

    # Auto-detect GPU and tensor parallelism based on model requirements
    gpu_enabled = deployment_data.gpu_enabled
    tensor_parallel = 1
    model = None

    if deployment_data.deployment_type != DeploymentType.METADATA:
        # Get version to find model
        version_repo = VersionRepository(db)
        version = await version_repo.get_by_id(deployment_data.release_id)
        if version and version.image_id:
            # Get model to check requires_gpu and parameter_count
            model_repo = ModelRepository(db)
            model = await model_repo.get_by_id(version.image_id)
            if model:
                # Calculate tensor parallelism based on model size (from parameter_count or model name)
                tensor_parallel = calculate_tensor_parallel(model.parameter_count, model.name)

                if model.requires_gpu:
                    # Model requires GPU - check if available
                    gpu_available, gpu_count = await local_executor.detect_gpu()
                    if gpu_available:
                        gpu_enabled = True  # Auto-enable GPU

    # Create deployment record
    deployment = await deployment_service.create_deployment(
        db=db,
        release_id=deployment_data.release_id,
        environment=deployment_data.environment,
        deployment_type=deployment_data.deployment_type.value,
        api_key_id=api_key.id,
        deployed_by=api_key.name,
        gpu_enabled=gpu_enabled,
        metadata=deployment_data.metadata,
        image_tag=image_tag,
    )

    # Create audit log
    await create_audit_log(
        db=db,
        action="execute_deployment",
        resource_type="deployment",
        resource_id=deployment.id,
        api_key_name=api_key.name,
        api_key_id=api_key.id,
        details={
            "environment": deployment.environment,
            "release_id": str(deployment_data.release_id),
            "deployment_type": deployment_data.deployment_type.value,
            "image_tag": image_tag,
            "gpu_enabled": gpu_enabled,  # Use auto-detected value
            "gpu_auto_enabled": gpu_enabled and not deployment_data.gpu_enabled,  # Track if auto-enabled
            "tensor_parallel": tensor_parallel,
            "parameter_count": model.parameter_count if model else None,
        },
        ip_address=request.client.host if request.client else None,
    )

    # Only trigger async execution for non-metadata deployments
    if deployment_data.deployment_type != DeploymentType.METADATA:
        # Build environment vars with TENSOR_PARALLEL
        environment_vars = dict(deployment_data.environment_vars)
        environment_vars["TENSOR_PARALLEL"] = str(tensor_parallel)

        # For multi-GPU deployments, configure for stability
        if tensor_parallel > 1:
            # Disable torch.compile to avoid dynamo issues
            environment_vars["TORCH_COMPILE_DISABLE"] = "1"
            # Reduce GPU memory utilization to handle residual memory from other processes
            environment_vars["GPU_MEMORY_UTIL"] = "0.8"

        # gpu_count must be at least tensor_parallel for proper GPU allocation
        gpu_count = max(deployment_data.gpu_count, tensor_parallel)

        # Build config - use auto-detected gpu_enabled and calculated tensor_parallel
        config = DeploymentConfig(
            image_tag=image_tag,
            environment_vars=environment_vars,
            volume_mounts=deployment_data.volume_mounts,
            gpu_enabled=gpu_enabled,  # Use auto-detected value
            gpu_count=gpu_count,
            memory_limit=deployment_data.memory_limit,
            cpu_limit=deployment_data.cpu_limit,
        )

        # Store config in deployment meta_data for restart capability
        repo = DeploymentRepository(db)
        await repo.update_metadata(deployment.id, {
            "environment_vars": environment_vars,
            "volume_mounts": deployment_data.volume_mounts,
            "gpu_count": gpu_count,
            "memory_limit": deployment_data.memory_limit,
            "cpu_limit": deployment_data.cpu_limit,
        })

        # Trigger async deployment
        task_dispatcher.dispatch_deployment(deployment.id, json.dumps(asdict(config)))

    return DeploymentExecuteResponse.from_deployment(deployment)


@router.post("/{deployment_id}/start", response_model=DeploymentExecuteResponse)
async def start_deployment(
    deployment_id: UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_operator),
) -> DeploymentExecuteResponse:
    """
    Start a stopped or failed deployment. Requires operator role.

    Raises:
        DeploymentNotFoundError: If deployment not found (404)
        DeploymentAlreadyRunningError: If already running (400)
    """
    from app.core.exceptions import DeploymentAlreadyRunningError, DeploymentExecutionError

    deployment = await deployment_service.get_deployment_by_id(db, deployment_id)

    # Validate deployment can be started
    if deployment.status == "running":
        raise DeploymentAlreadyRunningError(str(deployment_id))
    if deployment.status not in ("stopped", "failed"):
        raise DeploymentExecutionError(
            str(deployment_id),
            f"Cannot start deployment in '{deployment.status}' state"
        )

    # Get the Docker image tag
    image_tag = await deployment_service.get_docker_image_tag(db, deployment.release_id)

    # Build config from deployment metadata
    config = DeploymentConfig(
        image_tag=image_tag,
        environment_vars=deployment.meta_data.get("environment_vars", {}),
        volume_mounts=deployment.meta_data.get("volume_mounts", {}),
        gpu_enabled=deployment.gpu_enabled,
        gpu_count=deployment.meta_data.get("gpu_count", 1),
        memory_limit=deployment.meta_data.get("memory_limit"),
        cpu_limit=deployment.meta_data.get("cpu_limit"),
    )

    # Update status to pending BEFORE dispatching task to prevent race condition
    repo = DeploymentRepository(db)
    deployment = await repo.update_status(deployment_id, "pending")

    # Trigger async deployment
    task_dispatcher.dispatch_deployment(deployment_id, json.dumps(asdict(config)))

    await create_audit_log(
        db=db,
        action="start_deployment",
        resource_type="deployment",
        resource_id=deployment.id,
        api_key_name=api_key.name,
        api_key_id=api_key.id,
        details={"image_tag": image_tag},
        ip_address=request.client.host if request.client else None,
    )

    return DeploymentExecuteResponse.from_deployment(deployment)


@router.post("/{deployment_id}/stop", response_model=DeploymentExecuteResponse)
async def stop_deployment(
    deployment_id: UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_operator),
) -> DeploymentExecuteResponse:
    """
    Stop a running deployment. Requires operator role.

    Raises:
        DeploymentNotFoundError: If deployment not found (404)
        DeploymentNotRunningError: If not running (400)
    """
    from app.core.exceptions import DeploymentNotRunningError

    deployment = await deployment_service.get_deployment_by_id(db, deployment_id)

    # Validate deployment can be stopped
    if deployment.status != "running":
        raise DeploymentNotRunningError(str(deployment_id), deployment.status)

    # Update status to stopping BEFORE dispatching task to avoid race condition
    repo = DeploymentRepository(db)
    deployment = await repo.update_status(deployment_id, "stopping")

    # Trigger async stop
    task_dispatcher.dispatch_stop(deployment_id)

    await create_audit_log(
        db=db,
        action="stop_deployment",
        resource_type="deployment",
        resource_id=deployment.id,
        api_key_name=api_key.name,
        api_key_id=api_key.id,
        details={"container_id": deployment.container_id},
        ip_address=request.client.host if request.client else None,
    )

    return DeploymentExecuteResponse.from_deployment(deployment)


@router.post("/{deployment_id}/restart", response_model=DeploymentExecuteResponse)
async def restart_deployment(
    deployment_id: UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_operator),
) -> DeploymentExecuteResponse:
    """
    Restart a deployment. Requires operator role.

    Raises:
        DeploymentNotFoundError: If deployment not found (404)
        DeploymentExecutionError: If deployment cannot be restarted (400)
    """
    from app.core.exceptions import DeploymentExecutionError

    repo = DeploymentRepository(db)
    deployment = await deployment_service.get_deployment_by_id(db, deployment_id)

    # Validate deployment can be restarted (not in transitional state)
    if deployment.status not in ("running", "stopped", "failed"):
        raise DeploymentExecutionError(
            str(deployment_id),
            f"Cannot restart deployment in '{deployment.status}' state"
        )

    # Get the Docker image tag
    image_tag = await deployment_service.get_docker_image_tag(db, deployment.release_id)

    # Get environment_vars from meta_data, or recalculate if not stored
    environment_vars = deployment.meta_data.get("environment_vars", {})
    gpu_count = deployment.meta_data.get("gpu_count", 1)

    if not environment_vars.get("TENSOR_PARALLEL"):
        # Calculate tensor parallelism if not stored (for older deployments)
        tensor_parallel = 1
        version_repo = VersionRepository(db)
        version = await version_repo.get_by_id(deployment.release_id)
        if version and version.image_id:
            model_repo = ModelRepository(db)
            model = await model_repo.get_by_id(version.image_id)
            if model:
                tensor_parallel = calculate_tensor_parallel(model.parameter_count, model.name)
                environment_vars["TENSOR_PARALLEL"] = str(tensor_parallel)
                if tensor_parallel > 1:
                    environment_vars["TORCH_COMPILE_DISABLE"] = "1"
                    environment_vars["GPU_MEMORY_UTIL"] = "0.8"
                gpu_count = max(gpu_count, tensor_parallel)

        # Store updated config for future restarts
        deployment = await repo.update_metadata(deployment_id, {
            "environment_vars": environment_vars,
            "gpu_count": gpu_count,
        })

    # Build config
    config = DeploymentConfig(
        image_tag=image_tag,
        environment_vars=environment_vars,
        volume_mounts=deployment.meta_data.get("volume_mounts", {}),
        gpu_enabled=deployment.gpu_enabled,
        gpu_count=gpu_count,
        memory_limit=deployment.meta_data.get("memory_limit"),
        cpu_limit=deployment.meta_data.get("cpu_limit"),
    )

    # Update status to deploying BEFORE dispatching task (like /stop sets "stopping")
    deployment = await repo.update_status(deployment_id, "deploying")

    # Trigger async restart
    task_dispatcher.dispatch_restart(deployment_id, json.dumps(asdict(config)))

    await create_audit_log(
        db=db,
        action="restart_deployment",
        resource_type="deployment",
        resource_id=deployment.id,
        api_key_name=api_key.name,
        api_key_id=api_key.id,
        details={"image_tag": image_tag},
        ip_address=request.client.host if request.client else None,
    )

    return DeploymentExecuteResponse.from_deployment(deployment)


@router.get("/{deployment_id}/status", response_model=ContainerStatusResponse)
async def get_deployment_status(
    deployment_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(verify_api_key),
) -> ContainerStatusResponse:
    """
    Get the container status for a deployment.

    Raises:
        DeploymentNotFoundError: If deployment not found (404)
    """
    status = await deployment_service.get_deployment_status(db, deployment_id)

    return ContainerStatusResponse(
        running=status.running,
        healthy=status.healthy,
        exit_code=status.exit_code,
        started_at=status.started_at,
        error=status.error,
    )


@router.get("/{deployment_id}/logs", response_model=DeploymentLogsResponse)
async def get_deployment_logs(
    deployment_id: UUID,
    tail: int = Query(100, ge=1, le=10000, description="Number of lines to retrieve"),
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(verify_api_key),
) -> DeploymentLogsResponse:
    """
    Get logs from a deployment's container.

    Raises:
        DeploymentNotFoundError: If deployment not found (404)
    """
    logs = await deployment_service.get_deployment_logs(db, deployment_id, tail)

    return DeploymentLogsResponse(
        deployment_id=deployment_id,
        logs=logs,
        truncated=len(logs) >= tail * 200,  # Rough estimate
    )


@router.get("/{deployment_id}/logs/stream")
async def stream_deployment_logs(
    deployment_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(verify_api_key),
):
    """
    Stream logs from a deployment's container via Server-Sent Events.

    Raises:
        DeploymentNotFoundError: If deployment not found (404)
    """
    deployment = await deployment_service.get_deployment_by_id(db, deployment_id)

    if not deployment.container_id:
        async def empty_stream():
            yield "data: No container ID - deployment may not have started\n\n"

        return StreamingResponse(
            empty_stream(),
            media_type="text/event-stream",
        )

    executor = deployment_service.get_executor(deployment.deployment_type)

    async def log_stream():
        try:
            async for line in executor.stream_logs(deployment.container_id):
                yield f"data: {line}\n\n"
        except Exception as e:
            yield f"data: Error streaming logs: {e}\n\n"

    return StreamingResponse(
        log_stream(),
        media_type="text/event-stream",
    )


@router.get("/{deployment_id}/health")
async def check_deployment_health(
    deployment_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(verify_api_key),
) -> dict:
    """
    Check if a deployment is healthy via HTTP endpoint.

    Raises:
        DeploymentNotFoundError: If deployment not found (404)
    """
    deployment = await deployment_service.get_deployment_by_id(db, deployment_id)

    if not deployment.endpoint_url:
        return {
            "healthy": False,
            "reason": "No endpoint URL configured",
        }

    executor = deployment_service.get_executor(deployment.deployment_type)
    healthy = await executor.health_check(deployment.endpoint_url)

    # Update health status in database
    repo = DeploymentRepository(db)
    deployment = await repo.update_health_status(deployment_id, healthy)

    return {
        "healthy": healthy,
        "endpoint_url": deployment.endpoint_url,
        "health_status": deployment.health_status,
    }


@router.get("/{deployment_id}/api-spec")
async def get_deployment_api_spec(
    deployment_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(verify_api_key),
) -> dict:
    """
    Discover API endpoints from a running deployment.

    Probes the container for OpenAPI spec and common endpoints.

    Raises:
        DeploymentNotFoundError: If deployment not found (404)
    """
    deployment = await deployment_service.get_deployment_by_id(db, deployment_id)

    if not deployment.endpoint_url:
        return {
            "api_type": "unknown",
            "endpoints": [],
            "detected_endpoints": [],
            "error": "No endpoint URL - deployment may not be running",
        }

    if deployment.status != "running":
        return {
            "api_type": "unknown",
            "endpoints": [],
            "detected_endpoints": [],
            "error": f"Deployment is not running (status: {deployment.status})",
        }

    # Discover API spec from the running container
    result = await local_executor.discover_api_spec(deployment.endpoint_url)

    return result
