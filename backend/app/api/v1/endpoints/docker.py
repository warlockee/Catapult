"""
API endpoints for Docker builds.

Uses DockerBuildRepository for data access and domain exceptions for error handling.
"""
import os
from typing import Optional
from uuid import UUID
from fastapi import APIRouter, Depends, BackgroundTasks, Query, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, async_session_maker
from app.core.config import settings
from app.core.security import verify_api_key, require_operator
from app.core.exceptions import InvalidPathError, TemplateNotFoundError
from app.models.api_key import ApiKey
from app.repositories.docker_build_repository import DockerBuildRepository
from app.repositories.version_repository import VersionRepository
from app.schemas.docker_build import DockerBuildCreate, DockerBuildResponse, DockerDiskUsageResponse, DockerDiskUsageComponent
from app.schemas.pagination import PaginatedResponse
from app.services.docker_service import docker_service
from app.services.audit_service import create_audit_log
from app.services.task_dispatcher import task_dispatcher

router = APIRouter()

# Whitelist of allowed template types to prevent path traversal
ALLOWED_TEMPLATE_TYPES = {
    "default": "Dockerfile",
    "organic": "Dockerfile",  # Alias for default
    "azure": "Dockerfile.maap",
    "optimized": "Dockerfile.optimized",
    "test": "Dockerfile.test",
    "asr-vllm": "Dockerfile.asr_vllm",  # ASR with vLLM - raw audio API, client handles VAD
    "asr-allinone": "Dockerfile.asr_allinone",  # ASR all-in-one - simple file upload API with VAD
    "asr-azure-allinone": "Dockerfile.asr_azure_allinone",  # ASR Azure all-in-one - Azure ML base with VAD
}


@router.post("/builds", response_model=DockerBuildResponse, status_code=status.HTTP_201_CREATED)
async def create_build(
    build_data: DockerBuildCreate,
    request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_operator),
):
    """
    Trigger a new Docker build. Requires operator role.
    """
    build = await docker_service.create_build(
        db=db,
        release_id=build_data.release_id,
        artifact_id=build_data.artifact_id,
        artifact_ids=build_data.artifact_ids,
        image_tag=build_data.image_tag,
        build_type=build_data.build_type,
        dockerfile_content=build_data.dockerfile_content
    )

    await create_audit_log(
        db=db,
        action="create_docker_build",
        resource_type="docker_build",
        resource_id=build.id,
        api_key_name=api_key.name,
        api_key_id=api_key.id,
        details={
            "release_id": str(build_data.release_id),
            "image_tag": build_data.image_tag,
            "build_type": build_data.build_type,
        },
        ip_address=request.client.host if request.client else None,
    )

    # Trigger async build
    task_dispatcher.dispatch_docker_build(build.id)

    return build


@router.post("/builds/{build_id}/cancel", response_model=DockerBuildResponse)
async def cancel_build(
    build_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_operator),
):
    """
    Cancel a running Docker build.
    Requires operator role.
    """
    try:
        return await docker_service.cancel_build(db, build_id)
    except ValueError:
        # Service raises ValueError if not found (simple handling)
        from app.core.exceptions import DockerBuildNotFoundError
        raise DockerBuildNotFoundError(build_id)



@router.get("/templates/{template_type}")
async def get_dockerfile_template(
    template_type: str,
    release_id: Optional[UUID] = Query(None, description="Release ID to auto-detect server_type"),
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(verify_api_key),
):
    """
    Get default Dockerfile template.
    Requires authentication and uses whitelist to prevent path traversal.

    If release_id is provided, the server_type from the model will override the template_type
    for specialized builds (e.g., asr-vllm).

    Raises:
        InvalidPathError: If template type is not allowed (400)
        TemplateNotFoundError: If template file not found (404)
    """
    # Check if we should override based on model's server_type or model config
    # Only auto-detect if user requested a generic type (organic/default)
    # If user explicitly requested asr-vllm or asr-allinone, respect that choice
    effective_template_type = template_type
    asr_types = {"asr-vllm", "asr-allinone", "asr-azure-allinone"}
    should_auto_detect = template_type in ("organic", "default") or template_type not in ALLOWED_TEMPLATE_TYPES

    if release_id and should_auto_detect:
        import json

        version_repo = VersionRepository(db)
        version = await version_repo.get_by_id_with_model(release_id)
        if version and version.model:
            model = version.model
            # First check server_type
            if model.server_type and model.server_type in ALLOWED_TEMPLATE_TYPES:
                effective_template_type = model.server_type
            # Also detect audio models by reading config.json
            elif model.storage_path:
                config_path = os.path.join(model.storage_path, "config.json")
                try:
                    with open(config_path, "r") as f:
                        config = json.load(f)
                        model_type = config.get("model_type", "").lower()
                        audio_model_types = {"higgs_audio_3", "higgs_audio", "higgs_audio_2"}
                        if model_type in audio_model_types:
                            effective_template_type = "asr-vllm"
                except (FileNotFoundError, json.JSONDecodeError):
                    pass  # Use original template_type

    if effective_template_type not in ALLOWED_TEMPLATE_TYPES:
        raise InvalidPathError(
            effective_template_type,
            f"Invalid template type. Allowed types: {', '.join(ALLOWED_TEMPLATE_TYPES.keys())}"
        )

    filename = ALLOWED_TEMPLATE_TYPES[effective_template_type]
    template_path = os.path.join(settings.DOCKER_TEMPLATES_PATH, filename)

    # Additional safety: ensure resolved path is within templates directory
    resolved_path = os.path.realpath(template_path)
    templates_dir = os.path.realpath(settings.DOCKER_TEMPLATES_PATH)
    if not resolved_path.startswith(templates_dir):
        raise InvalidPathError(effective_template_type, "Invalid template path")

    try:
        with open(resolved_path, "r") as f:
            return {"content": f.read(), "template_type": effective_template_type}
    except FileNotFoundError:
        raise TemplateNotFoundError(effective_template_type)


@router.get("/builds", response_model=PaginatedResponse[DockerBuildResponse])
async def list_builds(
    release_id: Optional[UUID] = Query(None, description="Filter by release ID"),
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(10, ge=1, le=100, description="Page size"),
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(verify_api_key),
):
    """List Docker builds with optional filters (paginated)."""
    repo = DockerBuildRepository(db)
    builds, total = await repo.list_builds(release_id=release_id, page=page, size=size)

    return PaginatedResponse.create(items=builds, total=total, page=page, size=size)


@router.get("/builds/{build_id}", response_model=DockerBuildResponse)
async def get_build(
    build_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(verify_api_key),
):
    """
    Get build status.

    Raises:
        DockerBuildNotFoundError: If build not found (404)
    """
    repo = DockerBuildRepository(db)
    return await repo.get_by_id_or_raise(build_id)


@router.get("/builds/{build_id}/logs")
async def get_build_logs(
    build_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(verify_api_key),
):
    """
    Get build logs.

    Raises:
        DockerBuildNotFoundError: If build not found (404)
    """
    repo = DockerBuildRepository(db)
    build = await repo.get_by_id_or_raise(build_id)

    if not build.log_path:
        return {"logs": ""}

    try:
        with open(build.log_path, "r") as f:
            return {"logs": f.read()}
    except FileNotFoundError:
        return {"logs": "Log file not found"}


@router.get("/builds/{build_id}/logs/stream")
async def stream_build_logs(
    build_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(verify_api_key),
):
    """
    Stream build logs via SSE.

    Raises:
        DockerBuildNotFoundError: If build not found (404)
    """
    from app.models.docker_build import DockerBuild

    # Verify build exists first
    repo = DockerBuildRepository(db)
    await repo.get_by_id_or_raise(build_id)

    async def log_generator():
        import asyncio

        # Immediate feedback - user knows something is happening
        yield "⏳ Connecting to build worker...\n"

        # Wait for log_path to be assigned (Worker startup)
        log_path = None
        waited = 0
        last_status_msg = 0

        while not log_path:
            async with async_session_maker() as session:
                current_build = await session.get(DockerBuild, build_id)
                if not current_build:
                    yield "❌ Error: Build not found.\n"
                    return

                if current_build.log_path:
                    log_path = current_build.log_path
                    break

                # Check if build failed before getting log path
                if current_build.status == "failed":
                    yield f"❌ Build failed: {current_build.error_message or 'Unknown error'}\n"
                    return

            await asyncio.sleep(0.2)
            waited += 1

            # Send periodic status updates so user knows we're still working
            if waited - last_status_msg >= 15:  # Every 3 seconds (15 * 0.2s)
                yield f"⏳ Waiting for build worker to start... ({waited * 0.2:.0f}s)\n"
                last_status_msg = waited

            if waited > 150:  # 30 seconds timeout (150 * 0.2s)
                yield "❌ Error: Build failed to start (worker timeout).\n"
                return

        yield "✓ Build worker connected, waiting for logs...\n"

        # Initial wait for file creation
        retries = 0
        while not os.path.exists(log_path):
            await asyncio.sleep(0.2)
            retries += 1

            # Check if build failed immediately
            async with async_session_maker() as session:
                current_build = await session.get(DockerBuild, build_id)
                if current_build and current_build.status == "failed":
                    yield f"❌ Build failed: {current_build.error_message or 'Unknown error'}\n"
                    return

            if retries > 50:  # 10 seconds timeout (50 * 0.2s)
                yield "❌ Error: Log file creation timed out.\n"
                return

        yield "✓ Log stream started\n"
        yield "─" * 50 + "\n"

        try:
            with open(log_path, "r") as f:
                while True:
                    line = f.readline()
                    if line:
                        yield line
                    else:
                        # Check if build is finished using fresh session
                        async with async_session_maker() as session:
                            current_build = await session.get(DockerBuild, build_id)
                            if not current_build:
                                break
                            if current_build.status in ["success", "failed"]:
                                # Read remaining
                                rest = f.read()
                                if rest:
                                    yield rest
                                break

                        await asyncio.sleep(0.1)  # Fast polling for responsive logs
        except Exception as e:
            yield f"Error streaming logs: {str(e)}\n"

    return StreamingResponse(log_generator(), media_type="text/plain")


def _format_bytes(size_bytes: int) -> str:
    """Format bytes to human readable string."""
    if size_bytes == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    while size >= 1024 and i < len(units) - 1:
        size /= 1024
        i += 1
    return f"{size:.1f} {units[i]}"


def _parse_docker_size(size_str: str) -> int:
    """Parse Docker size string (e.g., '1.5GB', '500MB') to bytes."""
    if not size_str or size_str == "0B":
        return 0

    size_str = size_str.strip().upper()
    multipliers = {
        "B": 1,
        "KB": 1024,
        "MB": 1024 ** 2,
        "GB": 1024 ** 3,
        "TB": 1024 ** 4,
    }

    for unit, multiplier in multipliers.items():
        if size_str.endswith(unit):
            try:
                value = float(size_str[:-len(unit)])
                return int(value * multiplier)
            except ValueError:
                return 0
    return 0


@router.get("/disk-usage", response_model=DockerDiskUsageResponse)
async def get_disk_usage(
    api_key: ApiKey = Depends(verify_api_key),
):
    """
    Get Docker disk usage statistics.

    Returns information about Docker images, build cache, containers, volumes,
    and available disk space. Runs via Celery worker which has Docker socket access.
    """
    from app.worker import get_docker_disk_usage_task

    try:
        # Run task in worker (which has Docker socket access) with timeout
        result = get_docker_disk_usage_task.apply_async()
        data = result.get(timeout=10)  # Wait up to 10 seconds

        return DockerDiskUsageResponse(
            images=DockerDiskUsageComponent(
                count=data["images"]["count"],
                size_bytes=data["images"]["size_bytes"],
                size_human=data["images"]["size_human"],
            ),
            build_cache=DockerDiskUsageComponent(
                count=data["build_cache"]["count"],
                size_bytes=data["build_cache"]["size_bytes"],
                size_human=data["build_cache"]["size_human"],
            ),
            containers=DockerDiskUsageComponent(
                count=data["containers"]["count"],
                size_bytes=data["containers"]["size_bytes"],
                size_human=data["containers"]["size_human"],
            ),
            volumes=DockerDiskUsageComponent(
                count=data["volumes"]["count"],
                size_bytes=data["volumes"]["size_bytes"],
                size_human=data["volumes"]["size_human"],
            ),
            total_docker_bytes=data["total_docker_bytes"],
            total_docker_human=data["total_docker_human"],
            disk_available_bytes=data["disk_available_bytes"],
            disk_available_human=data["disk_available_human"],
            disk_total_bytes=data["disk_total_bytes"],
            disk_total_human=data["disk_total_human"],
        )
    except Exception as e:
        # Fallback if worker is unavailable
        import shutil
        try:
            disk = shutil.disk_usage("/")
        except Exception:
            disk = type('obj', (object,), {'free': 0, 'total': 0})()

        return DockerDiskUsageResponse(
            images=DockerDiskUsageComponent(count=0, size_bytes=0, size_human="N/A"),
            build_cache=DockerDiskUsageComponent(count=0, size_bytes=0, size_human="N/A"),
            containers=DockerDiskUsageComponent(count=0, size_bytes=0, size_human="N/A"),
            volumes=DockerDiskUsageComponent(count=0, size_bytes=0, size_human="N/A"),
            total_docker_bytes=0,
            total_docker_human="N/A",
            disk_available_bytes=disk.free,
            disk_available_human=_format_bytes(disk.free),
            disk_total_bytes=disk.total,
            disk_total_human=_format_bytes(disk.total),
        )
