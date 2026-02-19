"""
API endpoints for artifacts.

Uses ArtifactRepository for data access and domain exceptions for error handling.
"""
import os
import hashlib
from typing import List, Optional
from uuid import UUID, uuid4
from fastapi import APIRouter, Depends, Query, Request, status, UploadFile, File, Form
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.security import verify_api_key, require_operator
from app.core.exceptions import InvalidPathError, StorageError
from app.models.api_key import ApiKey
from app.repositories.artifact_repository import ArtifactRepository
from app.repositories.version_repository import VersionRepository
from app.schemas.artifact import (
    ArtifactCreate,
    ArtifactUpdate,
    ArtifactResponse,
    ArtifactWithRelease,
    ArtifactRegister,
)
from app.schemas.pagination import PaginatedResponse
from app.services.audit_service import create_audit_log

router = APIRouter()

# Get storage path from settings
STORAGE_PATH = settings.ARTIFACT_MOUNT_PATH
os.makedirs(STORAGE_PATH, exist_ok=True)

# Allowed directories for artifact registration (path traversal protection)
ALLOWED_ARTIFACT_DIRS = [
    settings.ARTIFACT_MOUNT_PATH,
    settings.CEPH_MOUNT_PATH,
    settings.SYNC_ARTIFACTS_PATH,
    settings.VLLM_WHEELS_PATH,
]

# Read-only artifact sources (browsable but not writable)
READONLY_ARTIFACT_SOURCES = {
    "vllm_wheels": {
        "path": settings.VLLM_WHEELS_PATH,
        "name": "vLLM Wheels",
        "description": "Prebuilt vLLM wheel files organized by version",
    },
}


def validate_file_path(file_path: str) -> str:
    """
    Validate that a file path is within allowed directories to prevent path traversal.

    Args:
        file_path: The file path to validate

    Returns:
        The resolved absolute path if valid

    Raises:
        InvalidPathError: If path is outside allowed directories
    """
    resolved_path = os.path.realpath(os.path.abspath(file_path))

    for allowed_dir in ALLOWED_ARTIFACT_DIRS:
        allowed_resolved = os.path.realpath(os.path.abspath(allowed_dir))
        if resolved_path.startswith(allowed_resolved + os.sep) or resolved_path == allowed_resolved:
            return resolved_path

    raise InvalidPathError(
        file_path,
        f"Path must be within allowed directories: {', '.join(ALLOWED_ARTIFACT_DIRS)}"
    )


def infer_artifact_type(filename: str) -> str:
    """Infer artifact type from filename extension."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.whl':
        return 'wheel'
    elif ext in ['.tar.gz', '.tgz', '.zip']:
        return 'sdist'
    return 'binary'


def calculate_file_checksum(file_path: str) -> tuple[str, int]:
    """
    Calculate SHA256 checksum and size of a file.

    Returns:
        Tuple of (checksum, size_bytes)
    """
    sha256_hash = hashlib.sha256()
    file_size = 0

    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256_hash.update(chunk)
            file_size += len(chunk)

    return sha256_hash.hexdigest(), file_size


@router.post("", response_model=ArtifactResponse, status_code=status.HTTP_201_CREATED)
async def create_artifact(
    artifact_data: ArtifactCreate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_operator),
) -> ArtifactResponse:
    """
    Create a new artifact. Requires operator role.

    Raises:
        ReleaseNotFoundError: If release not found (404)
        ArtifactAlreadyExistsError: If artifact with same name exists (409)
    """
    repo = ArtifactRepository(db)

    artifact = await repo.create_artifact(
        name=artifact_data.name,
        artifact_type=artifact_data.artifact_type,
        file_path=artifact_data.file_path,
        size_bytes=artifact_data.size_bytes,
        checksum=artifact_data.checksum,
        checksum_type=artifact_data.checksum_type,
        platform=artifact_data.platform,
        python_version=artifact_data.python_version,
        metadata=artifact_data.metadata,
        uploaded_by=api_key.name,
        release_id=artifact_data.release_id,
        model_id=artifact_data.model_id,
    )

    await create_audit_log(
        db=db,
        action="create_artifact",
        resource_type="artifact",
        resource_id=artifact.id,
        api_key_name=api_key.name,
        api_key_id=api_key.id,
        details={
            "name": artifact.name,
            "type": artifact.artifact_type,
            "size_bytes": artifact.size_bytes,
        },
        ip_address=request.client.host if request.client else None,
    )

    return ArtifactResponse.from_artifact(artifact)


@router.get("", response_model=PaginatedResponse[ArtifactWithRelease])
async def list_artifacts(
    release_id: Optional[UUID] = Query(None, description="Filter by release ID"),
    artifact_type: Optional[str] = Query(None, description="Filter by artifact type"),
    platform: Optional[str] = Query(None, description="Filter by platform"),
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(10, ge=1, le=100, description="Page size"),
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(verify_api_key),
) -> PaginatedResponse[ArtifactWithRelease]:
    """List artifacts with optional filters (paginated)."""
    repo = ArtifactRepository(db)
    artifacts, total = await repo.list_artifacts(
        release_id=release_id,
        artifact_type=artifact_type,
        platform=platform,
        page=page,
        size=size,
    )

    items = [ArtifactWithRelease.from_artifact(artifact) for artifact in artifacts]

    return PaginatedResponse.create(items=items, total=total, page=page, size=size)


@router.get("/{artifact_id}", response_model=ArtifactWithRelease)
async def get_artifact(
    artifact_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(verify_api_key),
) -> ArtifactWithRelease:
    """
    Get an artifact by ID.

    Raises:
        ArtifactNotFoundError: If artifact not found (404)
    """
    repo = ArtifactRepository(db)
    artifact = await repo.get_by_id_or_raise(artifact_id)

    return ArtifactWithRelease.from_artifact(artifact)


@router.put("/{artifact_id}", response_model=ArtifactResponse)
async def update_artifact(
    artifact_id: UUID,
    artifact_data: ArtifactUpdate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_operator),
) -> ArtifactResponse:
    """
    Update an artifact. Requires operator role.

    Raises:
        ArtifactNotFoundError: If artifact not found (404)
    """
    repo = ArtifactRepository(db)
    artifact = await repo.get_by_id_or_raise(artifact_id)

    artifact = await repo.update_artifact(
        artifact,
        file_path=artifact_data.file_path,
        metadata=artifact_data.metadata,
    )

    await create_audit_log(
        db=db,
        action="update_artifact",
        resource_type="artifact",
        resource_id=artifact.id,
        api_key_name=api_key.name,
        api_key_id=api_key.id,
        details={"name": artifact.name},
        ip_address=request.client.host if request.client else None,
    )

    return ArtifactResponse.from_artifact(artifact)


@router.delete("/{artifact_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_artifact(
    artifact_id: UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_operator),
) -> None:
    """
    Delete an artifact (soft delete). Requires operator role.

    Raises:
        ArtifactNotFoundError: If artifact not found (404)
    """
    repo = ArtifactRepository(db)
    artifact = await repo.get_by_id_or_raise(artifact_id)
    artifact_name = artifact.name

    await repo.soft_delete(artifact)

    await create_audit_log(
        db=db,
        action="delete_artifact",
        resource_type="artifact",
        resource_id=artifact_id,
        api_key_name=api_key.name,
        api_key_id=api_key.id,
        details={"name": artifact_name, "soft_delete": True},
        ip_address=request.client.host if request.client else None,
    )


@router.post("/upload", response_model=ArtifactResponse, status_code=status.HTTP_201_CREATED)
async def upload_artifact(
    file: UploadFile = File(...),
    release_id: Optional[str] = Form(None),
    artifact_type: Optional[str] = Form(None),
    platform: str = Form(default="any"),
    python_version: Optional[str] = Form(default=None),
    request: Request = None,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_operator),
) -> ArtifactResponse:
    """
    Upload a new artifact file. Requires operator role.

    Raises:
        ReleaseNotFoundError: If release not found (404)
        ArtifactAlreadyExistsError: If artifact with same name exists (409)
        StorageError: If file upload fails (500)
    """
    # Parse release_id
    release_uuid = None
    if release_id:
        try:
            release_uuid = UUID(release_id)
        except ValueError:
            raise InvalidPathError(release_id, "Invalid release_id format")

    repo = ArtifactRepository(db)

    # Verify release exists and get model_id
    model_id = None
    if release_uuid:
        version_repo = VersionRepository(db)
        version = await version_repo.get_by_id_or_raise(release_uuid)
        model_id = version.image_id

    # Check for duplicate name
    if await repo.check_name_exists(file.filename, model_id, release_uuid):
        from app.core.exceptions import ArtifactAlreadyExistsError
        raise ArtifactAlreadyExistsError(file.filename)

    # Generate unique file path
    file_id = str(uuid4())
    file_ext = os.path.splitext(file.filename)[1]
    storage_file_path = os.path.join(STORAGE_PATH, f"{file_id}{file_ext}")

    # Save file and calculate checksum
    sha256_hash = hashlib.sha256()
    file_size = 0

    try:
        with open(storage_file_path, "wb") as f:
            while chunk := await file.read(8192):
                sha256_hash.update(chunk)
                f.write(chunk)
                file_size += len(chunk)
    except Exception as e:
        if os.path.exists(storage_file_path):
            os.remove(storage_file_path)
        raise StorageError("file upload", str(e))

    checksum = sha256_hash.hexdigest()

    # Determine artifact type
    if not artifact_type:
        artifact_type = infer_artifact_type(file.filename)

    # Create artifact record
    artifact = await repo.create_artifact(
        name=file.filename,
        artifact_type=artifact_type,
        file_path=storage_file_path,
        size_bytes=file_size,
        checksum=checksum,
        checksum_type="sha256",
        platform=platform or "any",
        python_version=python_version,
        metadata={},
        uploaded_by=api_key.name,
        release_id=release_uuid,
        model_id=model_id,
    )

    await create_audit_log(
        db=db,
        action="upload_artifact",
        resource_type="artifact",
        resource_id=artifact.id,
        api_key_name=api_key.name,
        api_key_id=api_key.id,
        details={
            "name": artifact.name,
            "type": artifact.artifact_type,
            "size_bytes": artifact.size_bytes,
        },
        ip_address=request.client.host if request.client else None,
    )

    return ArtifactResponse.from_artifact(artifact)


@router.post("/register", response_model=ArtifactResponse, status_code=status.HTTP_201_CREATED)
async def register_artifact(
    artifact_data: ArtifactRegister,
    request: Request,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_operator),
) -> ArtifactResponse:
    """
    Register an existing artifact file from the filesystem. Requires operator role.

    Raises:
        ReleaseNotFoundError: If release not found (404)
        ArtifactAlreadyExistsError: If artifact with same name exists (409)
        InvalidPathError: If path is invalid or outside allowed directories (400)
        StorageError: If file cannot be read (500)
    """
    repo = ArtifactRepository(db)

    # Validate file path
    validated_path = validate_file_path(artifact_data.file_path)

    # Verify file exists
    if not os.path.exists(validated_path):
        raise InvalidPathError(artifact_data.file_path, "File not found")

    # Determine artifact name
    artifact_name = artifact_data.name or os.path.basename(artifact_data.file_path)

    # Calculate checksum and size
    try:
        checksum, file_size = calculate_file_checksum(validated_path)
    except Exception as e:
        raise StorageError("file read", str(e))

    # Determine artifact type
    artifact_type = artifact_data.artifact_type or infer_artifact_type(artifact_name)

    # Create artifact record
    artifact = await repo.create_artifact(
        name=artifact_name,
        artifact_type=artifact_type,
        file_path=validated_path,
        size_bytes=file_size,
        checksum=checksum,
        checksum_type="sha256",
        platform=artifact_data.platform or "any",
        python_version=artifact_data.python_version,
        metadata=artifact_data.metadata,
        uploaded_by=api_key.name,
        release_id=artifact_data.release_id,
        model_id=artifact_data.model_id,
    )

    await create_audit_log(
        db=db,
        action="register_artifact",
        resource_type="artifact",
        resource_id=artifact.id,
        api_key_name=api_key.name,
        api_key_id=api_key.id,
        details={
            "name": artifact.name,
            "type": artifact.artifact_type,
            "size_bytes": artifact.size_bytes,
            "path": artifact.file_path,
        },
        ip_address=request.client.host if request.client else None,
    )

    return ArtifactResponse.from_artifact(artifact)


@router.get("/{artifact_id}/download")
async def download_artifact(
    artifact_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(verify_api_key),
):
    """
    Download an artifact file.

    Raises:
        ArtifactNotFoundError: If artifact not found (404)
        InvalidPathError: If file path is invalid (403)
    """
    from app.core.exceptions import ArtifactNotFoundError as ArtifactNotFound

    repo = ArtifactRepository(db)
    artifact = await repo.get_by_id_or_raise(artifact_id)

    # Validate the file path
    try:
        validated_path = validate_file_path(artifact.file_path)
    except InvalidPathError:
        raise InvalidPathError(artifact.file_path, "Artifact file path is not within allowed directories")

    if not os.path.exists(validated_path):
        raise ArtifactNotFound(f"{artifact_id} (file missing from disk)")

    return FileResponse(
        path=validated_path,
        filename=artifact.name,
        media_type="application/octet-stream",
    )


# =============================================================================
# Filesystem Artifact Browser (Read-Only)
# =============================================================================

@router.get("/sources/list")
async def list_artifact_sources(
    api_key: ApiKey = Depends(verify_api_key),
) -> List[dict]:
    """
    List available read-only artifact sources.

    Returns a list of browsable artifact sources with their metadata.
    """
    sources = []
    for source_id, source_info in READONLY_ARTIFACT_SOURCES.items():
        source_path = source_info["path"]
        exists = os.path.isdir(source_path)
        sources.append({
            "id": source_id,
            "name": source_info["name"],
            "description": source_info["description"],
            "path": source_path,
            "available": exists,
            "readonly": True,
        })
    return sources


@router.get("/sources/{source_id}/browse")
async def browse_artifact_source(
    source_id: str,
    path: str = Query("", description="Relative path within the source (empty for root)"),
    api_key: ApiKey = Depends(verify_api_key),
) -> dict:
    """
    Browse contents of a read-only artifact source.

    Supports nested folder navigation. Returns folders and files at the specified path.

    Args:
        source_id: The artifact source identifier (e.g., 'vllm_wheels')
        path: Relative path within the source (empty string for root)

    Returns:
        Dictionary with current path info, breadcrumbs, folders, and files
    """
    if source_id not in READONLY_ARTIFACT_SOURCES:
        raise InvalidPathError(source_id, f"Unknown artifact source: {source_id}")

    source_info = READONLY_ARTIFACT_SOURCES[source_id]
    base_path = source_info["path"]

    # Validate and resolve the full path
    # Normalize path to prevent traversal attacks
    clean_path = os.path.normpath(path).lstrip(os.sep) if path else ""
    # Prevent path traversal
    if ".." in clean_path:
        raise InvalidPathError(path, "Path traversal not allowed")

    full_path = os.path.join(base_path, clean_path) if clean_path else base_path
    resolved_path = os.path.realpath(full_path)

    # Ensure we're still within the source directory
    if not resolved_path.startswith(os.path.realpath(base_path)):
        raise InvalidPathError(path, "Path outside source directory")

    if not os.path.exists(resolved_path):
        raise InvalidPathError(path, "Path not found")

    if not os.path.isdir(resolved_path):
        raise InvalidPathError(path, "Path is not a directory")

    # Build breadcrumbs for navigation
    breadcrumbs = [{"name": source_info["name"], "path": ""}]
    if clean_path:
        parts = clean_path.split(os.sep)
        accumulated = ""
        for part in parts:
            accumulated = os.path.join(accumulated, part) if accumulated else part
            breadcrumbs.append({"name": part, "path": accumulated})

    # List directory contents
    folders = []
    files = []

    try:
        entries = os.listdir(resolved_path)
        entries.sort()  # Sort alphabetically

        for entry in entries:
            entry_path = os.path.join(resolved_path, entry)
            relative_path = os.path.join(clean_path, entry) if clean_path else entry

            if os.path.isdir(entry_path):
                # Count items in folder
                try:
                    item_count = len(os.listdir(entry_path))
                except PermissionError:
                    item_count = 0

                folders.append({
                    "name": entry,
                    "path": relative_path,
                    "item_count": item_count,
                })
            else:
                # Get file info
                try:
                    stat_info = os.stat(entry_path)
                    file_size = stat_info.st_size
                    modified_at = stat_info.st_mtime
                except Exception:
                    file_size = 0
                    modified_at = 0

                # Determine file type
                ext = os.path.splitext(entry)[1].lower()
                file_type = "wheel" if ext == ".whl" else "unknown"

                files.append({
                    "name": entry,
                    "path": relative_path,
                    "size_bytes": file_size,
                    "modified_at": modified_at,
                    "file_type": file_type,
                })
    except PermissionError:
        raise InvalidPathError(path, "Permission denied")

    return {
        "source_id": source_id,
        "source_name": source_info["name"],
        "current_path": clean_path,
        "breadcrumbs": breadcrumbs,
        "folders": folders,
        "files": files,
        "readonly": True,
    }


@router.get("/sources/{source_id}/download")
async def download_from_source(
    source_id: str,
    path: str = Query(..., description="Relative path to the file within the source"),
    api_key: ApiKey = Depends(verify_api_key),
):
    """
    Download a file from a read-only artifact source.

    Args:
        source_id: The artifact source identifier
        path: Relative path to the file within the source

    Returns:
        FileResponse for download
    """
    if source_id not in READONLY_ARTIFACT_SOURCES:
        raise InvalidPathError(source_id, f"Unknown artifact source: {source_id}")

    source_info = READONLY_ARTIFACT_SOURCES[source_id]
    base_path = source_info["path"]

    # Validate and resolve the full path
    clean_path = os.path.normpath(path).lstrip(os.sep) if path else ""
    if ".." in clean_path or not clean_path:
        raise InvalidPathError(path, "Invalid path")

    full_path = os.path.join(base_path, clean_path)
    resolved_path = os.path.realpath(full_path)

    # Ensure we're still within the source directory
    if not resolved_path.startswith(os.path.realpath(base_path)):
        raise InvalidPathError(path, "Path outside source directory")

    if not os.path.exists(resolved_path):
        raise InvalidPathError(path, "File not found")

    if not os.path.isfile(resolved_path):
        raise InvalidPathError(path, "Path is not a file")

    filename = os.path.basename(resolved_path)
    return FileResponse(
        path=resolved_path,
        filename=filename,
        media_type="application/octet-stream",
    )


@router.get("/sources/{source_id}/files")
async def list_source_files(
    source_id: str,
    api_key: ApiKey = Depends(verify_api_key),
) -> dict:
    """
    List all files recursively from a read-only artifact source.

    Returns a flat list of all files with their paths, suitable for unified display
    with uploaded artifacts.

    Args:
        source_id: The artifact source identifier (e.g., 'vllm_wheels')

    Returns:
        Dictionary with source info and flat list of all files
    """
    if source_id not in READONLY_ARTIFACT_SOURCES:
        raise InvalidPathError(source_id, f"Unknown artifact source: {source_id}")

    source_info = READONLY_ARTIFACT_SOURCES[source_id]
    base_path = source_info["path"]
    resolved_base = os.path.realpath(base_path)

    if not os.path.exists(resolved_base):
        raise InvalidPathError(source_id, "Source directory not found")

    files = []

    for root, _, filenames in os.walk(resolved_base):
        for filename in filenames:
            full_path = os.path.join(root, filename)
            relative_path = os.path.relpath(full_path, resolved_base)

            try:
                stat_info = os.stat(full_path)
                file_size = stat_info.st_size
                modified_at = stat_info.st_mtime
            except Exception:
                file_size = 0
                modified_at = 0

            ext = os.path.splitext(filename)[1].lower()
            file_type = "wheel" if ext == ".whl" else "unknown"

            files.append({
                "name": filename,
                "path": relative_path,
                "size_bytes": file_size,
                "modified_at": modified_at,
                "file_type": file_type,
            })

    # Sort by path for consistent ordering
    files.sort(key=lambda f: f["path"])

    return {
        "source_id": source_id,
        "source_name": source_info["name"],
        "files": files,
        "total": len(files),
        "readonly": True,
    }
