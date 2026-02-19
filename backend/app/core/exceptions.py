"""
Custom exception hierarchy for domain-specific errors.

This module provides a clean separation between domain errors and HTTP concerns.
Services raise domain exceptions, and the exception handlers in main.py map them to HTTP responses.
"""
from typing import Optional, Dict, Any


class DomainException(Exception):
    """Base exception for all domain errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


# =============================================================================
# Not Found Errors (404)
# =============================================================================

class NotFoundError(DomainException):
    """Base class for resource not found errors."""
    pass


class ModelNotFoundError(NotFoundError):
    """Model does not exist."""

    def __init__(self, identifier: str):
        super().__init__(f"Model not found: {identifier}", {"identifier": identifier})


class VersionNotFoundError(NotFoundError):
    """Version does not exist."""

    def __init__(self, identifier: str):
        super().__init__(f"Version not found: {identifier}", {"identifier": identifier})


# Backward compatibility alias (deprecated)
ReleaseNotFoundError = VersionNotFoundError


class ArtifactNotFoundError(NotFoundError):
    """Artifact does not exist."""

    def __init__(self, identifier: str):
        super().__init__(f"Artifact not found: {identifier}", {"identifier": identifier})


class DeploymentNotFoundError(NotFoundError):
    """Deployment does not exist."""

    def __init__(self, identifier: str):
        super().__init__(f"Deployment not found: {identifier}", {"identifier": identifier})


class DockerBuildNotFoundError(NotFoundError):
    """Docker build does not exist."""

    def __init__(self, identifier: str):
        super().__init__(f"Docker build not found: {identifier}", {"identifier": identifier})


class ApiKeyNotFoundError(NotFoundError):
    """API key does not exist."""

    def __init__(self, identifier: str):
        super().__init__(f"API key not found: {identifier}", {"identifier": identifier})


class TemplateNotFoundError(NotFoundError):
    """Docker template does not exist."""

    def __init__(self, template_path: str):
        super().__init__(f"Template not found: {template_path}", {"template_path": template_path})


# =============================================================================
# Conflict Errors (409)
# =============================================================================

class AlreadyExistsError(DomainException):
    """Base class for resource already exists errors."""
    pass


class ModelAlreadyExistsError(AlreadyExistsError):
    """Model with this name already exists."""

    def __init__(self, name: str):
        super().__init__(f"Model already exists: {name}", {"name": name})


class VersionAlreadyExistsError(AlreadyExistsError):
    """Version with this version string already exists."""

    def __init__(self, version: str, model_name: Optional[str] = None):
        details = {"version": version}
        if model_name:
            details["model_name"] = model_name
        super().__init__(f"Version already exists: {version}", details)


# Backward compatibility alias (deprecated)
ReleaseAlreadyExistsError = VersionAlreadyExistsError


class ArtifactAlreadyExistsError(AlreadyExistsError):
    """Artifact with this path already exists."""

    def __init__(self, file_path: str):
        super().__init__(f"Artifact already exists: {file_path}", {"file_path": file_path})


class ApiKeyAlreadyExistsError(AlreadyExistsError):
    """API key with this name already exists."""

    def __init__(self, name: str):
        super().__init__(f"API key already exists: {name}", {"name": name})


# =============================================================================
# Validation Errors (400/422)
# =============================================================================

class ValidationError(DomainException):
    """Base class for validation errors."""
    pass


class InvalidPathError(ValidationError):
    """File path is invalid or not allowed."""

    def __init__(self, path: str, reason: str = "Invalid path"):
        super().__init__(f"{reason}: {path}", {"path": path, "reason": reason})


class InvalidImageTagError(ValidationError):
    """Docker image tag is invalid."""

    def __init__(self, tag: str, reason: str):
        super().__init__(f"Invalid image tag '{tag}': {reason}", {"tag": tag, "reason": reason})


class InvalidVersionError(ValidationError):
    """Version string is invalid."""

    def __init__(self, version: str, reason: str = "Invalid version format"):
        super().__init__(f"{reason}: {version}", {"version": version, "reason": reason})


class InvalidConfigurationError(ValidationError):
    """Configuration is invalid."""

    def __init__(self, field: str, reason: str):
        super().__init__(f"Invalid configuration for {field}: {reason}", {"field": field, "reason": reason})


# =============================================================================
# Authorization Errors (403)
# =============================================================================

class AuthorizationError(DomainException):
    """Base class for authorization errors."""
    pass


class InsufficientPermissionsError(AuthorizationError):
    """User lacks required permissions."""

    def __init__(self, required_role: str, action: str):
        super().__init__(
            f"Insufficient permissions: {required_role} role required for {action}",
            {"required_role": required_role, "action": action}
        )


class ApiKeyInactiveError(AuthorizationError):
    """API key is inactive."""

    def __init__(self, key_name: str):
        super().__init__(f"API key is inactive: {key_name}", {"key_name": key_name})


# =============================================================================
# Operation Errors (500)
# =============================================================================

class OperationError(DomainException):
    """Base class for operation failures."""
    pass


class StorageError(OperationError):
    """Storage operation failed."""

    def __init__(self, operation: str, reason: str):
        super().__init__(f"Storage error during {operation}: {reason}", {"operation": operation, "reason": reason})


class BuildError(OperationError):
    """Docker build operation failed."""

    def __init__(self, build_id: str, reason: str):
        super().__init__(f"Build failed ({build_id}): {reason}", {"build_id": build_id, "reason": reason})


class DatabaseError(OperationError):
    """Database operation failed."""

    def __init__(self, operation: str, reason: str):
        super().__init__(f"Database error during {operation}: {reason}", {"operation": operation, "reason": reason})


# =============================================================================
# Service Unavailable (503)
# =============================================================================

class ServiceUnavailableError(DomainException):
    """External service is unavailable."""

    def __init__(self, service: str, reason: str = "Service unavailable"):
        super().__init__(f"{service}: {reason}", {"service": service, "reason": reason})


# =============================================================================
# Deployment Execution Errors
# =============================================================================

class DeploymentExecutionError(OperationError):
    """Deployment execution failed."""

    def __init__(self, deployment_id: str, reason: str):
        super().__init__(
            f"Deployment execution failed ({deployment_id}): {reason}",
            {"deployment_id": deployment_id, "reason": reason}
        )


class PortAllocationError(OperationError):
    """No available ports in range."""

    def __init__(self, port_range_start: int, port_range_end: int):
        super().__init__(
            f"No available ports in range {port_range_start}-{port_range_end}",
            {"port_range_start": port_range_start, "port_range_end": port_range_end}
        )


class ContainerNotFoundError(NotFoundError):
    """Container does not exist."""

    def __init__(self, container_id: str):
        super().__init__(f"Container not found: {container_id}", {"container_id": container_id})


class DeploymentNotRunningError(ValidationError):
    """Deployment is not in running state."""

    def __init__(self, deployment_id: str, current_status: str):
        super().__init__(
            f"Deployment {deployment_id} is not running (status: {current_status})",
            {"deployment_id": deployment_id, "current_status": current_status}
        )


class DeploymentAlreadyRunningError(ValidationError):
    """Deployment is already running."""

    def __init__(self, deployment_id: str):
        super().__init__(
            f"Deployment {deployment_id} is already running",
            {"deployment_id": deployment_id}
        )


class DockerImageNotFoundError(NotFoundError):
    """Docker image does not exist."""

    def __init__(self, image_tag: str):
        super().__init__(f"Docker image not found: {image_tag}", {"image_tag": image_tag})


class BenchmarkNotFoundError(NotFoundError):
    """Benchmark does not exist."""

    def __init__(self, identifier: str):
        super().__init__(f"Benchmark not found: {identifier}", {"identifier": identifier})


class EvaluationNotFoundError(NotFoundError):
    """Evaluation does not exist."""

    def __init__(self, identifier: str):
        super().__init__(f"Evaluation not found: {identifier}", {"identifier": identifier})


# =============================================================================
# Benchmark/Evaluation State Errors (400)
# =============================================================================

class BenchmarkNotCancellableError(ValidationError):
    """Benchmark cannot be cancelled in its current state."""

    def __init__(self, benchmark_id: str, current_status: str):
        super().__init__(
            f"Cannot cancel benchmark {benchmark_id} with status: {current_status}",
            {"benchmark_id": benchmark_id, "current_status": current_status}
        )


class EvaluationNotCancellableError(ValidationError):
    """Evaluation cannot be cancelled in its current state."""

    def __init__(self, evaluation_id: str, current_status: str):
        super().__init__(
            f"Cannot cancel evaluation {evaluation_id} with status: {current_status}",
            {"evaluation_id": evaluation_id, "current_status": current_status}
        )


class BenchmarkNotUpdatableError(ValidationError):
    """Benchmark cannot be updated in its current state."""

    def __init__(self, benchmark_id: str, current_status: str):
        super().__init__(
            f"Cannot update benchmark {benchmark_id} with status: {current_status}",
            {"benchmark_id": benchmark_id, "current_status": current_status}
        )


class InvalidBenchmarkConfigError(ValidationError):
    """Benchmark configuration is invalid."""

    def __init__(self, reason: str):
        super().__init__(f"Invalid benchmark configuration: {reason}", {"reason": reason})
