"""
Exception handlers that map domain exceptions to HTTP responses.

Register these handlers in main.py to automatically convert domain exceptions
to appropriate HTTP status codes and response formats.
"""
import logging
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette import status

from app.core.exceptions import (
    DomainException,
    NotFoundError,
    AlreadyExistsError,
    ValidationError,
    AuthorizationError,
    OperationError,
    ServiceUnavailableError,
)

logger = logging.getLogger(__name__)


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Log validation errors for debugging."""
    logger.error(f"Validation error on {request.method} {request.url.path}: {exc.errors()}")
    logger.error(f"Request body: {exc.body}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()},
    )


async def domain_exception_handler(request: Request, exc: DomainException) -> JSONResponse:
    """
    Handle all domain exceptions and map to appropriate HTTP status codes.

    This provides a clean separation between domain logic and HTTP concerns.
    Services can raise domain exceptions without knowing about HTTP.
    """
    # Determine status code based on exception type
    if isinstance(exc, NotFoundError):
        status_code = status.HTTP_404_NOT_FOUND
    elif isinstance(exc, AlreadyExistsError):
        status_code = status.HTTP_409_CONFLICT
    elif isinstance(exc, ValidationError):
        status_code = status.HTTP_400_BAD_REQUEST
    elif isinstance(exc, AuthorizationError):
        status_code = status.HTTP_403_FORBIDDEN
    elif isinstance(exc, ServiceUnavailableError):
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif isinstance(exc, OperationError):
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        # Log operation errors as they indicate system issues
        logger.error(f"Operation error: {exc.message}", extra={"details": exc.details})
    else:
        # Default to 500 for unknown domain exceptions
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        logger.error(f"Unhandled domain exception: {exc.message}", extra={"details": exc.details})

    return JSONResponse(
        status_code=status_code,
        content={
            "detail": exc.message,
            "error_type": exc.__class__.__name__,
            **exc.details,
        },
    )


def register_exception_handlers(app):
    """
    Register all exception handlers with the FastAPI app.

    Call this function in main.py after creating the app instance.
    """
    # Register handler for base DomainException (catches all subclasses)
    app.add_exception_handler(DomainException, domain_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
