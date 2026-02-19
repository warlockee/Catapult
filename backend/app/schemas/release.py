"""
Backward compatibility shim for Release schemas.

DEPRECATED: Use app.schemas.version instead.
"""
from app.schemas.version import (
    VersionBase,
    VersionCreate,
    VersionUpdate,
    VersionResponse,
    VersionWithModel,
    VersionOption,
)

# Backward compatibility aliases
ReleaseBase = VersionBase
ReleaseCreate = VersionCreate
ReleaseUpdate = VersionUpdate
ReleaseResponse = VersionResponse
ReleaseWithModel = VersionWithModel
ReleaseWithImage = VersionWithModel
ReleaseOption = VersionOption

__all__ = [
    "ReleaseBase",
    "ReleaseCreate",
    "ReleaseUpdate",
    "ReleaseResponse",
    "ReleaseWithModel",
    "ReleaseWithImage",
    "ReleaseOption",
    # Also export Version names for transition
    "VersionBase",
    "VersionCreate",
    "VersionUpdate",
    "VersionResponse",
    "VersionWithModel",
    "VersionOption",
]
