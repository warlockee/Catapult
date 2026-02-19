"""
Backward compatibility shim for ReleaseRepository.

DEPRECATED: Use app.repositories.version_repository.VersionRepository instead.
"""
from app.repositories.version_repository import VersionRepository

# Backward compatibility alias
ReleaseRepository = VersionRepository

__all__ = ["ReleaseRepository", "VersionRepository"]
