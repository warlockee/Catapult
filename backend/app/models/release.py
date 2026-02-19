"""
Backward compatibility shim for Release model.

DEPRECATED: Use app.models.version.Version instead.
"""
from app.models.version import Version

# Backward compatibility alias
Release = Version

__all__ = ["Release", "Version"]
