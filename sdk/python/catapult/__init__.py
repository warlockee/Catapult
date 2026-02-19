"""Catapult SDK - Python client for Catapult."""

from .client import Registry, RegistryError
from .models import Model, Version, Release, Deployment, ApiKey

__version__ = "2.0.0"
__all__ = [
    "Registry",
    "RegistryError",
    "Model",
    "Version",
    "Release",  # Backward compatibility alias for Version
    "Deployment",
    "ApiKey",
]
