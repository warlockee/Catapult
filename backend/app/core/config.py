"""
Application configuration using Pydantic Settings.
"""
from typing import List, Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    # Application
    APP_NAME: str = "Model Registry"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "info"
    DEBUG: bool = False
    POSTGRES_DB: str = "registry"
    
    # Database Schema
    DB_SCHEMA: str = "model_registry"

    # Database
    DATABASE_URL: str

    # Security
    API_KEY_SALT: str
    DEFAULT_ADMIN_KEY: str = "admin"
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173,http://localhost"

    # Storage
    CEPH_MOUNT_PATH: str = "/data/ceph"
    ARTIFACT_MOUNT_PATH: str = "/data/artifacts"
    LOCAL_STORAGE_PATH: str = "/data/local"
    MODEL_STORAGE_DIR: str = "images"
    # These internal paths are now mounted from STORAGE_ROOT in docker-compose
    # But we define defaults here that match the container mount points
    SNAPSHOT_DIR: str = "./storage/snapshots"
    SYNC_MODELS_PATH: str = "./storage/models"
    SYNC_ARTIFACTS_PATH: str = "./storage/artifacts"
    SYNC_DEPLOYMENTS_PATH: str = "./storage/deployments"
    # Note: This path is specific to the container layout
    DOCKER_TEMPLATES_PATH: str = "/app/kb/dockers" 

    # Docker Build
    DOCKER_BUILD_DIR: str = "/tmp/docker_builds"
    # Use a standard path inside the container, mount mapped from host
    DOCKER_LOGS_DIR: str = "./storage/docker_logs"
    DOCKER_JOBS_ARCHIVE_DIR: str = "./storage/dockerbuild_jobs"
    VLLM_ARTIFACTS_PATH: str = "./storage/artifacts"

    # Read-only artifact sources
    VLLM_WHEELS_PATH: str = "./storage/vllm_wheels_prebuilt"

    # Docker Image GC
    DOCKER_IMAGE_RETENTION_DAYS: int = 7  # Days to keep superseded images before cleanup
    DOCKER_FAILED_BUILD_RETENTION_DAYS: int = 7  # Days to keep failed build images

    # Deployment Execution
    DEPLOYMENT_PORT_RANGE_START: int = 9000
    DEPLOYMENT_PORT_RANGE_END: int = 9999
    DEPLOYMENT_HEALTH_CHECK_INTERVAL: int = 30  # seconds between health checks
    DEPLOYMENT_HEALTH_CHECK_TIMEOUT: int = 5    # seconds to wait for health check response
    DEPLOYMENT_HEALTH_CHECK_ENDPOINT: str = "/health"
    DEPLOYMENT_CONTAINER_PREFIX: str = "deployment"
    DEPLOYMENT_DEFAULT_MEMORY_LIMIT: str = "8g"
    DEPLOYMENT_LOGS_DIR: str = "./storage/deployment_logs"
    DEPLOYMENT_CONTAINER_PORT: int = 8000  # Default port inside container

    # Limits
    MAX_UPLOAD_SIZE: int = 1073741824  # 1GB in bytes

    AVAILABLE_BACKENDS_PATH: str = "./available_backend.txt"

    # ASR (Automatic Speech Recognition)
    ASR_DEFAULT_BACKEND_URL: str = "http://localhost:26007/v1"
    ASR_DEFAULT_MODEL_NAME: str = "asr-model"
    ASR_TIMEOUT: float = 120.0

    # Benchmarker (Docker-based benchmark execution)
    BENCHMARKER_IMAGE: str = "benchmarker:latest"
    BENCHMARKER_TIMEOUT: int = 600  # 10 minutes max per benchmark
    BENCHMARK_LOGS_DIR: str = "./storage/benchmark_logs"
    BENCHMARK_CALLBACK_URL: Optional[str] = None  # Auto-constructed if not set

    # ASR vLLM Build
    ASR_VLLM_REPO_URL: str = ""
    ASR_VLLM_REPO_BRANCH: str = "main"
    VLLM_FORK_PATH: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"
    )

    def get_cors_origins(self) -> List[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]


settings = Settings()
