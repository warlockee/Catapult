"""
Service for Docker build workspace management.

Handles:
- Build directory creation and cleanup
- Artifact staging
- Model file copying
- Requirements and third-party dependencies
"""
import os
import shutil
import logging
from typing import List, Optional
from dataclasses import dataclass, field

from app.core.config import settings

logger = logging.getLogger(__name__)


def normalize_ceph_path(path: str) -> str:
    """
    Normalize CEPH paths for the current environment.

    External configs often use hardcoded /ceph/ paths, but the actual
    mount point may differ (e.g., /fsx in containers). This function translates
    /ceph/ paths to use the configured CEPH_MOUNT_PATH.

    Args:
        path: The path to normalize

    Returns:
        Normalized path using CEPH_MOUNT_PATH
    """
    if not path:
        return path

    # Common hardcoded prefixes that should map to CEPH_MOUNT_PATH
    ceph_prefixes = ['/ceph/', '/ceph']

    for prefix in ceph_prefixes:
        if path.startswith(prefix):
            # Replace the prefix with CEPH_MOUNT_PATH
            suffix = path[len(prefix):]
            if suffix.startswith('/'):
                suffix = suffix[1:]  # Remove leading slash
            normalized = os.path.join(settings.CEPH_MOUNT_PATH, suffix)
            logger.debug(f"Normalized path: {path} -> {normalized}")
            return normalized

    return path


@dataclass
class WorkspaceConfig:
    """Configuration for build workspace."""
    build_id: str
    build_type: str = "standard"
    model_source_path: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)  # List of artifact file paths


class BuildWorkspaceService:
    """
    Service for Docker build workspace management.

    Responsibilities:
    - Create and cleanup build directories
    - Stage artifacts (wheels, etc.)
    - Copy model files
    - Copy requirements and third-party dependencies
    """

    def __init__(
        self,
        build_dir_base: Optional[str] = None,
        vllm_artifacts_path: Optional[str] = None,
        vllm_fork_path: Optional[str] = None,
    ):
        """
        Initialize BuildWorkspaceService.

        Args:
            build_dir_base: Base directory for builds
            vllm_artifacts_path: Path to shared VLLM artifacts
            vllm_fork_path: Path to custom vLLM fork source
        """
        self.build_dir_base = build_dir_base or settings.DOCKER_BUILD_DIR
        self.vllm_artifacts_path = vllm_artifacts_path or settings.VLLM_ARTIFACTS_PATH
        self.vllm_fork_path = vllm_fork_path or settings.VLLM_FORK_PATH

        os.makedirs(self.build_dir_base, exist_ok=True)

    def create_workspace(self, build_id: str) -> str:
        """
        Create build workspace directory.

        Args:
            build_id: Build identifier

        Returns:
            Path to build directory
        """
        build_dir = os.path.join(self.build_dir_base, build_id)
        os.makedirs(build_dir, exist_ok=True)

        # Create dummy wheel to satisfy Dockerfile COPY logic
        dummy_wheel = os.path.join(build_dir, "dummy_wheel.whl")
        with open(dummy_wheel, "w") as f:
            pass

        logger.info(f"Created build workspace: {build_dir}")
        return build_dir

    def cleanup_workspace(self, build_dir: str, force: bool = False) -> None:
        """
        Cleanup build workspace.

        Args:
            build_dir: Path to build directory
            force: If True, cleanup even on failure
        """
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir, ignore_errors=True)
            logger.info(f"Cleaned up build workspace: {build_dir}")

    def stage_artifacts(
        self,
        build_dir: str,
        artifact_paths: List[str],
    ) -> Optional[str]:
        """
        Copy artifacts to build directory.

        Args:
            build_dir: Build directory path
            artifact_paths: List of artifact file paths

        Returns:
            Primary wheel filename if found, None otherwise
        """
        primary_wheel = None

        for artifact_path in artifact_paths:
            if not os.path.exists(artifact_path):
                logger.warning(f"Artifact not found: {artifact_path}")
                continue

            dest = os.path.join(build_dir, os.path.basename(artifact_path))
            shutil.copy(artifact_path, dest)
            logger.info(f"Staged artifact: {os.path.basename(artifact_path)}")

            # Track primary wheel for build args
            if not primary_wheel and artifact_path.endswith(".whl"):
                primary_wheel = os.path.basename(artifact_path)

        return primary_wheel

    def stage_model(
        self,
        build_dir: str,
        source_path: str,
        model_name: str,
    ) -> str:
        """
        Copy model files to build directory.

        Args:
            build_dir: Build directory path
            source_path: Source path to model files
            model_name: Model name (directory name)

        Returns:
            Path to staged model directory
        """
        model_dest_dir = os.path.join(build_dir, "models")
        os.makedirs(model_dest_dir, exist_ok=True)

        model_dest = os.path.join(model_dest_dir, model_name)

        # Normalize hardcoded /ceph paths to configured CEPH_MOUNT_PATH
        source_path = normalize_ceph_path(source_path)

        # Resolve relative paths
        if not os.path.isabs(source_path):
            source_path = os.path.join(settings.CEPH_MOUNT_PATH, source_path)

        if os.path.exists(model_dest):
            shutil.rmtree(model_dest)

        if not os.path.exists(source_path):
            raise FileNotFoundError(
                f"Model source not found: {source_path}. "
                f"Ensure the model files exist and are accessible before deployment. "
                f"Check CEPH mount and model sync status."
            )

        shutil.copytree(source_path, model_dest)
        logger.info(f"Staged model files from {source_path}")

        return model_dest

    def stage_requirements(
        self,
        build_dir: str,
        model_source_path: Optional[str] = None,
    ) -> None:
        """
        Copy requirements to build directory.

        Tries model source first, then falls back to shared location.

        Args:
            build_dir: Build directory path
            model_source_path: Optional model source path
        """
        reqs_dest = os.path.join(build_dir, "requirements")

        # Try model source first
        if model_source_path:
            reqs_src = os.path.join(model_source_path, "requirements")
            if os.path.exists(reqs_src):
                if os.path.exists(reqs_dest):
                    shutil.rmtree(reqs_dest)
                shutil.copytree(reqs_src, reqs_dest)
                logger.info(f"Staged requirements from model source")
                return

        # Fallback to shared location
        fallback_src = os.path.join(self.vllm_artifacts_path, "requirements")
        if os.path.exists(fallback_src):
            if os.path.exists(reqs_dest):
                shutil.rmtree(reqs_dest)
            shutil.copytree(fallback_src, reqs_dest)
            logger.info(f"Staged requirements from shared location")
        else:
            logger.warning("Requirements directory not found")

    def stage_third_party(
        self,
        build_dir: str,
        model_source_path: Optional[str] = None,
    ) -> None:
        """
        Copy third_party dependencies to build directory.

        Tries model source first, then falls back to shared location.

        Args:
            build_dir: Build directory path
            model_source_path: Optional model source path
        """
        tp_dest = os.path.join(build_dir, "third_party")

        # Try model source first
        if model_source_path:
            tp_src = os.path.join(model_source_path, "third_party")
            if os.path.exists(tp_src):
                if os.path.exists(tp_dest):
                    shutil.rmtree(tp_dest)
                shutil.copytree(tp_src, tp_dest)
                logger.info(f"Staged third_party from model source")
                return

        # Fallback to shared location
        fallback_src = os.path.join(self.vllm_artifacts_path, "third_party")
        if os.path.exists(fallback_src):
            if os.path.exists(tp_dest):
                shutil.rmtree(tp_dest)
            shutil.copytree(fallback_src, tp_dest)
            logger.info(f"Staged third_party from shared location")
        else:
            logger.warning("third_party directory not found")

    def stage_optimized_build_files(self, build_dir: str) -> None:
        """
        Stage additional files needed for optimized builds.

        Copies wheels, xcodec model, and xcodec source code.

        Args:
            build_dir: Build directory path
        """
        # Copy .build-wheels
        wheels_src = os.path.join(self.vllm_fork_path, ".build-wheels")
        wheels_dest = os.path.join(build_dir, ".build-wheels")
        if os.path.exists(wheels_src):
            if os.path.exists(wheels_dest):
                shutil.rmtree(wheels_dest)
            shutil.copytree(wheels_src, wheels_dest)
            logger.info("Staged .build-wheels")

        # Copy xcodec model
        xcodec_src = os.path.join(
            self.vllm_fork_path,
            "models/xcodec_tps25_0516_exp_1"
        )
        xcodec_dest = os.path.join(
            build_dir,
            "models/xcodec_tps25_0516_exp_1"
        )
        os.makedirs(os.path.dirname(xcodec_dest), exist_ok=True)
        if os.path.exists(xcodec_dest):
            shutil.rmtree(xcodec_dest)
        if os.path.exists(xcodec_src):
            shutil.copytree(xcodec_src, xcodec_dest)
            logger.info("Staged xcodec model")

        # Copy xcodec source code
        xcodec_code_src = os.path.join(self.vllm_fork_path, "third_party/xcodec")
        xcodec_code_dest = os.path.join(build_dir, "third_party/xcodec")
        os.makedirs(os.path.dirname(xcodec_code_dest), exist_ok=True)
        if os.path.exists(xcodec_code_src):
            if os.path.exists(xcodec_code_dest):
                shutil.rmtree(xcodec_code_dest)
            shutil.copytree(xcodec_code_src, xcodec_code_dest)
            logger.info("Staged xcodec source code")

        # Copy requirements
        reqs_src = os.path.join(self.vllm_fork_path, "requirements")
        reqs_dest = os.path.join(build_dir, "requirements")
        if os.path.exists(reqs_src):
            if os.path.exists(reqs_dest):
                shutil.rmtree(reqs_dest)
            shutil.copytree(reqs_src, reqs_dest)
            logger.info("Staged requirements for optimized build")

    def stage_asr_azure_patches(self, build_dir: str) -> None:
        """
        Stage audio model patches for ASR Azure allinone builds.

        The ASR Azure Dockerfile requires patches/ directory with
        model-specific Python files and a registry module.

        Args:
            build_dir: Build directory path
        """
        # Patches are located relative to the backend root (/app in container)
        # __file__ = /app/app/services/docker/workspace_service.py
        # We need to go up 4 levels to get to /app, then down to kb/dockers/asr/azure/patches
        backend_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        patches_src = os.path.join(backend_root, "kb/dockers/asr/azure/patches")
        patches_dest = os.path.join(build_dir, "patches")

        if not os.path.exists(patches_src):
            logger.warning(f"ASR Azure patches not found at {patches_src}")
            return

        if os.path.exists(patches_dest):
            shutil.rmtree(patches_dest)
        shutil.copytree(patches_src, patches_dest)
        logger.info(f"Staged ASR Azure patches from {patches_src}")

    def stage_hf_cache(self, build_dir: str, model_names: List[str] = None) -> None:
        """
        Stage HuggingFace model cache for offline builds.

        Copies specified models from HF cache to build directory.
        Required for ASR models that need whisper preprocessor at runtime.

        Args:
            build_dir: Build directory path
            model_names: List of HF model names (e.g., ["openai/whisper-large-v3-turbo"])
        """
        if model_names is None:
            # Default models needed for ASR
            model_names = ["openai/whisper-large-v3-turbo"]

        hf_cache_src = os.path.join(settings.CEPH_MOUNT_PATH, "hf_cache/hub")
        hf_cache_dest = os.path.join(build_dir, "hf_cache")
        os.makedirs(hf_cache_dest, exist_ok=True)

        for model_name in model_names:
            # HF cache uses models--org--name format
            cache_dir_name = f"models--{model_name.replace('/', '--')}"
            src = os.path.join(hf_cache_src, cache_dir_name)
            dest = os.path.join(hf_cache_dest, cache_dir_name)

            if os.path.exists(src):
                if os.path.exists(dest):
                    shutil.rmtree(dest)
                shutil.copytree(src, dest)
                logger.info(f"Staged HF cache: {model_name}")
            else:
                logger.warning(f"HF cache not found for {model_name}: {src}")


# Singleton instance
workspace_service = BuildWorkspaceService()
