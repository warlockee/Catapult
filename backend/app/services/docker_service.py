"""
Docker build orchestration service.

This service coordinates Docker builds using decomposed sub-services:
- DockerfileService: Template resolution and generation
- BuildWorkspaceService: Build directory and file staging
- BuildExecutorService: Docker command execution
- BuildArchiveService: Build job archiving
"""
import logging
import os
import re
import uuid
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.config import settings
from app.core.database import async_session_maker
from app.core.exceptions import InvalidImageTagError
from app.models.artifact import Artifact
from app.models.docker_build import DockerBuild
from app.models.docker_build_artifact import DockerBuildArtifact
from app.models.model import Model
from app.models.release import Release
from app.services.docker.archive_service import BuildArchiveService

# Import decomposed services
from app.services.docker.dockerfile_service import DockerfileConfig, DockerfileService
from app.services.docker.executor_service import BuildCommand, BuildExecutorService
from app.services.docker.workspace_service import BuildWorkspaceService, normalize_ceph_path

logger = logging.getLogger(__name__)

# Docker image tag validation pattern (allows uppercase for model names like Qwen3-8B)
DOCKER_IMAGE_TAG_PATTERN = re.compile(
    r'^[a-zA-Z0-9]([a-zA-Z0-9._/-]*[a-zA-Z0-9])?'
    r'(:[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,127})?$'
)


def validate_image_tag(image_tag: str) -> str:
    """
    Validate Docker image tag to prevent shell injection.

    Args:
        image_tag: The Docker image tag to validate

    Returns:
        The validated image tag

    Raises:
        InvalidImageTagError: If the image tag contains invalid characters
    """
    if not image_tag:
        raise InvalidImageTagError(image_tag or "", "Image tag cannot be empty")

    if len(image_tag) > 256:
        raise InvalidImageTagError(image_tag, "Exceeds maximum length of 256 characters")

    if not DOCKER_IMAGE_TAG_PATTERN.match(image_tag):
        raise InvalidImageTagError(
            image_tag,
            "Must contain only letters, digits, underscores, periods, hyphens, and forward slashes"
        )

    dangerous_chars = ['$', '`', '|', ';', '&', '>', '<', '\\', '\n', '\r', '"', "'", '(', ')']
    for char in dangerous_chars:
        if char in image_tag:
            raise InvalidImageTagError(image_tag, f"Contains forbidden character: {repr(char)}")

    return image_tag


class DockerService:
    """
    Docker build orchestration service.

    Coordinates the build process using focused sub-services.
    """

    def __init__(self):
        """Initialize DockerService with sub-services."""
        # Initialize sub-services
        self.dockerfile_service = DockerfileService()
        self.workspace_service = BuildWorkspaceService()
        self.executor_service = BuildExecutorService()
        self.archive_service = BuildArchiveService()

        # Keep these for backward compatibility
        self.build_dir_base = settings.DOCKER_BUILD_DIR
        self.logs_dir_base = settings.DOCKER_LOGS_DIR
        self.jobs_archive_dir = settings.DOCKER_JOBS_ARCHIVE_DIR

    async def create_build(
        self,
        db: AsyncSession,
        release_id: uuid.UUID,
        image_tag: str,
        build_type: str,
        artifact_id: uuid.UUID = None,
        artifact_ids: list[uuid.UUID] = None,
        dockerfile_content: str = None,
    ) -> DockerBuild:
        """
        Create a Docker build record.

        Args:
            db: Database session
            release_id: Associated release ID
            image_tag: Docker image tag
            build_type: Type of build (test, optimized, azure, standard)
            artifact_id: Legacy single artifact ID
            artifact_ids: List of artifact IDs
            dockerfile_content: Custom Dockerfile content

        Returns:
            Created DockerBuild record
        """
        validated_tag = validate_image_tag(image_tag)

        build = DockerBuild(
            release_id=release_id,
            artifact_id=artifact_id,
            artifact_ids=artifact_ids,
            image_tag=validated_tag,
            build_type=build_type,
            dockerfile_content=dockerfile_content,
            status="pending"
        )
        db.add(build)
        await db.flush()

        # Populate junction table
        all_artifact_ids = set()
        if artifact_id:
            all_artifact_ids.add(artifact_id)
        if artifact_ids:
            all_artifact_ids.update(artifact_ids)

        for aid in all_artifact_ids:
            build_artifact = DockerBuildArtifact(
                docker_build_id=build.id,
                artifact_id=aid
            )
            db.add(build_artifact)

        await db.commit()
        await db.refresh(build)
        return build

    async def run_build(self, build_id: uuid.UUID):
        """
        Execute a Docker build.

        Args:
            build_id: Build ID to execute
        """
        async with async_session_maker() as db:
            build = await db.get(DockerBuild, build_id)
            if not build:
                logger.error(f"Build {build_id} not found")
                return

            # Update status
            build.status = "building"
            build.log_path = self.executor_service.get_log_path(str(build_id))
            await db.commit()

            # Create log file immediately so streaming can start
            # (model staging can take a while for large models)
            log_dir = os.path.dirname(build.log_path)
            os.makedirs(log_dir, exist_ok=True)
            with open(build.log_path, "w") as f:
                f.write(f"Build {build_id} started. Preparing build context...\n")

            build_dir = None
            primary_wheel = None

            try:
                # Fetch related data
                release = await db.get(Release, build.release_id)
                if not release:
                    raise Exception("Release not found")

                model = await db.get(Model, release.image_id)
                if not model:
                    raise Exception("Model not found")

                # Fetch artifacts
                all_artifact_ids = build.get_all_artifact_ids()
                artifacts_list = []
                if all_artifact_ids:
                    stmt = select(Artifact).where(Artifact.id.in_(all_artifact_ids))
                    result = await db.execute(stmt)
                    artifacts_list = list(result.scalars().all())

                # Helper to log progress
                def log_progress(msg: str):
                    with open(build.log_path, "a") as f:
                        f.write(f"{msg}\n")

                # --- Phase 1: Create Workspace ---
                build_dir = self.workspace_service.create_workspace(str(build_id))
                log_progress("✓ Created build workspace")

                # --- Phase 2: Generate Dockerfile ---
                # Determine model source path: prefer release.ceph_path, fallback to model.storage_path
                model_source_path = release.ceph_path or model.storage_path
                # Normalize hardcoded /ceph paths to configured CEPH_MOUNT_PATH
                model_source_path = normalize_ceph_path(model_source_path) if model_source_path else None
                model_name = os.path.basename(model_source_path) if model_source_path else model.name

                # Resolve full model path for auto-detection
                full_model_path = model_source_path
                if full_model_path and not os.path.isabs(full_model_path):
                    full_model_path = os.path.join(settings.CEPH_MOUNT_PATH, full_model_path)

                dockerfile_config = DockerfileConfig(
                    content=build.dockerfile_content,
                    build_type=build.build_type,
                    model_name=model_name,
                    server_type=model.server_type,  # Use model's server_type
                    model_path=full_model_path,  # For auto-detection if server_type not set
                )
                dockerfile_path, detected_type = self.dockerfile_service.generate_dockerfile(build_dir, dockerfile_config)

                # Save server_type to build record
                # If user explicitly selected asr-vllm build type, use that
                if build.build_type == "asr-vllm":
                    final_server_type = "asr-vllm"
                else:
                    final_server_type = detected_type or model.server_type
                if final_server_type:
                    build.server_type = final_server_type
                    await db.commit()

                if detected_type:
                    log_progress(f"✓ Generated Dockerfile (auto-detected: {detected_type})")
                elif model.server_type:
                    log_progress(f"✓ Generated Dockerfile (server_type: {model.server_type})")
                else:
                    log_progress("✓ Generated Dockerfile")

                # --- Phase 3: Stage Files ---
                if build.build_type == "optimized" and not build.dockerfile_content:
                    # Optimized builds need additional files
                    self.workspace_service.stage_optimized_build_files(build_dir)
                    log_progress("✓ Staged optimized build files")

                if build.build_type != "test":
                    # Stage artifacts
                    artifact_paths = [a.file_path for a in artifacts_list]
                    primary_wheel = self.workspace_service.stage_artifacts(build_dir, artifact_paths)
                    if artifact_paths:
                        log_progress(f"✓ Staged {len(artifact_paths)} artifact(s)")

                    # Stage model files - use model_source_path (release.ceph_path or model.storage_path)
                    if model_source_path:
                        log_progress(f"⏳ Staging model files from {model_name}... (this may take a while)")
                        self.workspace_service.stage_model(
                            build_dir,
                            model_source_path,
                            model_name,
                        )
                        log_progress("✓ Model files staged")

                        # Stage requirements and third_party
                        source_path = model_source_path
                        if not os.path.isabs(source_path):
                            source_path = os.path.join(settings.CEPH_MOUNT_PATH, source_path)
                        self.workspace_service.stage_requirements(build_dir, source_path)
                        self.workspace_service.stage_third_party(build_dir, source_path)
                        log_progress("✓ Staged requirements and dependencies")

                # Stage HF cache for ASR builds (needed for whisper preprocessor)
                if build.build_type == "asr-vllm":
                    self.workspace_service.stage_hf_cache(build_dir, ["openai/whisper-large-v3-turbo"])
                    log_progress("✓ Staged HuggingFace cache for offline ASR support")

                # Stage ASR Azure patches (Audio model files)
                if build.build_type == "asr-azure-allinone":
                    self.workspace_service.stage_asr_azure_patches(build_dir)
                    log_progress("✓ Staged ASR Azure patches (audio model files)")

                # --- Phase 4: Execute Build ---
                # Standard build flow for all build types (including asr-vllm)
                build_args = {}
                # Pass model name as build arg
                build_args["MODEL_NAME"] = model_name
                if primary_wheel:
                    build_args["VLLM_PRECOMPILED_WHEEL_LOCATION"] = primary_wheel

                build_command = BuildCommand(
                    image_tag=build.image_tag,
                    dockerfile_path=dockerfile_path,
                    context_path=build_dir,
                    build_args=build_args,
                )

                log_progress("\n🔨 Starting Docker build...\n")

                result = await self.executor_service.execute(str(build_id), build_command)

                if not result.success:
                    raise Exception(result.error_message or "Docker build failed")

                # --- Phase 5: Success Handling ---
                build.status = "success"
                build.completed_at = datetime.now()

                # Mark previous successful builds for this release as superseded
                await self._mark_previous_builds_superseded(db, build)

                # Update release metadata
                if not release.meta_data:
                    release.meta_data = {}
                new_metadata = dict(release.meta_data)
                new_metadata["docker_image"] = build.image_tag
                release.meta_data = new_metadata

                # Archive build job
                try:
                    self.archive_service.archive_build(
                        build_id=str(build_id),
                        release_id=str(build.release_id),
                        model_id=str(release.image_id) if release else None,
                        image_tag=build.image_tag,
                        build_type=build.build_type,
                        dockerfile_path=dockerfile_path,
                        started_at=build.created_at,
                        completed_at=build.completed_at,
                        build_args=build_args,
                    )
                except Exception as archive_err:
                    logger.error(f"Failed to archive build job: {archive_err}")

                await db.commit()

            except Exception as e:
                logger.error(f"Build failed: {e}")
                build.status = "failed"
                build.error_message = str(e)
                build.completed_at = datetime.now()
                await db.commit()

            finally:
                # Cleanup workspace on success only
                if build_dir and build.status == "success":
                    self.workspace_service.cleanup_workspace(build_dir)
                elif build_dir:
                    logger.info(f"Build failed. Keeping build directory for debugging: {build_dir}")

    async def _mark_previous_builds_superseded(
        self,
        db: AsyncSession,
        current_build: DockerBuild
    ):
        """
        Mark all previous successful builds for the same release as superseded.

        This starts the 7-day cleanup countdown for old builds.

        Args:
            db: Database session
            current_build: The newly successful build
        """
        now = datetime.now()

        # Find all other successful builds for this release that aren't superseded yet
        stmt = select(DockerBuild).where(
            DockerBuild.release_id == current_build.release_id,
            DockerBuild.id != current_build.id,
            DockerBuild.status == "success",
            DockerBuild.superseded_at.is_(None),
        )
        result = await db.execute(stmt)
        old_builds = result.scalars().all()

        if old_builds:
            for old_build in old_builds:
                old_build.superseded_at = now
                logger.info(
                    f"Marked build {old_build.id} ({old_build.image_tag}) as superseded. "
                    f"Cleanup scheduled in {settings.DOCKER_IMAGE_RETENTION_DAYS} days."
                )

    async def _run_asr_vllm_build(
        self,
        build_id: str,
        build_dir: str,
        image_tag: str,
        artifacts: list,
        log_progress,
        dockerfile_content: str = None,
    ):
        """
        Execute ASR vLLM build using custom vLLM fork repository.

        This clones the custom vLLM fork repo and uses its Dockerfile with the
        precompiled vLLM wheel. If dockerfile_content is provided (user edited),
        it will be used instead of the repo's Dockerfile.

        Args:
            build_id: Unique build identifier
            build_dir: Workspace directory
            image_tag: Target Docker image tag
            artifacts: List of artifacts (should include vLLM wheel)
            log_progress: Callback for progress logging
            dockerfile_content: Custom Dockerfile content (if user edited)

        Returns:
            BuildResult with success status
        """
        import shutil
        import subprocess

        from app.services.docker.executor_service import BuildResult

        log_progress("\n🔨 Starting ASR vLLM build (using custom vLLM fork)...\n")

        # Log file path for this build
        log_file = os.path.join(settings.DOCKER_LOGS_DIR, f"{build_id}.log")

        # Find vLLM wheel in artifacts
        vllm_wheel = None
        vllm_version = "0.10.2"  # Default version
        for artifact in artifacts:
            if artifact.file_path and "vllm" in artifact.file_path.lower() and artifact.file_path.endswith(".whl"):
                vllm_wheel = artifact.file_path
                # Extract version from wheel filename (e.g., vllm-0.10.2+asr-cp312...)
                import re
                match = re.search(r'vllm[_-](\d+\.\d+(?:\.\d+)?)', os.path.basename(vllm_wheel))
                if match:
                    vllm_version = match.group(1)
                break

        if not vllm_wheel:
            return BuildResult(
                success=False,
                return_code=1,
                log_path=log_file,
                error_message="No vLLM wheel artifact found. ASR vLLM builds require a vLLM wheel.",
            )

        log_progress(f"✓ Found vLLM wheel: {os.path.basename(vllm_wheel)} (version: {vllm_version})")

        # Clone custom vLLM fork repo
        repo_dir = os.path.join(build_dir, "vllm-fork")
        repo_branch = settings.ASR_VLLM_REPO_BRANCH
        repo_url = settings.ASR_VLLM_REPO_URL

        log_progress(f"⏳ Cloning custom vLLM fork (branch: {repo_branch})...")

        try:
            # Setup gh auth for git (in case it wasn't configured)
            subprocess.run(["gh", "auth", "setup-git"], capture_output=True, timeout=30)

            # Configure git to use HTTPS instead of SSH for GitHub (for submodules)
            subprocess.run(
                ["git", "config", "--global", "url.https://github.com/.insteadOf", "git@github.com:"],
                capture_output=True, timeout=10
            )

            clone_result = subprocess.run(
                ["git", "clone", "--recursive", "-b", repo_branch, repo_url, repo_dir],
                capture_output=True,
                text=True,
                timeout=600,  # 10 min timeout for clone
            )
            if clone_result.returncode != 0:
                return BuildResult(
                    success=False,
                    return_code=clone_result.returncode,
                    log_path=log_file,
                    error_message=f"Failed to clone custom vLLM fork: {clone_result.stderr}",
                )
            log_progress("✓ Cloned custom vLLM fork repository")
        except subprocess.TimeoutExpired:
            return BuildResult(
                success=False,
                return_code=1,
                log_path=log_file,
                error_message="Git clone timed out after 10 minutes",
            )
        except Exception as e:
            return BuildResult(
                success=False,
                return_code=1,
                log_path=log_file,
                error_message=f"Git clone failed: {e}",
            )

        # Copy wheel to repo
        wheel_filename = os.path.basename(vllm_wheel)
        wheel_dest = os.path.join(repo_dir, wheel_filename)

        # Normalize source path
        wheel_src = vllm_wheel
        if not os.path.isabs(wheel_src):
            wheel_src = os.path.join(settings.CEPH_MOUNT_PATH, wheel_src)

        try:
            shutil.copy2(wheel_src, wheel_dest)
            log_progress(f"✓ Copied wheel to build context: {wheel_filename}")
        except Exception as e:
            return BuildResult(
                success=False,
                return_code=1,
                log_path=log_file,
                error_message=f"Failed to copy wheel: {e}",
            )

        # Use custom Dockerfile content if provided (user edited)
        dockerfile_path = "docker/Dockerfile"
        if dockerfile_content:
            custom_dockerfile_path = os.path.join(repo_dir, "docker/Dockerfile")
            try:
                with open(custom_dockerfile_path, "w") as f:
                    f.write(dockerfile_content)
                log_progress("✓ Using custom (edited) Dockerfile")
            except Exception as e:
                return BuildResult(
                    success=False,
                    return_code=1,
                    log_path=log_file,
                    error_message=f"Failed to write custom Dockerfile: {e}",
                )
        else:
            log_progress("✓ Using default custom vLLM fork Dockerfile")

        # Build Docker image
        build_args = {
            "SETUPTOOLS_SCM_PRETEND_VERSION": vllm_version,
            "RUN_WHEEL_CHECK": "false",
            "VLLM_USE_PRECOMPILED": "1",
            "VLLM_PRECOMPILED_WHEEL_LOCATION": wheel_filename,
        }

        # Construct docker build command
        cmd = [
            "docker", "build",
            "--network=host",
            "--tag", image_tag,
            "--target", "vllm-base",
            "--file", dockerfile_path,
        ]

        for key, value in build_args.items():
            cmd.extend(["--build-arg", f"{key}={value}"])

        cmd.append(".")  # Context is the repo directory

        log_progress(f"⏳ Building Docker image: {image_tag}")
        log_progress(f"  Build args: {build_args}")

        # Execute build
        try:
            env = os.environ.copy()
            env["DOCKER_BUILDKIT"] = "1"

            # Stream build output to log file
            with open(log_file, "a") as log_f:
                process = subprocess.Popen(
                    cmd,
                    cwd=repo_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                )

                for line in process.stdout:
                    log_f.write(line)
                    log_f.flush()
                    # Log key milestones
                    if "Step " in line or "Successfully" in line or "ERROR" in line:
                        log_progress(line.strip())

                process.wait()

                if process.returncode != 0:
                    return BuildResult(
                        success=False,
                        return_code=process.returncode,
                        log_path=log_file,
                        error_message=f"Docker build failed with exit code {process.returncode}",
                    )

            log_progress(f"✓ Successfully built image: {image_tag}")
            return BuildResult(success=True, return_code=0, log_path=log_file)

        except Exception as e:
            return BuildResult(
                success=False,
                return_code=1,
                log_path=log_file,
                error_message=f"Docker build failed: {e}",
            )

    async def sync_archived_jobs(self):
        """
        Sync archived Docker build jobs from file system to database.

        Runs on startup to ensure persistence consistency.
        """
        logger.info("Starting sync of archived Docker build jobs...")

        archived_jobs = self.archive_service.list_archived_jobs()
        if not archived_jobs:
            logger.info("No archived jobs found. Skipping sync.")
            return

        count = 0
        async with async_session_maker() as db:
            for job_id_str in archived_jobs:
                try:
                    job_id = uuid.UUID(job_id_str)

                    # Check if already exists
                    existing = await db.get(DockerBuild, job_id)
                    if existing:
                        continue

                    # Load metadata
                    meta = self.archive_service.load_job_metadata(job_id_str)
                    if not meta:
                        continue

                    release_id_str = meta.get("release_id")
                    if not release_id_str:
                        logger.warning(f"Skipping archived job {job_id}: Missing release_id")
                        continue

                    release_id = uuid.UUID(release_id_str)

                    # Verify release exists
                    release = await db.get(Release, release_id)
                    if not release:
                        logger.warning(f"Skipping archived job {job_id}: Release not found")
                        continue

                    # Create build record
                    build = DockerBuild(
                        id=job_id,
                        release_id=release_id,
                        image_tag=meta.get("image_tag", "unknown"),
                        build_type=meta.get("build_type", "unknown"),
                        status="success",
                        created_at=datetime.fromisoformat(meta["started_at"]) if meta.get("started_at") else datetime.now(),
                        completed_at=datetime.fromisoformat(meta["completed_at"]) if meta.get("completed_at") else datetime.now(),
                        log_path=self.executor_service.get_log_path(job_id_str),
                    )

                    db.add(build)
                    count += 1
                    logger.info(f"Restored DockerBuild record for job {job_id}")

                except Exception as e:
                    logger.error(f"Failed to sync archived job {job_id_str}: {e}")

            if count > 0:
                await db.commit()
                logger.info(f"Successfully synced {count} archived Docker build jobs.")
            else:
                logger.info("No new archived jobs to sync.")


    async def cancel_build(self, db: AsyncSession, build_id: uuid.UUID) -> DockerBuild:
        """
        Cancel a running Docker build.

        Args:
            db: Database session
            build_id: Build ID to cancel

        Returns:
            Updated DockerBuild record

        Raises:
            DockerBuildNotFoundError: If build not found
            Exception: If cancellation fails
        """
        from app.core.celery_app import celery_app

        build = await db.get(DockerBuild, build_id)
        if not build:
            # We can't raise DockerBuildNotFoundError easily here without circular imports/repository
            # But the endpoint handles 404. Ideally we use repository there.
            # For now, let's return None and handle in endpoint, or raise generic.
            raise ValueError(f"Build {build_id} not found")

        if build.status not in ["pending", "building"]:
            # Already finished, just return it
            return build

        # Revoke Celery task if ID exists
        if build.celery_task_id:
            logger.info(f"Revoking Celery task {build.celery_task_id} for build {build_id}")
            celery_app.control.revoke(build.celery_task_id, terminate=True, signal="SIGKILL")

        # Update status
        build.status = "cancelled"
        build.error_message = "Cancelled by user"
        build.completed_at = datetime.now()
        
        # Log cancellation
        if build.log_path:
            try:
                with open(build.log_path, "a") as f:
                    f.write("\n❌ Build cancelled by user.\n")
            except Exception as e:
                logger.error(f"Failed to append to log file: {e}")

        await db.commit()
        await db.refresh(build)
        return build


# Singleton instance
docker_service = DockerService()
