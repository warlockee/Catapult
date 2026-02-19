# Models package
from app.models.model import Model
from app.models.deployment import Deployment
from app.models.version import Version
from app.models.artifact import Artifact
from app.models.api_key import ApiKey
from app.models.audit_log import AuditLog
from app.models.docker_build import DockerBuild
from app.models.docker_build_artifact import DockerBuildArtifact
from app.models.benchmark import Benchmark
from app.models.evaluation import Evaluation

# Backward compatibility alias (deprecated)
Release = Version
