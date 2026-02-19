"""ASR evaluation module."""
# Use docker-based evaluator (runs in container with heavy dependencies)
from app.services.eval.asr.docker_evaluator import ASRDockerEvaluator

__all__ = ["ASRDockerEvaluator"]
