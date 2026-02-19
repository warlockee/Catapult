"""
Docker-based ASR Evaluator - runs WER/CER evaluation in a docker container.

This evaluator uses the asr-eval Docker container which has all heavy dependencies
(torch, jiwer, pyarrow, etc.) pre-installed and includes an embedded evaluation dataset.
"""
import re
from typing import Any, Dict, List

from app.services.eval.base import EvaluationConfig
from app.services.eval.docker_base import DockerEvaluator
from app.services.eval.factory import register_evaluator


@register_evaluator('asr')
class ASRDockerEvaluator(DockerEvaluator):
    """
    Docker-based evaluator for ASR models using WER/CER metrics.

    Uses the asr-eval Docker container which:
    - Contains embedded Common Voice evaluation dataset
    - Supports multiple API modes (chat, transcriptions, allinone)
    - Outputs standard EVAL_PROGRESS format for progress tracking
    """

    docker_image = "catapult/asr-eval:latest"
    docker_build_path = "/app/kb/dockers/asr_eval"
    dockerfile_path = "/app/kb/dockers/asr_eval/Dockerfile"

    @property
    def evaluation_type(self) -> str:
        return "asr"

    @property
    def evaluator_name(self) -> str:
        return "ASRDockerEvaluator"

    def get_supported_metrics(self) -> List[str]:
        return ["wer", "cer", "samples_evaluated", "no_speech_count"]

    def build_docker_args(self, config: EvaluationConfig) -> List[str]:
        """Build command-line arguments for the ASR eval container."""
        # Ensure endpoint URL ends with /v1
        endpoint_url = config.endpoint_url.rstrip('/')
        if not endpoint_url.endswith('/v1'):
            endpoint_url = f"{endpoint_url}/v1"

        return [
            "--endpoint_url", endpoint_url,
            "--model", config.model_name,
            "--limit", str(config.limit) if config.limit > 0 else "500",
            "--api-mode", "auto",  # Auto-detect API type
            "--no-baseline",
        ]

    def parse_metrics(self, stdout: str) -> Dict[str, Any]:
        """Parse WER/CER metrics from container output."""
        wer = None
        cer = None
        samples = 0

        for line in stdout.split('\n'):
            # Match "Actual WER:   0.1234" or "WER:   0.1234"
            wer_match = re.search(r'(?:Actual\s+)?WER:\s+([\d.]+)', line)
            if wer_match:
                wer = float(wer_match.group(1))

            # Match "CER:          0.0567"
            cer_match = re.search(r'CER:\s+([\d.]+)', line)
            if cer_match:
                cer = float(cer_match.group(1))

            # Match sample count
            samples_match = re.search(r'Samples:\s+(\d+)', line)
            if samples_match:
                samples = int(samples_match.group(1))

            # Also try progress line format for sample count
            if not samples:
                progress_match = re.search(r'\((\d+)/\d+\s+@', line)
                if progress_match:
                    samples = int(progress_match.group(1))

        if wer is None:
            return {}

        return {
            'primary_metric': wer,
            'primary_metric_name': 'wer',
            'secondary_metric': cer,
            'secondary_metric_name': 'cer',
            'samples_evaluated': samples,
        }
