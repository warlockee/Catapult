"""
ASR Evaluator - Word Error Rate (WER) and Character Error Rate (CER) evaluation.
"""
import asyncio
import base64
import io
import logging
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import httpx

from app.services.eval.base import (
    EvaluationConfig,
    EvaluationMetrics,
    EvaluationResult,
    Evaluator,
    ProgressCallback,
)
from app.services.eval.factory import register_evaluator

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

# Audio constants
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 4000
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
MAX_CHUNKS_PER_REQUEST = 4

# ASR prompts
SYSTEM_MESSAGE = "You are an automatic speech recognition (ASR) system."
USER_MESSAGE_WITH_LANGUAGE = "Your task is to listen to audio input and output the exact spoken words as plain text in {language}."
USER_MESSAGE_AUTO = "Your task is to listen to audio input and output the exact spoken words as plain text."


def _get_user_message(language: Optional[str] = None) -> str:
    """Get user message for ASR transcription."""
    if language is None:
        return USER_MESSAGE_AUTO
    return USER_MESSAGE_WITH_LANGUAGE.format(language=language)


@dataclass
class VADConfig:
    """Configuration for Silero VAD."""
    threshold: float = 0.55
    min_speech_duration_ms: int = 125
    min_silence_duration_ms: int = 200
    speech_pad_ms: int = 300


@dataclass
class ASRSampleResult:
    """Result for a single ASR sample."""
    unique_id: int
    reference: str
    hypothesis: str
    reference_normalized: str
    hypothesis_normalized: str
    num_segments: int
    audio_duration_seconds: float
    error: Optional[str] = None


@register_evaluator('asr')
class ASRWERCEREvaluator(Evaluator):
    """
    Evaluator for ASR models using WER/CER metrics.

    Implements VAD + segmentation + vLLM API call pipeline.
    """

    def __init__(self):
        self._vad_model = None
        self._get_speech_timestamps = None
        self._english_normalizer = None

    @property
    def evaluation_type(self) -> str:
        return "asr"

    @property
    def evaluator_name(self) -> str:
        return "ASRWERCEREvaluator"

    def get_supported_metrics(self) -> List[str]:
        return ["wer", "cer", "samples_evaluated", "no_speech_count"]

    def validate_config(self, config: EvaluationConfig) -> tuple[bool, Optional[str]]:
        """Validate ASR evaluation configuration."""
        if not config.endpoint_url:
            return False, "endpoint_url is required"
        if not config.model_name:
            return False, "model_name is required"
        if not config.dataset_path:
            return False, "dataset_path is required for ASR evaluation"
        if not Path(config.dataset_path).exists():
            return False, f"Dataset not found: {config.dataset_path}"
        return True, None

    def _load_vad(self):
        """Load Silero VAD model (lazy initialization)."""
        if self._vad_model is None:
            try:
                import torch
                model, utils = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    trust_repo=True,
                )
                self._vad_model = model
                self._get_speech_timestamps = utils[0]
                logger.info("Loaded Silero VAD model")
            except Exception as e:
                logger.error(f"Failed to load Silero VAD: {e}")
                raise
        return self._vad_model, self._get_speech_timestamps

    def _segment_audio(
        self,
        wav_np: "np.ndarray",
        vad_config: Optional[VADConfig] = None,
    ) -> List["np.ndarray"]:
        """Run VAD and split audio into <=4s segments."""
        import numpy as np
        import torch

        if vad_config is None:
            vad_config = VADConfig()

        vad_model, get_speech_ts = self._load_vad()
        wav_t = torch.from_numpy(wav_np).float()
        timestamps = get_speech_ts(
            wav_t,
            vad_model,
            sampling_rate=SAMPLE_RATE,
            threshold=vad_config.threshold,
            min_speech_duration_ms=vad_config.min_speech_duration_ms,
            min_silence_duration_ms=vad_config.min_silence_duration_ms,
            speech_pad_ms=vad_config.speech_pad_ms,
        )

        # Match reference: if no speech detected, use full audio
        if not timestamps:
            timestamps = [{"start": 0, "end": len(wav_np)}]

        segments = []
        for ts in timestamps:
            seg = wav_np[ts["start"]:ts["end"]].astype(np.float32)
            if len(seg) > CHUNK_SAMPLES:
                for i in range(0, len(seg), CHUNK_SAMPLES):
                    chunk = seg[i:i + CHUNK_SAMPLES]
                    if len(chunk) > 0:
                        segments.append(chunk)
            else:
                segments.append(seg)
        return segments

    def _segments_to_base64(self, segments: List["np.ndarray"]) -> List[str]:
        """Encode numpy segments as base64 WAV strings."""
        import soundfile as sf

        chunks = []
        for seg in segments:
            buf = io.BytesIO()
            sf.write(buf, seg, SAMPLE_RATE, format="WAV")
            buf.seek(0)
            chunks.append(base64.b64encode(buf.read()).decode("utf-8"))
        return chunks

    async def _call_api(
        self,
        client: httpx.AsyncClient,
        b64_chunks: List[str],
        endpoint_url: str,
        model_name: str,
        timeout: float = 120.0,
        language: Optional[str] = "English",
    ) -> str:
        """Send a batch of audio chunks to vLLM chat completions endpoint."""
        user_message = _get_user_message(language)
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_message}]
        for i, chunk in enumerate(b64_chunks):
            content.append({
                "type": "audio_url",
                "audio_url": {"url": f"data:audio/wav_{i};base64,{chunk}"},
            })

        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": content},
        ]

        url = f"{endpoint_url.rstrip('/')}/chat/completions"
        payload = {
            "model": model_name,
            "messages": messages,
            "max_completion_tokens": 256,
            "temperature": 0.0,
            "stop": ["<|eot_id|>", "<|endoftext|>", "<|audio_eos|>", "<|im_end|>"],
            "extra_body": {"skip_special_tokens": False, "repetition_penalty": 1.1},
        }

        response = await client.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def _transcribe(
        self,
        client: httpx.AsyncClient,
        b64_chunks: List[str],
        endpoint_url: str,
        model_name: str,
        timeout: float = 120.0,
        language: Optional[str] = "English",
    ) -> str:
        """Transcribe audio chunks, batching into groups to avoid repetition."""
        if len(b64_chunks) <= MAX_CHUNKS_PER_REQUEST:
            return await self._call_api(
                client, b64_chunks, endpoint_url, model_name, timeout, language
            )

        parts = []
        for start in range(0, len(b64_chunks), MAX_CHUNKS_PER_REQUEST):
            batch = b64_chunks[start:start + MAX_CHUNKS_PER_REQUEST]
            text = await self._call_api(
                client, batch, endpoint_url, model_name, timeout, language
            )
            if text:
                parts.append(text.strip())
        return " ".join(parts)

    def _get_normalizer(self):
        """Get or create the English text normalizer (lazy initialization)."""
        if self._english_normalizer is None:
            try:
                from whisper_normalizer.english import EnglishTextNormalizer
                self._english_normalizer = EnglishTextNormalizer()
                logger.info("Using whisper_normalizer for WER calculation")
            except ImportError:
                logger.warning("whisper_normalizer not available, using basic normalization")
                self._english_normalizer = False  # Mark as unavailable
        return self._english_normalizer if self._english_normalizer else None

    def _normalize_text(self, text: str) -> str:
        """Normalize text for WER comparison."""
        normalizer = self._get_normalizer()
        if normalizer is not None:
            return normalizer(text)
        # Fallback to basic normalization
        text = text.lower().strip()
        text = re.sub(r"[^\w\s']", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    async def evaluate(
        self,
        config: EvaluationConfig,
        progress_callback: Optional[ProgressCallback] = None,
        evaluation_id: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Run WER/CER evaluation on an Arrow dataset.

        Args:
            config: Evaluation configuration
            progress_callback: Optional async callback(current, total) for progress

        Returns:
            EvaluationResult with WER/CER metrics
        """
        is_valid, error = self.validate_config(config)
        if not is_valid:
            return EvaluationResult(
                success=False,
                error_message=error,
            )

        started_at = datetime.utcnow()
        t_start = time.time()

        # Import heavy dependencies lazily
        try:
            import jiwer
            import numpy as np
            import pyarrow as pa
        except ImportError as e:
            return EvaluationResult(
                success=False,
                error_message=f"Missing required packages: {e}",
            )

        # Load dataset
        try:
            logger.info(f"Loading dataset: {config.dataset_path}")
            table = pa.ipc.open_file(config.dataset_path).read_all()
            total = table.num_rows
            end_idx = total if config.limit == 0 else min(config.offset + config.limit, total)
            logger.info(f"Dataset has {total} samples, evaluating [{config.offset}:{end_idx}]")
        except Exception as e:
            return EvaluationResult(
                success=False,
                error_message=f"Failed to load dataset: {e}",
            )

        # Initialize VAD
        try:
            self._load_vad()
        except Exception as e:
            return EvaluationResult(
                success=False,
                error_message=f"Failed to load VAD model: {e}",
            )

        # Get VAD config from extra_config
        vad_config_dict = config.extra_config.get('vad_config')
        vad_config = VADConfig(**vad_config_dict) if vad_config_dict else None
        language = config.language or "English"

        refs: List[str] = []
        hyps: List[str] = []
        sample_results: List[ASRSampleResult] = []
        errors = 0
        no_speech = 0

        async with httpx.AsyncClient() as client:
            for idx in range(config.offset, end_idx):
                try:
                    audio = np.array(
                        table.column("audio_array")[idx].as_py(),
                        dtype=np.float32
                    )
                    ref = table.column("audio_transcript")[idx].as_py()
                    audio_duration = len(audio) / SAMPLE_RATE

                    # Segment audio
                    segments = self._segment_audio(audio, vad_config)
                    if not segments:
                        hyp = ""
                        no_speech += 1
                    else:
                        b64 = self._segments_to_base64(segments)
                        hyp = await self._transcribe(
                            client, b64, config.endpoint_url, config.model_name,
                            config.timeout, language
                        )

                    # Normalize for comparison
                    ref_n = self._normalize_text(ref)
                    hyp_n = self._normalize_text(hyp)

                    sample_results.append(ASRSampleResult(
                        unique_id=idx,
                        reference=ref,
                        hypothesis=hyp,
                        reference_normalized=ref_n,
                        hypothesis_normalized=hyp_n,
                        num_segments=len(segments),
                        audio_duration_seconds=audio_duration,
                    ))

                    if ref_n:
                        refs.append(ref_n)
                        hyps.append(hyp_n)

                    # Progress callback
                    current = idx - config.offset + 1
                    if progress_callback:
                        result = progress_callback(current, end_idx - config.offset)
                        if asyncio.iscoroutine(result):
                            await result

                    if current % 10 == 0 or current == 1:
                        elapsed = time.time() - t_start
                        rate = current / elapsed if elapsed > 0 else 0
                        running_wer = jiwer.wer(refs, hyps) if refs else 0
                        logger.info(
                            f"[{idx:>5}] WER={running_wer:.3f} "
                            f"({current}/{end_idx - config.offset} @ {rate:.1f}/s)"
                        )

                except Exception as e:
                    errors += 1
                    logger.warning(f"[{idx:>5}] Error: {e}")
                    sample_results.append(ASRSampleResult(
                        unique_id=idx,
                        reference="",
                        hypothesis="",
                        reference_normalized="",
                        hypothesis_normalized="",
                        num_segments=0,
                        audio_duration_seconds=0.0,
                        error=str(e),
                    ))

        elapsed = time.time() - t_start
        completed_at = datetime.utcnow()

        if not refs:
            return EvaluationResult(
                success=True,
                metrics=EvaluationMetrics(
                    primary_metric=0.0,
                    primary_metric_name="wer",
                    secondary_metric=0.0,
                    secondary_metric_name="cer",
                    samples_evaluated=0,
                    samples_with_errors=errors,
                    extra_metrics={
                        "no_speech_count": no_speech,
                        "dataset_path": config.dataset_path,
                    },
                ),
                error_message="No scored utterances (all samples had empty reference or no speech)",
                duration_seconds=elapsed,
                started_at=started_at,
                completed_at=completed_at,
                sample_results=[asdict(s) for s in sample_results],
            )

        # Calculate final WER/CER
        wer = jiwer.wer(refs, hyps)
        cer = jiwer.cer(refs, hyps)

        logger.info(
            f"WER evaluation complete: WER={wer:.4f} ({wer*100:.2f}%), "
            f"CER={cer:.4f} ({cer*100:.2f}%), "
            f"samples={len(refs)}, elapsed={elapsed:.1f}s"
        )

        return EvaluationResult(
            success=True,
            metrics=EvaluationMetrics(
                primary_metric=wer,
                primary_metric_name="wer",
                secondary_metric=cer,
                secondary_metric_name="cer",
                samples_evaluated=len(refs),
                samples_with_errors=errors,
                extra_metrics={
                    "no_speech_count": no_speech,
                    "dataset_path": config.dataset_path,
                },
            ),
            duration_seconds=elapsed,
            started_at=started_at,
            completed_at=completed_at,
            sample_results=[asdict(s) for s in sample_results],
        )
