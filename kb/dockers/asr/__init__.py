"""
ASR Docker tools and utilities.

This package provides:
- audio_utils: VAD, segmentation, and encoding utilities
- asr_client: CLI for transcribing audio files
- eval_asr: WER/CER evaluation against Arrow datasets
- test_asr_docker: Smoke test for vLLM ASR endpoint
"""
from .audio_utils import (
    CHUNK_DURATION_MS,
    CHUNK_SAMPLES,
    SAMPLE_RATE,
    SYSTEM_MESSAGE,
    USER_MESSAGE_AUTO,
    USER_MESSAGE_WITH_LANGUAGE,
    VADConfig,
    get_user_message,
    load_audio_from_bytes,
    load_audio_from_path,
    load_vad,
    segment_audio,
    segment_audio_np,
    segments_to_base64,
)

__all__ = [
    "CHUNK_DURATION_MS",
    "CHUNK_SAMPLES",
    "SAMPLE_RATE",
    "SYSTEM_MESSAGE",
    "USER_MESSAGE_AUTO",
    "USER_MESSAGE_WITH_LANGUAGE",
    "VADConfig",
    "get_user_message",
    "load_audio_from_bytes",
    "load_audio_from_path",
    "load_vad",
    "segment_audio",
    "segment_audio_np",
    "segments_to_base64",
]
