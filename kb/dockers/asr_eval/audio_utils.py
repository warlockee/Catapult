"""
Shared audio processing utilities for ASR Docker.

Provides VAD, segmentation, and encoding functions used by:
- asr_client.py (CLI tool)
- eval_asr.py (evaluation script)

This is a self-contained version for the ASR Docker release.
"""
import base64
import io
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 4000
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)

SYSTEM_MESSAGE = (
    "You are an automatic speech recognition (ASR) system."
)

# User message templates matching multimodal evaluation
USER_MESSAGE_WITH_LANGUAGE = (
    "Your task is to listen to audio input and output the exact spoken words as plain text in {language}."
)
USER_MESSAGE_AUTO = (
    "Your task is to listen to audio input and output the exact spoken words as plain text."
)


def get_user_message(language: str | None = None) -> str:
    """
    Get user message for ASR transcription.

    Args:
        language: Target language (e.g., "English", "Chinese"). If None, uses auto-detect mode.

    Returns:
        User message string for ASR prompt
    """
    if language is None:
        return USER_MESSAGE_AUTO
    return USER_MESSAGE_WITH_LANGUAGE.format(language=language)


@dataclass
class VADConfig:
    """
    Configuration for Silero VAD.

    Default values match the multimodal evaluation parameters.
    """
    threshold: float = 0.55  # Speech probability threshold
    min_speech_duration_ms: int = 125  # Minimum speech chunk duration
    min_silence_duration_ms: int = 200  # Minimum silence between speech chunks
    speech_pad_ms: int = 300  # Padding added to speech chunks


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------
def load_audio_from_path(path: str) -> torch.Tensor:
    """Load audio file, resample to 16 kHz mono. Returns (T,) float32 tensor."""
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(wav)
    return wav.squeeze(0)


def load_audio_from_bytes(audio_bytes: bytes) -> torch.Tensor:
    """Load audio from bytes, resample to 16 kHz mono. Returns (T,) float32 tensor."""
    buf = io.BytesIO(audio_bytes)
    wav, sr = torchaudio.load(buf)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(wav)
    return wav.squeeze(0)


# ---------------------------------------------------------------------------
# VAD
# ---------------------------------------------------------------------------
_vad_model = None
_get_speech_timestamps = None


def load_vad() -> Tuple[torch.nn.Module, Callable]:
    """
    Load Silero VAD model (cached globally).

    Returns:
        Tuple of (vad_model, get_speech_timestamps function)
    """
    global _vad_model, _get_speech_timestamps
    if _vad_model is None:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        _vad_model = model
        _get_speech_timestamps = utils[0]
    return _vad_model, _get_speech_timestamps


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
def segment_audio(
    wav: torch.Tensor,
    vad_model: torch.nn.Module,
    get_speech_timestamps: Callable,
    vad_config: Optional[VADConfig] = None,
) -> List[np.ndarray]:
    """
    Run VAD and split into <=4s numpy segments.

    Args:
        wav: Audio tensor of shape (T,) at 16kHz
        vad_model: Silero VAD model
        get_speech_timestamps: Silero timestamp function
        vad_config: Optional VAD configuration (uses defaults if None)

    Returns:
        List of numpy arrays, each <=4s of audio
    """
    if vad_config is None:
        vad_config = VADConfig()

    timestamps = get_speech_timestamps(
        wav,
        vad_model,
        sampling_rate=SAMPLE_RATE,
        threshold=vad_config.threshold,
        min_speech_duration_ms=vad_config.min_speech_duration_ms,
        min_silence_duration_ms=vad_config.min_silence_duration_ms,
        speech_pad_ms=vad_config.speech_pad_ms,
    )
    # Match reference: if no speech detected, use full audio
    if not timestamps:
        timestamps = [{"start": 0, "end": len(wav)}]

    segments = []
    for ts in timestamps:
        start, end = ts["start"], ts["end"]
        seg = wav[start:end].numpy().astype(np.float32)
        if len(seg) > CHUNK_SAMPLES:
            for i in range(0, len(seg), CHUNK_SAMPLES):
                chunk = seg[i : i + CHUNK_SAMPLES]
                if len(chunk) > 0:
                    segments.append(chunk)
        else:
            segments.append(seg)
    return segments


def segment_audio_np(
    wav_np: np.ndarray,
    vad_model: torch.nn.Module,
    get_speech_timestamps: Callable,
    vad_config: Optional[VADConfig] = None,
) -> List[np.ndarray]:
    """
    Run VAD and split into <=4s segments (for numpy input).

    Args:
        wav_np: Audio numpy array at 16kHz
        vad_model: Silero VAD model
        get_speech_timestamps: Silero timestamp function
        vad_config: Optional VAD configuration (uses defaults if None)

    Returns:
        List of numpy arrays, each <=4s of audio
    """
    if vad_config is None:
        vad_config = VADConfig()

    wav_t = torch.from_numpy(wav_np).float()
    timestamps = get_speech_timestamps(
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


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
def segments_to_base64(segments: List[np.ndarray]) -> List[str]:
    """Encode each numpy segment as a base64 WAV string."""
    chunks = []
    for seg in segments:
        buf = io.BytesIO()
        sf.write(buf, seg, SAMPLE_RATE, format="WAV")
        buf.seek(0)
        chunks.append(base64.b64encode(buf.read()).decode("utf-8"))
    return chunks
