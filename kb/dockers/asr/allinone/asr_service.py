"""
ASR All-in-One Service

Complete ASR pipeline: Audio → VAD → 4s Cut → Inference → Text

Provides a simple API:
    POST /transcribe - Upload audio file, get transcription
    GET /health - Health check
"""
import asyncio
import base64
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:26007/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "asr-model")
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 64000  # 4 seconds at 16kHz

# Match tools eval: max 4 chunks (~16s) per API call
MAX_CHUNKS_PER_REQUEST = 4

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VADConfig:
    """Silero VAD configuration matching reference implementation."""
    threshold: float = 0.55
    min_speech_duration_ms: int = 125
    min_silence_duration_ms: int = 200
    speech_pad_ms: int = 300


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
vad_model = None
get_speech_timestamps = None
openai_client = None


def load_vad():
    """Load Silero VAD model."""
    global vad_model, get_speech_timestamps
    if vad_model is None:
        logger.info("Loading Silero VAD model...")
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        vad_model = model
        get_speech_timestamps = utils[0]
        logger.info("VAD model loaded")
    return vad_model, get_speech_timestamps


def init_openai_client():
    """Initialize OpenAI client for vLLM."""
    global openai_client
    if openai_client is None:
        import openai
        openai_client = openai.OpenAI(
            base_url=VLLM_URL,
            api_key="not-needed",
        )
        logger.info(f"OpenAI client initialized: {VLLM_URL}")
    return openai_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup."""
    load_vad()
    init_openai_client()
    yield


app = FastAPI(
    title="ASR Service",
    description="All-in-one ASR: Audio → VAD → 4s Cut → Inference → Text",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Audio Processing
# ---------------------------------------------------------------------------
def load_audio(audio_bytes: bytes) -> np.ndarray:
    """Load audio from bytes, resample to 16kHz mono."""
    buf = io.BytesIO(audio_bytes)
    try:
        # Try soundfile first
        audio, sr = sf.read(buf)
    except Exception:
        # Fallback to torchaudio
        buf.seek(0)
        wav, sr = torchaudio.load(buf)
        audio = wav.numpy()
        if len(audio.shape) > 1:
            audio = audio.mean(axis=0)

    # Convert to mono if needed
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != SAMPLE_RATE:
        wav_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        audio = resampler(wav_tensor).squeeze(0).numpy()

    return audio.astype(np.float32)


def segment_audio(audio: np.ndarray, vad_config: VADConfig) -> list:
    """Run VAD and segment audio into <=4s chunks."""
    model, get_ts = load_vad()

    wav_tensor = torch.from_numpy(audio).float()
    timestamps = get_ts(
        wav_tensor,
        model,
        sampling_rate=SAMPLE_RATE,
        threshold=vad_config.threshold,
        min_speech_duration_ms=vad_config.min_speech_duration_ms,
        min_silence_duration_ms=vad_config.min_silence_duration_ms,
        speech_pad_ms=vad_config.speech_pad_ms,
    )

    # Match reference: if no speech detected, use full audio
    if not timestamps:
        timestamps = [{"start": 0, "end": len(audio)}]

    segments = []
    for ts in timestamps:
        seg = audio[ts["start"]:ts["end"]]
        # Split into 4s chunks if needed
        if len(seg) > CHUNK_SAMPLES:
            for i in range(0, len(seg), CHUNK_SAMPLES):
                chunk = seg[i:i + CHUNK_SAMPLES]
                if len(chunk) > 0:
                    segments.append(chunk)
        else:
            segments.append(seg)

    return segments


def encode_segment(segment: np.ndarray) -> str:
    """Encode audio segment as base64 WAV."""
    buf = io.BytesIO()
    sf.write(buf, segment, SAMPLE_RATE, format="WAV")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _call_vllm(b64_chunks: list[str], language: str = "English") -> str:
    """Send a batch of audio chunks to vLLM."""
    client = init_openai_client()

    # Build user message with language
    if language.lower() == "auto":
        user_text = "Your task is to listen to audio input and output the exact spoken words as plain text."
    else:
        user_text = f"Your task is to listen to audio input and output the exact spoken words as plain text in {language}."

    # Build content with audio chunks
    content = [{"type": "text", "text": user_text}]
    for i, chunk in enumerate(b64_chunks):
        content.append({
            "type": "audio_url",
            "audio_url": {"url": f"data:audio/wav_{i};base64,{chunk}"}
        })

    messages = [
        {"role": "system", "content": "You are an automatic speech recognition (ASR) system."},
        {"role": "user", "content": content}
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_completion_tokens=256,
        temperature=0.0,
        stop=["<|eot_id|>", "<|endoftext|>", "<|audio_eos|>", "<|im_end|>"],
        extra_body={"skip_special_tokens": False},
    )

    return response.choices[0].message.content


def transcribe_segments(segments: list, language: str = "English") -> str:
    """Send segments to vLLM and get transcription (batched to match tools eval)."""
    if not segments:
        return ""

    # Encode all segments
    b64_chunks = [encode_segment(seg) for seg in segments]

    # If few enough chunks, single request
    if len(b64_chunks) <= MAX_CHUNKS_PER_REQUEST:
        return _call_vllm(b64_chunks, language).strip()

    # Otherwise batch into groups of MAX_CHUNKS_PER_REQUEST
    parts = []
    for start in range(0, len(b64_chunks), MAX_CHUNKS_PER_REQUEST):
        batch = b64_chunks[start:start + MAX_CHUNKS_PER_REQUEST]
        text = _call_vllm(batch, language)
        if text:
            parts.append(text.strip())
    return " ".join(parts)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model": MODEL_NAME, "vllm_url": VLLM_URL}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(default="English"),
):
    """
    Transcribe audio file.

    Full pipeline: Audio → VAD → 4s Cut → Inference → Text

    Args:
        file: Audio file (wav, mp3, etc.)
        language: Target language (default: English, use "auto" for auto-detect)

    Returns:
        JSON with transcription and metadata
    """
    start_time = time.time()

    try:
        # Read audio
        audio_bytes = await file.read()
        audio = load_audio(audio_bytes)
        audio_duration = len(audio) / SAMPLE_RATE

        # VAD + segmentation
        vad_config = VADConfig()
        segments = segment_audio(audio, vad_config)

        # Transcribe
        transcription = transcribe_segments(segments, language)

        elapsed = time.time() - start_time

        return JSONResponse({
            "transcription": transcription,
            "language": language,
            "audio_duration_seconds": round(audio_duration, 2),
            "num_segments": len(segments),
            "processing_time_seconds": round(elapsed, 2),
            "realtime_factor": round(audio_duration / elapsed, 2) if elapsed > 0 else 0,
        })

    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe/batch")
async def transcribe_batch(
    files: list[UploadFile] = File(...),
    language: str = Form(default="English"),
):
    """
    Transcribe multiple audio files.

    Args:
        files: List of audio files
        language: Target language

    Returns:
        JSON with list of transcriptions
    """
    results = []
    for f in files:
        try:
            audio_bytes = await f.read()
            audio = load_audio(audio_bytes)
            segments = segment_audio(audio, VADConfig())
            transcription = transcribe_segments(segments, language)
            results.append({
                "filename": f.filename,
                "transcription": transcription,
                "status": "success",
            })
        except Exception as e:
            results.append({
                "filename": f.filename,
                "transcription": None,
                "status": "error",
                "error": str(e),
            })

    return JSONResponse({"results": results})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ASR Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--vllm-url", default="http://localhost:26007/v1", help="vLLM URL")
    parser.add_argument("--model", default="asr-model", help="Model name")
    args = parser.parse_args()

    os.environ["VLLM_URL"] = args.vllm_url
    os.environ["MODEL_NAME"] = args.model

    uvicorn.run(app, host=args.host, port=args.port)
