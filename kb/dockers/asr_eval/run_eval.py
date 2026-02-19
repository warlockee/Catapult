#!/usr/bin/env python3
"""
ASR Evaluation Docker Entrypoint.

Usage:
    docker run <image> --endpoint_url http://host:port/v1

Evaluates ASR model against embedded Common Voice subset and compares WER to expected baseline.

Supports two API modes:
- chat: Chat Completions API (for vLLM V1 engine with VLLM_USE_V1=1)
- transcriptions: Audio Transcriptions API (for vLLM V0 engine with VLLM_USE_V1=0)
"""
import argparse
import io
import json
import re
import requests
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import jiwer
import numpy as np
import openai
import pyarrow as pa
import soundfile as sf
import torch

from audio_utils import (
    CHUNK_SAMPLES,
    SAMPLE_RATE,
    SYSTEM_MESSAGE,
    VADConfig,
    get_user_message,
    load_vad,
    segments_to_base64,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_PATH = "/app/data/eval_subset.arrow"
BASELINE_PATH = "/app/baseline.json"
MAX_CHUNKS_PER_REQUEST = 4


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SampleResult:
    """Result for a single sample."""
    unique_id: int
    reference: str
    hypothesis: str
    reference_normalized: str
    hypothesis_normalized: str
    num_segments: int
    audio_duration_seconds: float
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def segment_audio_np(
    wav_np: np.ndarray,
    vad_model,
    get_speech_timestamps,
    vad_config: Optional[VADConfig] = None,
) -> list[np.ndarray]:
    """Run VAD and split into <=4s segments."""
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
    if not timestamps:
        return []
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


def call_api(b64_chunks: list[str], client: openai.Client, model: str, language: str | None = "English", repetition_penalty: float = 1.0) -> str:
    user_message = get_user_message(language)
    content = [{"type": "text", "text": user_message}]
    for i, chunk in enumerate(b64_chunks):
        content.append({
            "type": "audio_url",
            "audio_url": {"url": f"data:audio/wav_{i};base64,{chunk}"},
        })
    extra_body = {"skip_special_tokens": False}
    if repetition_penalty != 1.0:
        extra_body["repetition_penalty"] = repetition_penalty
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": content},
        ],
        max_completion_tokens=256,
        temperature=0.0,
        stop=["<|eot_id|>", "<|endoftext|>", "<|audio_eos|>", "<|im_end|>"],
        extra_body=extra_body,
    )
    return response.choices[0].message.content


def transcribe(b64_chunks: list[str], client: openai.Client, model: str, language: str | None = "English", repetition_penalty: float = 1.0) -> str:
    if len(b64_chunks) <= MAX_CHUNKS_PER_REQUEST:
        return call_api(b64_chunks, client, model, language, repetition_penalty)
    parts = []
    for start in range(0, len(b64_chunks), MAX_CHUNKS_PER_REQUEST):
        batch = b64_chunks[start:start + MAX_CHUNKS_PER_REQUEST]
        text = call_api(batch, client, model, language, repetition_penalty)
        if text:
            parts.append(text.strip())
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Transcriptions API (for vLLM V0 engine)
# ---------------------------------------------------------------------------
def transcribe_via_transcriptions_api(
    segments: list[np.ndarray],
    endpoint_url: str,
    model: str,
    language: str | None = "English",
) -> str:
    """
    Transcribe audio segments using the Audio Transcriptions API.

    This is for vLLM V0 engine which doesn't support Chat Completions API.
    Endpoint: POST /v1/audio/transcriptions
    """
    if not segments:
        return ""

    # Concatenate all segments into one audio buffer
    combined = np.concatenate(segments)
    buf = io.BytesIO()
    sf.write(buf, combined, SAMPLE_RATE, format="WAV")
    buf.seek(0)

    # Build the transcriptions endpoint URL
    base_url = endpoint_url.rstrip("/")
    if base_url.endswith("/v1"):
        transcriptions_url = f"{base_url}/audio/transcriptions"
    else:
        transcriptions_url = f"{base_url}/v1/audio/transcriptions"

    # Prepare the request
    files = {
        "file": ("audio.wav", buf, "audio/wav"),
    }
    data = {
        "model": model,
    }
    if language and language != "auto":
        data["language"] = language.lower()[:2]  # e.g., "English" -> "en"

    # Make the request
    response = requests.post(transcriptions_url, files=files, data=data)
    response.raise_for_status()

    result = response.json()
    return result.get("text", "")


# ---------------------------------------------------------------------------
# Allinone API (for ASR allinone Docker container)
# ---------------------------------------------------------------------------
def transcribe_via_allinone_api(
    audio_np: np.ndarray,
    endpoint_url: str,
    language: str | None = "English",
) -> str:
    """
    Transcribe audio using the allinone container's /transcribe endpoint.

    This endpoint handles VAD internally, so we send the raw audio.
    Endpoint: POST /transcribe
    """
    if len(audio_np) == 0:
        return ""

    # Create WAV buffer from raw audio
    buf = io.BytesIO()
    sf.write(buf, audio_np, SAMPLE_RATE, format="WAV")
    buf.seek(0)

    # Build the transcribe endpoint URL
    base_url = endpoint_url.rstrip("/")
    # Remove /v1 suffix if present since allinone doesn't use it
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    transcribe_url = f"{base_url}/transcribe"

    # Prepare the request (multipart/form-data)
    files = {
        "file": ("audio.wav", buf, "audio/wav"),
    }
    data = {}
    if language and language != "auto":
        data["language"] = language

    # Make the request
    response = requests.post(transcribe_url, files=files, data=data)
    response.raise_for_status()

    result = response.json()
    return result.get("transcription", "")


# ---------------------------------------------------------------------------
# WER normalization
# ---------------------------------------------------------------------------
_english_normalizer = None

def get_normalizer():
    global _english_normalizer
    if _english_normalizer is None:
        try:
            from whisper_normalizer.english import EnglishTextNormalizer
            _english_normalizer = EnglishTextNormalizer()
        except ImportError:
            pass
    return _english_normalizer


def normalize(text: str) -> str:
    normalizer = get_normalizer()
    if normalizer is not None:
        return normalizer(text)
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# API Auto-Detection
# ---------------------------------------------------------------------------
def detect_api_mode(endpoint_url: str, timeout: float = 10.0) -> str:
    """
    Auto-detect the API mode by probing the endpoint.

    Detection order:
    1. Check /v1/models - if responds, it's vLLM raw API -> use 'chat'
    2. Check /health for vllm_url field - if present, it's allinone -> use 'allinone'
    3. Check /transcribe endpoint exists - if yes, use 'allinone'
    4. Default to 'chat'

    Returns: 'chat', 'allinone', or 'transcriptions'
    """
    base_url = endpoint_url.rstrip("/")
    # Remove /v1 suffix for base URL probing
    if base_url.endswith("/v1"):
        base_url_no_v1 = base_url[:-3]
    else:
        base_url_no_v1 = base_url

    # 1. Check for vLLM raw API (/v1/models)
    try:
        v1_base = base_url if base_url.endswith("/v1") else f"{base_url}/v1"
        resp = requests.get(f"{v1_base}/models", timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            if "data" in data and isinstance(data["data"], list):
                print(f"  Detected: vLLM raw API at {v1_base}/models")
                return "chat"
    except Exception:
        pass

    # 2. Check /health for allinone signature (has vllm_url field)
    try:
        resp = requests.get(f"{base_url_no_v1}/health", timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            if "vllm_url" in data:
                print(f"  Detected: Allinone container (has vllm_url in /health)")
                return "allinone"
    except Exception:
        pass

    # 3. Check if /transcribe endpoint exists (OPTIONS or GET)
    try:
        resp = requests.options(f"{base_url_no_v1}/transcribe", timeout=timeout)
        if resp.status_code in (200, 204, 405):  # 405 = Method Not Allowed means endpoint exists
            print(f"  Detected: Allinone container (/transcribe endpoint exists)")
            return "allinone"
    except Exception:
        pass

    # 4. Default to chat mode
    print(f"  Detection inconclusive, defaulting to 'chat' mode")
    return "chat"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ASR Evaluation")
    parser.add_argument("--endpoint_url", required=True, help="ASR endpoint URL (e.g., http://host:port/v1)")
    parser.add_argument("--model", default="asr-model", help="Model name at the endpoint")
    parser.add_argument("--limit", type=int, default=500, help="Max samples to eval")
    parser.add_argument("--offset", type=int, default=0, help="Start index")
    parser.add_argument("--language", default="English", help="Language for transcription")
    parser.add_argument("--output", "-o", help="Output JSON file for per-sample results")
    parser.add_argument("--expected-wer", type=float, help="Override expected WER (e.g., 0.13 for 13%%)")
    parser.add_argument("--tolerance", type=float, help="Override WER tolerance (e.g., 0.02 for 2%%)")
    parser.add_argument("--no-baseline", action="store_true", help="Just measure WER without baseline comparison")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty for generation (default: 1.1 to prevent loops)")
    parser.add_argument("--api-mode", choices=["auto", "chat", "transcriptions", "allinone"], default="auto",
                        help="API mode: 'auto' to auto-detect, 'chat' for Chat Completions API (vLLM), 'transcriptions' for Audio Transcriptions API, 'allinone' for ASR allinone container")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Load baseline
    baseline = json.loads(Path(BASELINE_PATH).read_text())
    expected_wer = args.expected_wer if args.expected_wer is not None else baseline["expected_wer"]
    tolerance = args.tolerance if args.tolerance is not None else baseline["expected_wer_tolerance"]

    print("=" * 70)
    print("ASR Evaluation Docker")
    print("=" * 70)
    print(f"Endpoint:     {args.endpoint_url}")
    print(f"Model:        {args.model}")

    # Auto-detect API mode if needed
    api_mode = args.api_mode
    if api_mode == "auto":
        print("API Mode:     auto (detecting...)")
        api_mode = detect_api_mode(args.endpoint_url)
        print(f"API Mode:     {api_mode} (auto-detected)")
    else:
        api_mode_desc = {
            "chat": "Chat Completions (vLLM)",
            "allinone": "Allinone (/transcribe)",
            "transcriptions": "Audio Transcriptions",
        }
        print(f"API Mode:     {api_mode} ({api_mode_desc.get(api_mode, api_mode)})")

    if not args.no_baseline:
        print(f"Expected WER: {expected_wer:.2%} (+/- {tolerance:.2%})")
    else:
        print("Mode:         Measurement only (no baseline comparison)")
    print("=" * 70)

    # VAD config matching baseline
    vad_config = VADConfig(
        threshold=0.55,
        min_speech_duration_ms=125,
        min_silence_duration_ms=200,
        speech_pad_ms=300,
    )

    # Load dataset
    print(f"\nLoading dataset {DATASET_PATH}...")
    table = pa.ipc.open_file(DATASET_PATH).read_all()
    total = table.num_rows
    end_idx = min(args.offset + args.limit, total)
    print(f"  {total} total samples, evaluating [{args.offset}:{end_idx}]")

    # Init
    print("\nLoading VAD model...")
    vad_model, get_speech_timestamps = load_vad()

    # Ensure endpoint URL has /v1 suffix for OpenAI client (chat mode)
    endpoint_url_for_client = args.endpoint_url.rstrip("/")
    if api_mode == "chat" and not endpoint_url_for_client.endswith("/v1"):
        endpoint_url_for_client = f"{endpoint_url_for_client}/v1"
    client = openai.Client(api_key="EMPTY", base_url=endpoint_url_for_client)

    refs = []
    hyps = []
    sample_results: List[SampleResult] = []
    errors = 0
    no_speech = 0
    t_start = time.time()

    for idx in range(args.offset, end_idx):
        audio = np.array(table.column("audio_array")[idx].as_py(), dtype=np.float32)
        ref = table.column("audio_transcript")[idx].as_py()
        dur = len(audio) / SAMPLE_RATE

        try:
            lang = None if args.language == "auto" else args.language

            if api_mode == "allinone":
                # Use allinone container's /transcribe endpoint (handles VAD internally)
                hyp = transcribe_via_allinone_api(audio, args.endpoint_url, lang)
                segments = []  # VAD is done server-side
            else:
                # Do VAD locally for chat/transcriptions modes
                segments = segment_audio_np(audio, vad_model, get_speech_timestamps, vad_config)
                if not segments:
                    hyp = ""
                    no_speech += 1
                elif api_mode == "transcriptions":
                    # Use Audio Transcriptions API (vLLM V0)
                    hyp = transcribe_via_transcriptions_api(
                        segments, args.endpoint_url, args.model, lang
                    )
                else:
                    # Use Chat Completions API (vLLM V1)
                    b64 = segments_to_base64(segments)
                    hyp = transcribe(b64, client, args.model, lang, args.repetition_penalty)

            ref_n = normalize(ref)
            hyp_n = normalize(hyp)

            sample_results.append(SampleResult(
                unique_id=idx,
                reference=ref,
                hypothesis=hyp,
                reference_normalized=ref_n,
                hypothesis_normalized=hyp_n,
                num_segments=len(segments),
                audio_duration_seconds=dur,
            ))

            if ref_n:
                refs.append(ref_n)
                hyps.append(hyp_n)

            n = idx - args.offset + 1
            total = end_idx - args.offset
            # Always print progress for real-time feedback
            elapsed = time.time() - t_start
            rate = n / elapsed
            if refs:
                running_wer = jiwer.wer(refs, hyps)
            else:
                running_wer = 0
            # Standard progress format for all eval containers (parsed by backend)
            print(f"EVAL_PROGRESS: {n}/{total}")
            print(
                f"  [{idx:>5}] {dur:>5.1f}s  segs={len(segments):<2}  "
                f"WER={running_wer:.3f}  ({n}/{total} @ {rate:.1f}/s)"
            )

        except Exception as e:
            errors += 1
            n = idx - args.offset + 1
            total = end_idx - args.offset
            # Standard progress format (even for errors)
            print(f"EVAL_PROGRESS: {n}/{total}")
            print(f"  [{idx:>5}] ERROR: {e}")
            sample_results.append(SampleResult(
                unique_id=idx,
                reference=ref if 'ref' in dir() else "",
                hypothesis="",
                reference_normalized="",
                hypothesis_normalized="",
                num_segments=0,
                audio_duration_seconds=dur if 'dur' in dir() else 0.0,
                error=str(e),
            ))

    # Final metrics
    elapsed = time.time() - t_start

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Dataset:     common_voice_15_en_subset")
    print(f"Samples:     {end_idx - args.offset} (offset={args.offset})")
    print(f"Elapsed:     {elapsed:.1f}s ({(end_idx - args.offset) / elapsed:.1f} samples/s)")
    print(f"Errors:      {errors}")
    print(f"No-speech:   {no_speech}")

    wer = 0.0
    cer = 0.0
    if refs:
        wer = jiwer.wer(refs, hyps)
        cer = jiwer.cer(refs, hyps)
        print(f"\nActual WER:   {wer:.4f} ({wer * 100:.2f}%)")
        print(f"CER:          {cer:.4f} ({cer * 100:.2f}%)")

        if args.no_baseline:
            # Just report WER without comparison
            print("\n" + "-" * 70)
            print("Measurement complete (no baseline comparison)")
            print("-" * 70)
            result_status = "PASS"
        else:
            print(f"Expected WER: {expected_wer:.4f} ({expected_wer * 100:.2f}%)")
            # Compare with baseline
            diff = wer - expected_wer
            print("\n" + "-" * 70)
            if abs(diff) <= tolerance:
                print(f"PASS: WER within tolerance (diff: {diff:+.4f})")
                result_status = "PASS"
            elif diff < 0:
                print(f"PASS: WER BETTER than baseline (diff: {diff:+.4f})")
                result_status = "PASS"
            else:
                print(f"FAIL: WER exceeds tolerance (diff: {diff:+.4f}, tolerance: {tolerance:.4f})")
                result_status = "FAIL"
            print("-" * 70)
    else:
        print("\nNo scored utterances.")
        result_status = "FAIL"

    # Write JSON output if requested
    if args.output:
        output_data = {
            "status": result_status,
            "wer": wer,
            "expected_wer": expected_wer,
            "wer_diff": wer - expected_wer,
            "cer": cer,
            "samples_evaluated": len(refs),
            "samples_with_errors": errors,
            "no_speech_count": no_speech,
            "elapsed_seconds": elapsed,
            "model": args.model,
            "endpoint_url": args.endpoint_url,
            "results": [asdict(s) for s in sample_results],
        }
        Path(args.output).write_text(
            json.dumps(output_data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"\nResults saved to: {args.output}")

    print("=" * 70)

    # Exit with appropriate code
    sys.exit(0 if result_status == "PASS" else 1)


if __name__ == "__main__":
    main()
