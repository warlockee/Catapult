"""
ASR Client - VAD + 4s chunking + vLLM OpenAI API.

Usage:
    python asr_client.py <audio_file> [--url http://localhost:26007/v1] [--language English]

Pipeline: audio file -> resample 16kHz -> Silero VAD -> 4s segments
          -> base64 encode -> vLLM /v1/chat/completions -> text
"""
import argparse
import math
import time
from typing import Optional

import openai

from audio_utils import (
    CHUNK_SAMPLES,
    SAMPLE_RATE,
    SYSTEM_MESSAGE,
    VADConfig,
    get_user_message,
    load_audio_from_path,
    load_vad,
    segment_audio,
    segments_to_base64,
)

# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------
MAX_CHUNKS_PER_REQUEST = 4  # ~16s of audio max per API call


def _call_api(
    base64_chunks: list[str],
    client: openai.Client,
    model: str,
    language: Optional[str] = "English",
) -> str:
    """Send a batch of audio chunks to vLLM chat completions endpoint."""
    user_message = get_user_message(language)
    content = [{"type": "text", "text": user_message}]
    for i, chunk in enumerate(base64_chunks):
        content.append({
            "type": "audio_url",
            "audio_url": {"url": f"data:audio/wav_{i};base64,{chunk}"},
        })

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": content},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=256,
        temperature=0.0,
        stop=["<|eot_id|>", "<|endoftext|>", "<|audio_eos|>", "<|im_end|>"],
        extra_body={"skip_special_tokens": False},
    )
    return response.choices[0].message.content


def transcribe(
    base64_chunks: list[str],
    client: openai.Client,
    model: str,
    language: Optional[str] = "English",
) -> str:
    """Transcribe audio chunks, batching into groups to avoid repetition."""
    if len(base64_chunks) <= MAX_CHUNKS_PER_REQUEST:
        return _call_api(base64_chunks, client, model, language)

    parts = []
    for start in range(0, len(base64_chunks), MAX_CHUNKS_PER_REQUEST):
        batch = base64_chunks[start : start + MAX_CHUNKS_PER_REQUEST]
        batch_num = start // MAX_CHUNKS_PER_REQUEST + 1
        total_batches = math.ceil(len(base64_chunks) / MAX_CHUNKS_PER_REQUEST)
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunk(s))...")
        text = _call_api(batch, client, model, language)
        if text:
            parts.append(text.strip())
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ASR client with VAD + vLLM")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--url", default="http://localhost:26007/v1", help="vLLM base URL")
    parser.add_argument("--model", default="asr-model", help="Served model name")
    parser.add_argument(
        "--language",
        default="English",
        help="Language for transcription (e.g., 'English', 'Chinese'). Use 'auto' for auto-detect.",
    )
    # VAD parameters
    parser.add_argument("--vad-threshold", type=float, default=0.55)
    parser.add_argument("--vad-min-speech-ms", type=int, default=125)
    parser.add_argument("--vad-min-silence-ms", type=int, default=200)
    parser.add_argument("--vad-speech-pad-ms", type=int, default=300)
    args = parser.parse_args()

    # Build VAD config
    vad_config = VADConfig(
        threshold=args.vad_threshold,
        min_speech_duration_ms=args.vad_min_speech_ms,
        min_silence_duration_ms=args.vad_min_silence_ms,
        speech_pad_ms=args.vad_speech_pad_ms,
    )

    print("Loading VAD model...")
    vad_model, get_speech_timestamps = load_vad()

    print(f"Loading audio: {args.audio}")
    wav = load_audio_from_path(args.audio)
    print(f"  Duration: {len(wav)/SAMPLE_RATE:.2f}s, samples: {len(wav)}")

    print("Running VAD + segmentation...")
    segments = segment_audio(wav, vad_model, get_speech_timestamps, vad_config)
    if not segments:
        print("No speech detected.")
        return
    print(f"  {len(segments)} segment(s): {[f'{len(s)/SAMPLE_RATE:.2f}s' for s in segments]}")

    print("Encoding segments to base64...")
    b64_chunks = segments_to_base64(segments)

    print(f"Calling vLLM at {args.url} (model={args.model})...")
    client = openai.Client(api_key="EMPTY", base_url=args.url)
    t0 = time.time()
    # 'auto' means use auto-detect (None)
    lang = None if args.language == "auto" else args.language
    text = transcribe(b64_chunks, client, args.model, lang)
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"Transcription ({elapsed:.2f}s):")
    print(text)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
