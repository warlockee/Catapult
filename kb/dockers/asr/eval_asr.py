"""
Evaluate ASR against Common Voice dataset.

Usage:
    python eval_asr.py /data/audio_eval/asr/common_voice_15_en.arrow \
        [--url http://localhost:26007/v1] [--model asr-model] \
        [--limit 100] [--offset 0] [--language English]

Reads audio_array + audio_transcript from Arrow file, runs each sample
through VAD + vLLM, and computes WER/CER.
"""
import argparse
import io
import json
import re
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
MAX_CHUNKS_PER_REQUEST = 4


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SampleResult:
    """Result for a single sample."""
    unique_id: int  # Index in dataset (matches UNIQUE_KEY_NAME convention)
    reference: str  # Ground truth transcript
    hypothesis: str  # Model prediction
    reference_normalized: str  # Normalized reference
    hypothesis_normalized: str  # Normalized hypothesis
    num_segments: int  # Number of VAD segments
    audio_duration_seconds: float
    error: Optional[str] = None  # Error message if processing failed


# ---------------------------------------------------------------------------
# Pipeline (adapted for numpy arrays from Arrow)
# ---------------------------------------------------------------------------
def segment_audio_np(
    wav_np: np.ndarray,
    vad_model,
    get_speech_timestamps,
    vad_config: Optional[VADConfig] = None,
) -> list[np.ndarray]:
    """Run VAD and split into <=4s segments (for numpy input)."""
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
# WER normalization (using whisper_normalizer to match reference implementation)
# ---------------------------------------------------------------------------
_english_normalizer = None

def get_normalizer():
    """Get or create the English text normalizer (lazy initialization)."""
    global _english_normalizer
    if _english_normalizer is None:
        try:
            from whisper_normalizer.english import EnglishTextNormalizer
            _english_normalizer = EnglishTextNormalizer()
            print("Using whisper_normalizer.english.EnglishTextNormalizer")
        except ImportError:
            print("WARNING: whisper_normalizer not available, falling back to basic normalization")
    return _english_normalizer

def normalize(text: str) -> str:
    """Normalize text for WER comparison using whisper_normalizer."""
    normalizer = get_normalizer()
    if normalizer is not None:
        return normalizer(text)
    # Fallback to basic normalization
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", "", text)  # keep apostrophes
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate ASR on Arrow dataset")
    parser.add_argument("dataset", help="Path to .arrow file")
    parser.add_argument("--url", default="http://localhost:26007/v1")
    parser.add_argument("--model", default="asr-model")
    parser.add_argument("--limit", type=int, default=100, help="Max samples to eval")
    parser.add_argument("--offset", type=int, default=0, help="Start index")
    parser.add_argument("--language", default="English", help="Language for transcription (e.g., 'English', 'Chinese'). Use 'auto' for auto-detect.")
    parser.add_argument("--output", "-o", help="Output JSON file for per-sample results")
    # VAD parameters (defaults match reference evaluation)
    parser.add_argument("--vad-threshold", type=float, default=0.55, help="VAD speech probability threshold (default: 0.55)")
    parser.add_argument("--vad-min-speech-ms", type=int, default=125, help="Minimum speech chunk duration in ms (default: 125)")
    parser.add_argument("--vad-min-silence-ms", type=int, default=200, help="Minimum silence between speech chunks in ms (default: 200)")
    parser.add_argument("--vad-speech-pad-ms", type=int, default=300, help="Padding added to speech chunks in ms (default: 300)")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty for generation (default: 1.0, use 1.1+ to prevent loops)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Build VAD config from CLI args
    vad_config = VADConfig(
        threshold=args.vad_threshold,
        min_speech_duration_ms=args.vad_min_speech_ms,
        min_silence_duration_ms=args.vad_min_silence_ms,
        speech_pad_ms=args.vad_speech_pad_ms,
    )

    # Load dataset
    print(f"Loading {args.dataset}...")
    table = pa.ipc.open_file(args.dataset).read_all()
    total = table.num_rows
    end_idx = min(args.offset + args.limit, total)
    print(f"  {total} total samples, evaluating [{args.offset}:{end_idx}]")

    # Init
    print("Loading VAD model...")
    vad_model, get_speech_timestamps = load_vad()
    client = openai.Client(api_key="EMPTY", base_url=args.url)

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
            segments = segment_audio_np(audio, vad_model, get_speech_timestamps, vad_config)
            if not segments:
                hyp = ""
                no_speech += 1
            else:
                b64 = segments_to_base64(segments)
                # 'auto' means use auto-detect (None)
                lang = None if args.language == "auto" else args.language
                hyp = transcribe(b64, client, args.model, lang, args.repetition_penalty)

            ref_n = normalize(ref)
            hyp_n = normalize(hyp)

            # Track per-sample result (unique_id matches dataset index)
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
            if args.verbose or n % 10 == 0 or n == 1:
                elapsed = time.time() - t_start
                rate = n / elapsed
                # Running WER
                if refs:
                    running_wer = jiwer.wer(refs, hyps)
                else:
                    running_wer = 0
                print(
                    f"  [{idx:>5}] {dur:>5.1f}s  segs={len(segments):<2}  "
                    f"WER={running_wer:.3f}  ({n}/{end_idx - args.offset} "
                    f"@ {rate:.1f}/s)"
                )
                if args.verbose:
                    print(f"         ref: {ref_n[:80]}")
                    print(f"         hyp: {hyp_n[:80]}")

        except Exception as e:
            errors += 1
            print(f"  [{idx:>5}] ERROR: {e}")
            # Track failed sample
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
    print(f"\n{'=' * 60}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {end_idx - args.offset} (offset={args.offset})")
    print(f"Elapsed: {elapsed:.1f}s ({(end_idx - args.offset) / elapsed:.1f} samples/s)")
    print(f"Errors:  {errors}")
    print(f"No-speech: {no_speech}")

    wer = 0.0
    cer = 0.0
    if refs:
        wer = jiwer.wer(refs, hyps)
        cer = jiwer.cer(refs, hyps)
        print(f"\nWER: {wer:.4f} ({wer * 100:.2f}%)")
        print(f"CER: {cer:.4f} ({cer * 100:.2f}%)")
        print(f"Scored utterances: {len(refs)}")
    else:
        print("\nNo scored utterances.")

    # Write JSON output if requested
    if args.output:
        output_data = {
            "wer": wer,
            "cer": cer,
            "samples_evaluated": len(refs),
            "samples_with_errors": errors,
            "no_speech_count": no_speech,
            "dataset_path": args.dataset,
            "elapsed_seconds": elapsed,
            "model": args.model,
            "language": args.language,
            "results": [asdict(s) for s in sample_results],
        }
        Path(args.output).write_text(
            json.dumps(output_data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"\nResults saved to: {args.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
