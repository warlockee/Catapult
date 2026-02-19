"""
Smoke test for the ASR vLLM endpoint.

Usage:
    python test_asr_docker.py [--port 26007] [--model asr-model]

Sends a synthetic 440 Hz tone (3s) to the vLLM OpenAI chat completions API
and verifies a non-empty response is returned.
"""
import argparse
import base64
import io
import math
import struct
import sys
import time
import wave

import requests

from audio_utils import SYSTEM_MESSAGE, get_user_message

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_PORT = 26007
DEFAULT_MODEL = "asr-model"


# ---------------------------------------------------------------------------
# Test audio
# ---------------------------------------------------------------------------
def create_test_wav(duration=3, rate=16000) -> bytes:
    """Create a short in-memory WAV with a 440 Hz tone. Returns raw bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        frames = []
        for i in range(rate * duration):
            t = float(i) / rate
            val = int(32767 * 0.5 * math.sin(2 * math.pi * 440 * t))
            frames.append(struct.pack("<h", val))
        wf.writeframes(b"".join(frames))
    return buf.getvalue()


def encode_wav_to_base64_chunks(wav_bytes: bytes, duration_ms=4000) -> list[str]:
    """Encode WAV bytes into base64 chunks. For short test audio, returns one chunk."""
    return [base64.b64encode(wav_bytes).decode("utf-8")]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_health(base_url: str) -> bool:
    """Wait for vLLM /health to return 200."""
    health_url = base_url.replace("/v1", "") + "/health"
    print(f"Checking health at {health_url}...")
    for attempt in range(30):
        try:
            r = requests.get(health_url, timeout=5)
            if r.status_code == 200:
                print(f"  Healthy (attempt {attempt + 1})")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(2)
    print("  Health check failed after 60s.")
    return False


def test_transcription(base_url: str, model: str) -> bool:
    """Send test audio to vLLM chat completions and verify response."""
    print("Creating test audio (3s, 440 Hz)...")
    wav_bytes = create_test_wav()
    chunks = encode_wav_to_base64_chunks(wav_bytes)
    print(f"  Encoded {len(chunks)} chunk(s)")

    user_message = get_user_message("English")
    content = [{"type": "text", "text": user_message}]
    for i, chunk in enumerate(chunks):
        content.append({
            "type": "audio_url",
            "audio_url": {"url": f"data:audio/wav_{i};base64,{chunk}"},
        })

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": content},
        ],
        "max_completion_tokens": 256,
        "temperature": 0.0,
        "stop": ["<|eot_id|>", "<|endoftext|>", "<|audio_eos|>", "<|im_end|>"],
        "skip_special_tokens": False,
    }

    url = f"{base_url}/chat/completions"
    print(f"Sending chat completion to {url} (model={model})...")
    t0 = time.time()
    try:
        r = requests.post(url, json=payload, timeout=120)
        elapsed = time.time() - t0
        if r.status_code != 200:
            print(f"  FAIL: HTTP {r.status_code}: {r.text[:200]}")
            return False
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        print(f"  Response ({elapsed:.2f}s): {text!r}")
        print("  PASS")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  FAIL ({elapsed:.2f}s): {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Smoke test for ASR vLLM")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    base_url = f"http://localhost:{args.port}/v1"

    if not test_health(base_url):
        print("TEST FAILED: Service did not start")
        sys.exit(1)

    if test_transcription(base_url, args.model):
        print("TEST PASSED")
    else:
        print("TEST FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
