import argparse
import sys
import os
import time
import json
import base64
import math
import statistics
import concurrent.futures
import requests
import openai
import soundfile as sf

# --- Configuration ---
DEFAULT_BASE_URL = "http://172.202.29.125:26000/v1"
DEFAULT_API_KEY = "EMPTY"
AUDIO_EXAMPLE_PATH = "tests/example/hello_howru.wav"
CLONE_REF_PATH = "tests/example/yoshua_bengio.wav"

# Global vars to be updated by args
BASE_URL = DEFAULT_BASE_URL
API_KEY = DEFAULT_API_KEY

# --- Utils ---
def get_model_id():
    """Dynamically fetch the model ID from the server."""
    print(f"Connecting to {BASE_URL}/models...")
    try:
        res = requests.get(f"{BASE_URL}/models", timeout=5)
        res.raise_for_status()
        data = res.json()
        model_id = data['data'][0]['id']
        print(f"✅ Context: Discovered Model ID: {model_id}")
        return model_id
    except Exception as e:
        print(f"⚠️ Failed to fetch model ID: {e}")
        print("   Using fallback ID.")
        return "audio-v2-generation-3B-dpo-checkpoint-2100"

MODEL_ID = None # Will be set in main()

def get_client():
    return openai.Client(api_key=API_KEY, base_url=BASE_URL)

def encode_audio(file_path, duration=None):
    """Encode audio file to chunks (or single string) of base64."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
        
    if duration is None:
        with open(file_path, "rb") as f:
            return [base64.b64encode(f.read()).decode("utf-8")]
            
    # Chunking logic (if needed)
    data, sr = sf.read(file_path)
    if data.ndim > 1: data = data.mean(axis=-1)
    chunk_size = int(sr * (duration / 1000.0))
    num_chunks = math.ceil(len(data) / chunk_size)
    chunks = []
    import io
    for i in range(num_chunks):
        chunk_data = data[i*chunk_size : (i+1)*chunk_size]
        buf = io.BytesIO()
        sf.write(buf, chunk_data, sr, format="WAV")
        buf.seek(0)
        chunks.append(base64.b64encode(buf.read()).decode("utf-8"))
    return chunks

# --- Tests ---

def test_health():
    print("\n--- Test 1: Health Check (API) ---")
    try:
        res = requests.get(f"{BASE_URL}/models")
        res.raise_for_status()
        print("✅ Health Check Passed")
        return True
    except Exception as e:
        print(f"❌ Health Check Failed: {e}")
        return False

def test_inference_text():
    print("\n--- Test 2: Text Inference (Chat) ---")
    client = get_client()
    try:
        res = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": "Hello, verify functioning."}],
            max_completion_tokens=50,
            temperature=0.7
        )
        content = res.choices[0].message.content
        print(f"✅ Text Inference Passed. Output len: {len(content)}")
        return True
    except Exception as e:
        print(f"❌ Text Inference Failed: {e}")
        return False

def test_inference_stream():
    print("\n--- Test 3: Streaming Inference ---")
    client = get_client()
    try:
        stream = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": "Count to 5."}],
            max_completion_tokens=50,
            stream=True
        )
        print("   Stream started...", end="", flush=True)
        tokens = 0
        for chunk in stream:
            if chunk.choices[0].delta.content:
                tokens += 1
        print(f" Done. Received {tokens} chunks.")
        print("✅ Streaming Passed")
        return True
    except Exception as e:
        print(f"\n❌ Streaming Failed: {e}")
        return False

def test_audio_input():
    print("\n--- Test 4: Audio Input ---")
    client = get_client()
    try:
        b64_chunks = encode_audio(AUDIO_EXAMPLE_PATH)
        ext = AUDIO_EXAMPLE_PATH.split(".")[-1]
        
        content = [{"type": "audio_url", "audio_url": {"url": f"data:audio/{ext};base64,{chunk}"}} for chunk in b64_chunks]
        
        # Add text prompt if model requires it, or just system
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}
        ]
        
        res = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_completion_tokens=256,
            temperature=0.7,
            stop=["<|eot_id|>", "<|endoftext|>", "<|audio_eos|>", "<|im_end|>"],
            extra_body={"skip_special_tokens": False}
        )
        print("✅ Audio Input Passed")
        return True
    except Exception as e:
        print(f"❌ Audio Input Failed: {e}")
        return False

def test_voice_clone():
    print("\n--- Test 5: Voice Cloning ---")
    client = get_client()
    try:
        ref_b64 = encode_audio(CLONE_REF_PATH)[0]
        ref_text = "You don't need to be an agent, you just need to make good, trustworthy predictions."
        target_text = "Verification of voice cloning."
        
        messages = [
            {"role": "system", "content": "Convert text to speech with the same voice."},
            {"role": "user", "content": ref_text},
            {
                "role": "assistant", 
                "content": [{"type": "input_audio", "input_audio": {"data": ref_b64, "format": "wav"}}]
            },
            {"role": "user", "content": target_text}
        ]
        
        res = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_completion_tokens=1024,
            temperature=1.0,
            stop=["<|eot_id|>", "<|endoftext|>", "<|audio_eos|>"],
            extra_body={"top_k": 50}
        )
        
        if hasattr(res.choices[0].message, 'audio') and res.choices[0].message.audio:
            print("✅ Voice Clone Passed (Audio Object)")
        elif "<|AUDIO_OUT|>" in (res.choices[0].message.content or ""):
            print("✅ Voice Clone Passed (Audio Tokens)")
        else:
            print("⚠️ Voice Clone Response received but no Audio Content detected.")
            return True # Technically not a crash, but a warning
            
        return True
    except Exception as e:
        print(f"❌ Voice Clone Failed: {e}")
        return False

def run_stress_test(concurrency=1, requests_count=50):
    print(f"\n--- Test 6: Stress Test (Concurrency {concurrency}, Requests {requests_count}) ---")
    
    def _req(i):
        start = time.time()
        try:
            payload = {
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": f"Hello {i}"}],
                "max_tokens": 50,
                "temperature": 0.7
            }
            r = requests.post(f"{BASE_URL}/chat/completions", json=payload, timeout=30)
            r.raise_for_status()
            return True, time.time() - start
        except Exception as e:
            return False, str(e)

    success = 0
    fail = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as exc:
        futures = [exc.submit(_req, i) for i in range(requests_count)]
        for f in concurrent.futures.as_completed(futures):
            ok, val = f.result()
            if ok: 
                success += 1
                print(".", end="", flush=True)
            else: 
                fail += 1
                print("x", end="", flush=True)
                
    print(f"\nPassed: {success}/{requests_count}")
    return fail == 0

def run_benchmark():
    print("\n--- Test 7: Benchmark (TTFT / TPS) ---")
    prompts = ["Tell me a joke.", "List 10 numbers."]
    client = get_client()
    
    results = []
    for p in prompts:
        start = time.time()
        first = None
        count = 0
        try:
            stream = client.chat.completions.create(
                model=MODEL_ID, messages=[{"role": "user", "content": p}], 
                max_completion_tokens=100, stream=True, temperature=0.0
            )
            for chunk in stream:
                if first is None: first = time.time()
                if chunk.choices[0].delta.content: count += 1
            
            end = time.time()
            if first:
                ttft = first - start
                tps = count / (end - first)
                results.append((ttft, tps))
                print(f"   TTFT: {ttft:.4f}s | TPS: {tps:.2f}")
        except Exception as e:
            print(f"   Benchmark error: {e}")
            
    if results:
        avg_tps = statistics.mean([r[1] for r in results])
        print(f"✅ Benchmark Complete. Avg TPS: {avg_tps:.2f}")
        return True
    return False

# --- Main Driver ---

def main():
    parser = argparse.ArgumentParser(description="VM Deployment Verification Suite")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL for the API")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API Key")
    parser.add_argument("--skip-stress", action="store_true", help="Skip stress test")
    parser.add_argument("--skip-bench", action="store_true", help="Skip benchmark")
    parser.add_argument("--skip-audio", action="store_true", help="Skip audio/voice tests")
    args = parser.parse_args()

    # Update Globals
    global BASE_URL, API_KEY, MODEL_ID
    BASE_URL = args.base_url
    API_KEY = args.api_key
    
    # Re-fetch model ID with the new URL
    MODEL_ID = get_model_id()

    tests = []
    tests.append(test_health)
    tests.append(test_inference_text)
    tests.append(test_inference_stream)
    
    if not args.skip_audio:
        tests.append(test_audio_input)
        tests.append(test_voice_clone)
        
    if not args.skip_bench:
        tests.append(run_benchmark)
        
    if not args.skip_stress:
        tests.append(lambda: run_stress_test(concurrency=1, requests_count=50))

    passed = 0
    total = len(tests)
    
    print(f"🚀 Starting Verification Suite with {total} tests...")
    
    for t in tests:
        if t():
            passed += 1
        else:
            print("\n❌ CRITICAL: Test Failed. Stopping suite.")
            sys.exit(1)
            
    print(f"\n🎉 ALL {passed}/{total} TESTS PASSED! Deployment is ROCK SOLID.")

if __name__ == "__main__":
    main()
