#!/usr/bin/env python3
"""
Patch for audio model to fix Azure ML vLLM compatibility.

The issue: compute_logits returns a tuple (text_logits, audio_logits) but
Azure ML's standard vLLM sampler expects a single tensor.

This script patches the installed vllm package to fix the issue.
"""
import sys
import os

def patch_higgs_audio():
    """Patch the compute_logits method to return only text_logits for ASR."""

    # Find the audio model source file
    import vllm
    vllm_path = os.path.dirname(vllm.__file__)
    higgs_audio_path = os.path.join(vllm_path, "model_executor", "models", "higgs_audio.py")

    if not os.path.exists(higgs_audio_path):
        print(f"Error: {higgs_audio_path} not found")
        return False

    print(f"Patching {higgs_audio_path}...")

    with open(higgs_audio_path, 'r') as f:
        content = f.read()

    # Find and replace the compute_logits return statement
    # Original: return text_logits, audio_logits
    # Patched: return only text_logits when audio_logits is None (ASR mode)

    old_return = "        return text_logits, audio_logits"
    new_return = """        # Azure ML vLLM compatibility: return tensor for standard sampler
        if audio_logits is None:
            return text_logits
        return text_logits, audio_logits"""

    if old_return in content:
        content = content.replace(old_return, new_return)
        with open(higgs_audio_path, 'w') as f:
            f.write(content)
        print("Patch applied successfully!")
        return True
    elif "# Azure ML vLLM compatibility" in content:
        print("Patch already applied.")
        return True
    else:
        print("Error: Could not find the code to patch")
        print("Looking for:", repr(old_return))
        return False

if __name__ == "__main__":
    success = patch_higgs_audio()
    sys.exit(0 if success else 1)
