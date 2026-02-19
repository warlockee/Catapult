"""
Sample audio data for benchmarking /transcribe endpoints.

Contains a small silent WAV file (1 second, mono, 16kHz, 16-bit) for consistent load testing.
The goal is performance measurement, not transcription quality.
"""
import base64
import io
import struct
import wave


def generate_silent_wav(duration_seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    """
    Generate a silent WAV file in memory.

    Args:
        duration_seconds: Duration of the audio in seconds
        sample_rate: Sample rate in Hz (default 16kHz for speech)

    Returns:
        WAV file bytes
    """
    num_channels = 1
    sample_width = 2  # 16-bit
    num_frames = int(sample_rate * duration_seconds)

    # Create silent audio (all zeros)
    audio_data = b'\x00\x00' * num_frames

    # Write to WAV format
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)

    buffer.seek(0)
    return buffer.read()


def generate_tone_wav(
    duration_seconds: float = 1.0,
    frequency: float = 440.0,
    sample_rate: int = 16000,
    amplitude: float = 0.5,
) -> bytes:
    """
    Generate a simple sine wave tone WAV file.

    Args:
        duration_seconds: Duration of the audio
        frequency: Tone frequency in Hz (440 = A4 note)
        sample_rate: Sample rate in Hz
        amplitude: Volume (0.0 to 1.0)

    Returns:
        WAV file bytes
    """
    import math

    num_channels = 1
    sample_width = 2  # 16-bit
    num_frames = int(sample_rate * duration_seconds)
    max_amplitude = 32767  # Max for 16-bit signed

    # Generate sine wave
    audio_data = []
    for i in range(num_frames):
        t = i / sample_rate
        sample = int(amplitude * max_amplitude * math.sin(2 * math.pi * frequency * t))
        audio_data.append(struct.pack('<h', sample))

    # Write to WAV format
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b''.join(audio_data))

    buffer.seek(0)
    return buffer.read()


# Pre-generated sample audio for benchmarking
# Using a 1-second tone at 440Hz for consistent benchmarking
SAMPLE_AUDIO_WAV = generate_tone_wav(duration_seconds=1.0, frequency=440.0)

# File name to use when uploading
SAMPLE_AUDIO_FILENAME = "benchmark_sample.wav"


def get_sample_audio() -> tuple[bytes, str]:
    """
    Get sample audio data and filename for benchmarking.

    Returns:
        Tuple of (audio_bytes, filename)
    """
    return SAMPLE_AUDIO_WAV, SAMPLE_AUDIO_FILENAME
