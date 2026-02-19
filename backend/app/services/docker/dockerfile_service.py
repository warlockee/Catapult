"""
Service for Dockerfile template resolution and generation.

Handles:
- Template selection based on build type and server type
- Template variable substitution
- Custom Dockerfile content
- Auto-detection of model server type
"""
import os
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

from app.core.config import settings
from app.core.exceptions import TemplateNotFoundError

logger = logging.getLogger(__name__)


# Server type to Dockerfile template mapping
# Templates should exist in kb/dockers/ - resolve_template_path handles fallbacks
SERVER_TYPE_TEMPLATES = {
    "vllm": "Dockerfile",              # LLMs - vLLM OpenAI-compatible server
    "audio": "Dockerfile.audio",       # Generic audio models
    "audio-understanding": "Dockerfile.audio",  # Audio understanding
    "audio-generation": "Dockerfile.audio",     # Audio generation/TTS
    "asr-vllm": "Dockerfile.asr_vllm", # ASR with vLLM backend (audio) - raw audio API
    "asr-allinone": "Dockerfile.asr_allinone",  # ASR all-in-one (VAD + segmentation + vLLM)
    "asr-azure-allinone": "Dockerfile.asr_azure_allinone",  # ASR Azure all-in-one (Azure ML base)
    "whisper": "Dockerfile.whisper",   # Whisper/ASR models (dedicated template)
    "tts": "Dockerfile.tts",           # Text-to-speech models (dedicated template)
    "stt": "Dockerfile.audio",         # Speech-to-text models
    "codec": "Dockerfile.codec",       # Audio codec models (dedicated template)
    "embedding": "Dockerfile.embedding",  # Embedding models (dedicated template)
    "multimodal": "Dockerfile.multimodal",  # Vision-language models (LLaVA, Qwen-VL)
    "onnx": "Dockerfile.onnx",         # ONNX runtime
    "triton": "Dockerfile.triton",     # Triton inference server
    "generic": "Dockerfile.generic",   # Generic FastAPI server (safe fallback)
    "custom": None,                    # Requires custom Dockerfile
}


def detect_server_type_from_files(model_path: str) -> Tuple[str, str]:
    """
    Auto-detect server type by scanning model files.

    This function uses multiple heuristics to determine the appropriate server type:
    1. HuggingFace config.json model_type and architectures
    2. File patterns (audio files, model weights, etc.)
    3. Directory structure (Triton, custom servers)

    Args:
        model_path: Path to model directory

    Returns:
        Tuple of (server_type, reason)
    """
    if not os.path.isdir(model_path):
        return "generic", "default (path not found)"

    files = set()
    for root, dirs, filenames in os.walk(model_path):
        for f in filenames:
            files.add(f.lower())
        # Only scan top 2 levels
        if root.count(os.sep) - model_path.count(os.sep) >= 2:
            break

    # Check for HuggingFace transformer model - determine if vLLM compatible or audio
    if "config.json" in files:
        # Read config.json to check architectures and model_type
        config_path = os.path.join(model_path, "config.json")
        try:
            import json
            with open(config_path) as f:
                config = json.load(f)
            architectures = config.get("architectures", [])
            model_type = config.get("model_type", "").lower()

            # Audio ASR models - use vLLM backend (Architecture B)
            # These models support vLLM's OpenAI-compatible API with audio input
            asr_audio_types = {"higgs_audio_3", "higgs_audio", "higgs_audio_2"}
            if model_type in asr_audio_types:
                return "asr-vllm", f"detected audio model_type: {model_type}"
            if any("HiggsAudio" in arch for arch in architectures):
                return "asr-vllm", f"detected audio architecture: {architectures}"

            # Whisper/ASR models - return specific "whisper" type
            whisper_model_types = {"whisper"}
            if model_type in whisper_model_types:
                return "whisper", f"detected whisper model_type: {model_type}"
            if any("Whisper" in arch for arch in architectures):
                return "whisper", f"detected whisper architecture: {architectures}"

            # TTS models - return specific "tts" type
            tts_model_types = {
                "speecht5", "vits", "fastspeech2_conformer", "univnet", "bark",
            }
            if model_type in tts_model_types:
                return "tts", f"detected TTS model_type: {model_type}"
            if any(any(p in arch for p in ["SpeechT5", "Vits", "Bark"]) for arch in architectures):
                return "tts", f"detected TTS architecture: {architectures}"

            # Codec models - return specific "codec" type
            codec_model_types = {"encodec", "dac"}
            if model_type in codec_model_types:
                return "codec", f"detected codec model_type: {model_type}"
            if any("EnCodec" in arch or "DAC" in arch for arch in architectures):
                return "codec", f"detected codec architecture: {architectures}"

            # Other ASR/STT models - return "stt" type
            stt_model_types = {
                "wav2vec2", "wav2vec2-bert", "wav2vec2-conformer",
                "hubert", "wavlm", "sew", "sew-d", "unispeech", "unispeech-sat",
                "speech_to_text", "speech_to_text_2", "speech2text", "speech2text2",
                "mctct", "mms",
            }
            stt_arch_patterns = [
                "Wav2Vec", "Hubert", "WavLM", "SEW", "UniSpeech",
                "Speech2Text", "SpeechToText",
            ]
            if model_type in stt_model_types:
                return "stt", f"detected STT model_type: {model_type}"
            if any(any(p in arch for p in stt_arch_patterns) for arch in architectures):
                return "stt", f"detected STT architecture: {architectures}"

            # Generic audio models
            audio_model_types = {
                "audio-spectrogram-transformer", "clap", "pop2piano", "musicgen",
                "musicgen_melody", "seamless_m4t", "seamless_m4t_v2",
                "step_audio", "step_audio_2",  # Step-Audio models
            }
            audio_arch_patterns = ["Audio", "Clap", "MusicGen", "SeamlessM4T", "StepAudio"]
            if model_type in audio_model_types:
                return "audio", f"detected audio model_type: {model_type}"
            if any(any(p in arch for p in audio_arch_patterns) for arch in architectures):
                return "audio", f"detected audio architecture: {architectures}"

            # Multimodal/Vision-Language models (LLaVA, Qwen-VL, etc.)
            multimodal_model_types = {
                "llava", "llava_next", "llava-next", "qwen2_vl", "qwen-vl",
                "blip-2", "blip2", "instructblip", "kosmos-2", "kosmos2",
                "paligemma", "idefics", "idefics2", "fuyu", "git",
            }
            multimodal_arch_patterns = [
                "LlavaForConditionalGeneration", "Qwen2VL", "QwenVL",
                "Blip2", "InstructBlip", "Kosmos", "PaliGemma", "Idefics", "Fuyu",
            ]
            if model_type in multimodal_model_types:
                return "multimodal", f"detected multimodal model_type: {model_type}"
            if any(any(p in arch for p in multimodal_arch_patterns) for arch in architectures):
                return "multimodal", f"detected multimodal architecture: {architectures}"

            # Diffusion models (Stable Diffusion, SDXL, Flux) - need custom serving
            diffusion_model_types = {
                "stable-diffusion", "stable_diffusion", "sdxl", "sd3",
                "flux", "kandinsky", "dit", "pixart",
            }
            if model_type in diffusion_model_types:
                return "generic", f"detected diffusion model_type: {model_type}"

            # Pure vision models (not VLM) - use generic
            vision_model_types = {
                "vit", "deit", "beit", "swin", "convnext", "clip", "blip",
                "owlvit", "detr", "yolos", "segformer", "mask2former",
            }
            if model_type in vision_model_types:
                return "generic", f"detected vision model_type: {model_type}"

            # Embedding models - check before LLM detection
            # These often share architecture with LLMs but need different serving
            embedding_indicators = {
                "sentence_transformers_config.json",  # sentence-transformers
                "modules.json",                       # sentence-transformers modules
                "pooling_config.json",                # pooling layer config
                "1_pooling",                          # common pooling folder name
            }
            if any(ind in files for ind in embedding_indicators):
                return "embedding", "detected embedding model files"

            # Check model name hints for embedding (before architecture check)
            model_name_lower = os.path.basename(model_path).lower()
            if "embed" in model_name_lower or "gte" in model_name_lower or "bge" in model_name_lower:
                return "embedding", f"detected embedding model from name: {model_name_lower}"

            # LLM architectures that vLLM supports (pattern-based for future compatibility)
            # vLLM supports most *ForCausalLM models
            llm_arch_patterns = [
                "ForCausalLM",  # Generic causal LM
                "LMHeadModel",  # Alternative naming
            ]

            # Check architectures for LLM patterns
            for arch in architectures:
                for pattern in llm_arch_patterns:
                    if pattern in arch:
                        return "vllm", f"detected LLM architecture: {arch}"

            # Encoder-only models (BERT, RoBERTa, etc.) - use generic
            encoder_patterns = ["BertModel", "RobertaModel", "ElectraModel", "DebertaModel"]
            if any(any(p in arch for p in encoder_patterns) for arch in architectures):
                return "generic", f"detected encoder-only architecture: {architectures}"

            # For other HuggingFace models with config.json but unknown architecture, use generic
            return "generic", f"unknown HuggingFace architecture: {architectures or model_type or 'none'}"
        except Exception as e:
            logger.debug(f"Failed to read config.json: {e}")

    # Check for audio model patterns
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg"}
    audio_keywords = {"audio", "speech", "tts", "stt", "voice", "sound", "wav2vec", "whisper", "podcast", "fiction"}

    has_audio_files = any(any(f.endswith(ext) for ext in audio_extensions) for f in files)
    has_audio_keywords = any(any(kw in f for kw in audio_keywords) for f in files)

    # Check for speech tokenizer or audio-specific model files
    audio_model_files = {"speech_tokenizer", "vocoder", "hifi", "hift", "flow.pt", "campplus", "mel", "acoustic"}
    has_audio_model = any(any(amf in f for amf in audio_model_files) for f in files)

    # Check for RecordIO format (common for audio/custom models)
    has_recordio = any(f.endswith(".rec") or f.endswith(".idx") for f in files)

    if has_audio_model or (has_audio_files and has_audio_keywords) or has_recordio:
        return "audio", "detected audio model files"

    # Check for diffusers models (model_index.json is diffusers-specific)
    if "model_index.json" in files:
        return "generic", "detected diffusers model (model_index.json)"

    # Check for ONNX models (without config.json = not HuggingFace)
    onnx_files = [f for f in files if f.endswith(".onnx")]
    if onnx_files and "config.json" not in files:
        return "onnx", f"detected ONNX files: {onnx_files[:3]}"

    # Check for Triton model repository structure
    if "config.pbtxt" in files:
        return "triton", "detected Triton config.pbtxt"

    # Check for model's own Dockerfile - these need custom handling
    if "dockerfile" in files:
        return "custom", "model has its own Dockerfile"

    # Check for custom entrypoint
    if "server.py" in files or "entrypoint.py" in files or "app.py" in files:
        return "generic", "detected custom server script"

    # Check for PyTorch models without config.json (not vLLM compatible)
    pytorch_files = [f for f in files if f.endswith(".pt") or f.endswith(".pth") or f.endswith(".bin")]
    if pytorch_files and "config.json" not in files:
        return "generic", f"detected PyTorch files without config.json: {pytorch_files[:3]}"

    # Check for safetensors without config.json
    safetensor_files = [f for f in files if f.endswith(".safetensors")]
    if safetensor_files and "config.json" not in files:
        return "generic", f"detected safetensors without config.json: {safetensor_files[:3]}"

    # Default to generic for unknown models (safer than vllm which requires config.json)
    return "generic", "default (no specific patterns detected, using generic server)"


@dataclass
class DockerfileConfig:
    """Configuration for Dockerfile generation."""
    content: Optional[str] = None  # Custom Dockerfile content
    build_type: str = "standard"  # azure, test, optimized, standard
    model_name: Optional[str] = None  # For template substitution
    server_type: Optional[str] = None  # vllm, audio, onnx, triton, generic, custom
    model_path: Optional[str] = None  # Path to model for auto-detection


class DockerfileService:
    """
    Service for Dockerfile template resolution and generation.

    Responsibilities:
    - Resolve template path based on build type
    - Apply template variable substitution
    - Write Dockerfile to build directory
    """

    def __init__(self, templates_path: Optional[str] = None):
        """
        Initialize DockerfileService.

        Args:
            templates_path: Path to Docker templates directory
        """
        self.templates_path = templates_path or settings.DOCKER_TEMPLATES_PATH

    def get_template_name(self, build_type: str, server_type: Optional[str] = None) -> str:
        """
        Get template filename for build type and server type.

        Args:
            build_type: Type of build (azure, test, optimized, standard)
            server_type: Server type (vllm, audio, onnx, triton, generic, custom)

        Returns:
            Template filename
        """
        # Build type takes precedence for special builds
        build_type_map = {
            "azure": "Dockerfile.maap",
            "test": "Dockerfile.test",
            "optimized": "Dockerfile.optimized",
            "asr-vllm": "Dockerfile.asr_vllm",  # ASR with vLLM backend
            "asr-allinone": "Dockerfile.asr_allinone",  # ASR all-in-one
            "asr-azure-allinone": "Dockerfile.asr_azure_allinone",  # ASR Azure all-in-one
        }

        if build_type in build_type_map:
            return build_type_map[build_type]

        # Use server type template if specified
        if server_type and server_type in SERVER_TYPE_TEMPLATES:
            template = SERVER_TYPE_TEMPLATES[server_type]
            if template:
                return template

        # Unknown server_type - use generic (safer than vLLM which requires specific format)
        if server_type:
            logger.warning(f"Unknown server_type '{server_type}', falling back to generic template")
            return "Dockerfile.generic"

        # No server_type specified - use generic as safe default
        return "Dockerfile.generic"

    def resolve_template_path(self, build_type: str, server_type: Optional[str] = None) -> str:
        """
        Resolve full path to template file.

        Args:
            build_type: Type of build
            server_type: Server type for model deployment

        Returns:
            Full path to template file

        Raises:
            TemplateNotFoundError: If template doesn't exist
        """
        template_name = self.get_template_name(build_type, server_type)

        # Try primary templates path
        template_path = os.path.join(self.templates_path, template_name)
        if os.path.exists(template_path):
            return template_path

        # Fallback to kb/dockers
        fallback_path = os.path.join("kb/dockers", template_name)
        if os.path.exists(fallback_path):
            return fallback_path

        # If server_type template not found, fall back to generic (safer than vLLM)
        if server_type and template_name not in ("Dockerfile", "Dockerfile.generic"):
            logger.warning(f"Template {template_name} not found, falling back to Dockerfile.generic")
            # Try generic in primary path
            generic_path = os.path.join(self.templates_path, "Dockerfile.generic")
            if os.path.exists(generic_path):
                return generic_path
            # Try generic in kb/dockers
            generic_fallback = os.path.join("kb/dockers", "Dockerfile.generic")
            if os.path.exists(generic_fallback):
                return generic_fallback

        raise TemplateNotFoundError(template_name)

    def render_template(self, template_path: str, variables: dict) -> str:
        """
        Render template with variable substitution.

        Args:
            template_path: Path to template file
            variables: Dict of variable names to values

        Returns:
            Rendered template content
        """
        with open(template_path, "r") as f:
            content = f.read()

        for key, value in variables.items():
            placeholder = "{{" + key + "}}"
            content = content.replace(placeholder, str(value))

        return content

    def generate_dockerfile(
        self,
        build_dir: str,
        config: DockerfileConfig,
    ) -> Tuple[str, Optional[str]]:
        """
        Generate Dockerfile in build directory.

        Args:
            build_dir: Build directory path
            config: Dockerfile configuration

        Returns:
            Tuple of (path to generated Dockerfile, detected server_type or None)
        """
        dockerfile_path = os.path.join(build_dir, "Dockerfile")
        detected_server_type = None

        if config.content:
            # Use provided custom content
            with open(dockerfile_path, "w") as f:
                f.write(config.content)
            logger.info(f"Wrote custom Dockerfile to {dockerfile_path}")
        else:
            # Determine server_type: use explicit, or auto-detect
            server_type = config.server_type

            if not server_type and config.model_path:
                server_type, reason = detect_server_type_from_files(config.model_path)
                detected_server_type = server_type
                logger.info(f"Auto-detected server_type '{server_type}' for model: {reason}")

            # Use template based on build_type and server_type
            template_path = self.resolve_template_path(config.build_type, server_type)

            if config.build_type == "optimized" and config.model_name:
                # Render template with substitution
                content = self.render_template(
                    template_path,
                    {"MODEL_NAME": config.model_name}
                )
                with open(dockerfile_path, "w") as f:
                    f.write(content)
            else:
                # Copy template directly
                import shutil
                shutil.copy(template_path, dockerfile_path)

            logger.info(f"Created Dockerfile from template {template_path} (server_type={server_type})")

        return dockerfile_path, detected_server_type


# Singleton instance
dockerfile_service = DockerfileService()
