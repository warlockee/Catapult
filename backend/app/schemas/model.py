"""
Pydantic schemas for Model.
"""
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field

# Server types for model deployment
# These must stay in sync with SERVER_TYPE_TEMPLATES in dockerfile_service.py
ServerType = Literal[
    "vllm",                # LLMs - uses vLLM OpenAI-compatible server
    "audio",               # Generic audio models - uses FastAPI audio server
    "audio-understanding", # Audio understanding models
    "audio-generation",    # Audio generation models
    "asr-vllm",            # ASR with vLLM backend (audio models) - raw audio API
    "asr-allinone",        # ASR all-in-one (VAD + segmentation + vLLM) - simple file upload API
    "whisper",             # Whisper ASR models - uses dedicated whisper template
    "tts",                 # Text-to-speech models - uses dedicated TTS template
    "stt",                 # Speech-to-text models - uses audio server
    "codec",               # Audio codec models - uses dedicated codec template
    "embedding",           # Embedding models - uses dedicated embedding template
    "multimodal",          # Vision-language models (LLaVA, Qwen-VL) - uses multimodal template
    "onnx",                # ONNX runtime models
    "triton",              # Triton inference server models
    "generic",             # Generic FastAPI server (safe fallback)
    "custom",              # Custom Dockerfile required
]


# Source types for model discovery
SourceType = Literal["filesystem", "manual", "orphaned"]


class ModelBase(BaseModel):
    """Base schema for Model."""
    name: str = Field(..., max_length=255)
    storage_path: str = Field(..., max_length=1000)
    repository: Optional[str] = Field(None, max_length=500)
    company: Optional[str] = Field(None, max_length=255)
    base_model: Optional[str] = Field(None, max_length=100)
    parameter_count: Optional[str] = Field(None, max_length=50)
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    requires_gpu: bool = Field(default=True, description="Whether this model requires GPU for deployment")
    server_type: Optional[ServerType] = Field(
        None,
        description="Server type for deployment: vllm (LLMs), audio (TTS/STT), onnx, triton, generic, custom. Auto-detected if not specified."
    )
    source: SourceType = Field(
        default="filesystem",
        description="Source of model discovery: filesystem (local), manual, orphaned"
    )


class ModelCreate(ModelBase):
    """Schema for creating a Model."""
    pass


class ModelUpdate(BaseModel):
    """Schema for updating a Model."""
    repository: Optional[str] = Field(None, max_length=500)
    company: Optional[str] = Field(None, max_length=255)
    base_model: Optional[str] = Field(None, max_length=100)
    parameter_count: Optional[str] = Field(None, max_length=50)
    storage_path: Optional[str] = Field(None, max_length=1000)
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    requires_gpu: Optional[bool] = None
    server_type: Optional[ServerType] = None


class ModelResponse(ModelBase):
    """Schema for Model response."""
    id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

    @classmethod
    def from_model(cls, model) -> "ModelResponse":
        """Convert a Model ORM instance to response schema."""
        return cls(
            id=model.id,
            name=model.name,
            storage_path=model.storage_path,
            repository=model.repository,
            company=model.company,
            base_model=model.base_model,
            parameter_count=model.parameter_count,
            description=model.description,
            tags=model.tags or [],
            metadata=model.meta_data or {},
            requires_gpu=model.requires_gpu,
            server_type=model.server_type,
            source=model.source,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )


class ModelWithVersions(ModelResponse):
    """Schema for Model with version count."""
    version_count: int = 0

    @classmethod
    def from_model_with_count(cls, model, version_count: int) -> "ModelWithVersions":
        """Convert a Model ORM instance with version count to response schema."""
        return cls(
            id=model.id,
            name=model.name,
            storage_path=model.storage_path,
            repository=model.repository,
            company=model.company,
            base_model=model.base_model,
            parameter_count=model.parameter_count,
            description=model.description,
            tags=model.tags or [],
            metadata=model.meta_data or {},
            requires_gpu=model.requires_gpu,
            server_type=model.server_type,
            source=model.source,
            created_at=model.created_at,
            updated_at=model.updated_at,
            version_count=version_count,
        )


# Backward compatibility alias (deprecated)
ModelWithReleases = ModelWithVersions


class ModelOption(BaseModel):
    """Slim schema for dropdown/select options - reduces data transfer."""
    id: UUID
    name: str

    class Config:
        from_attributes = True
