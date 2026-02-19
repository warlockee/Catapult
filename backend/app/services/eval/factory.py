"""
Factory for creating evaluator instances based on model type.

## How it works

Evaluators register themselves using the `@register_evaluator('type')` decorator.
The factory looks up evaluators by type and creates instances.

## Type Aliases

Some model/server types map to the same evaluator:
- 'asr', 'audio', 'audio-understanding' -> 'asr' evaluator
- 'llm', 'text', 'vllm' -> 'llm' evaluator
- 'vision', 'multimodal' -> 'vision' evaluator
"""
import logging
from typing import Dict, Optional, Type

from app.services.eval.base import Evaluator

logger = logging.getLogger(__name__)

# Registry of evaluator classes
_evaluator_registry: Dict[str, Type[Evaluator]] = {}

# Type aliases - maps alternative names to canonical types
_type_aliases: Dict[str, str] = {
    # ASR aliases
    'audio': 'asr',
    'audio-understanding': 'asr',
    'asr-vllm': 'asr',
    'asr-allinone': 'asr',
    'asr-azure-allinone': 'asr',
    # LLM aliases
    'text': 'llm',
    'vllm': 'llm',
    # Vision aliases
    'multimodal': 'vision',
}


def register_evaluator(evaluation_type: str):
    """
    Decorator to register an evaluator class.

    Usage:
        @register_evaluator('asr')
        class ASRDockerEvaluator(Evaluator):
            ...
    """
    def decorator(cls: Type[Evaluator]):
        _evaluator_registry[evaluation_type.lower()] = cls
        logger.info(f"Registered evaluator '{cls.__name__}' for type '{evaluation_type}'")
        return cls
    return decorator


def add_type_alias(alias: str, canonical_type: str):
    """Add a type alias mapping."""
    _type_aliases[alias.lower()] = canonical_type.lower()


class EvaluatorFactory:
    """Factory for creating evaluator instances."""

    @staticmethod
    def create(
        model_type: str,
        server_type: Optional[str] = None,
    ) -> Optional[Evaluator]:
        """
        Create an evaluator instance based on model type.

        Args:
            model_type: Type of model ('asr', 'llm', 'vision', etc.)
            server_type: Optional server type (used as fallback or for aliases)

        Returns:
            Evaluator instance or None if no evaluator matches
        """
        # Normalize inputs
        model_type_lower = (model_type or "").lower()
        server_type_lower = (server_type or "").lower()

        # Resolve type (check server_type first, then model_type)
        eval_type = None

        # Try server_type first (more specific)
        if server_type_lower:
            if server_type_lower in _evaluator_registry:
                eval_type = server_type_lower
            elif server_type_lower in _type_aliases:
                eval_type = _type_aliases[server_type_lower]

        # Then try model_type
        if eval_type is None and model_type_lower:
            if model_type_lower in _evaluator_registry:
                eval_type = model_type_lower
            elif model_type_lower in _type_aliases:
                eval_type = _type_aliases[model_type_lower]

        # Create evaluator if found
        if eval_type and eval_type in _evaluator_registry:
            evaluator_class = _evaluator_registry[eval_type]
            logger.info(f"Creating evaluator: {evaluator_class.__name__} for type={eval_type}")
            return evaluator_class()

        logger.debug(f"No evaluator found for model_type={model_type}, server_type={server_type}")
        return None

    @staticmethod
    def get_available_types() -> list[str]:
        """Return list of available evaluation types."""
        return list(_evaluator_registry.keys())

    @staticmethod
    def get_all_supported_types() -> list[str]:
        """Return all supported types including aliases."""
        return list(set(list(_evaluator_registry.keys()) + list(_type_aliases.keys())))

    @staticmethod
    def supports_evaluation(model_type: str, server_type: Optional[str] = None) -> bool:
        """Check if evaluation is supported for the given model/server type."""
        return EvaluatorFactory.create(model_type, server_type) is not None
