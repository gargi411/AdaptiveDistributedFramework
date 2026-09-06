"""Factory for instantiating LLM providers based on configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from adaptive_framework.core.exceptions import LLMProviderError
from adaptive_framework.rag.generation.fake_llm_provider import FakeLLMProvider
from adaptive_framework.rag.generation.real_llm_provider import RealLLMProvider
from adaptive_framework.rag.interfaces.i_llm_provider import ILLMProvider

if TYPE_CHECKING:
    from adaptive_framework.config.models import GenerationConfig


def create_llm_provider(config: GenerationConfig) -> ILLMProvider:
    """Create and return an ILLMProvider instance matching the given configuration.

    Args:
        config: GenerationConfig containing provider name, model, parameters, etc.

    Returns:
        Concrete ILLMProvider instance (FakeLLMProvider or RealLLMProvider).

    Raises:
        LLMProviderError: If the specified provider is unrecognized.
    """
    provider_type = config.provider.strip().lower()

    if provider_type == "fake":
        return FakeLLMProvider(model_name=config.model)

    if provider_type in ("real", "gemini", "openai", "groq", "ollama"):
        return RealLLMProvider(
            backend=provider_type,
            model_name=config.model,
            api_key_env_var=config.api_key_env_var,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout_seconds=config.timeout_seconds,
            base_url=config.base_url,
        )

    raise LLMProviderError(f"Unsupported LLM provider type: '{config.provider}'.")
