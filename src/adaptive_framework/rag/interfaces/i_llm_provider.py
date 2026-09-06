"""Abstract interface for LLM generation providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from adaptive_framework.rag.generation.llm_response import LLMResponse
    from adaptive_framework.rag.generation.prompt_builder import BuiltPrompt


class ILLMProvider(ABC):
    """Abstract interface defining standard LLM provider operations."""

    @abstractmethod
    def generate(self, prompt: BuiltPrompt, **kwargs: Any) -> LLMResponse:
        """Generate a completion from a structured prompt.

        Args:
            prompt: The BuiltPrompt to submit.
            **kwargs: Provider-specific inference parameters (e.g. temperature).

        Returns:
            LLMResponse containing generation text and execution metadata.
        """

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the identifier for the provider implementation."""

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model identifier currently in use."""

    @abstractmethod
    def get_metrics(self) -> dict[str, Any]:
        """Return telemetry metrics (e.g., call count, tokens, latencies)."""

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up resources, open sessions, or thread pools."""
