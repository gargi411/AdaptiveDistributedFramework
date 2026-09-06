"""LLM response data model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LLMResponse:
    """Standardised response produced by an LLM provider."""

    text: str
    provider_name: str
    model_name: str
    generation_time_s: float
    finish_reason: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        """Convert response to a serialisable dictionary."""
        return {
            "text": self.text,
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "generation_time_s": self.generation_time_s,
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "metadata": self.metadata or {},
        }
