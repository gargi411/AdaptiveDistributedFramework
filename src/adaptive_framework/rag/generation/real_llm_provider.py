"""Real LLM Provider interfacing with production LLM APIs."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

from adaptive_framework.core.exceptions import (
    LLMGenerationError,
    LLMProviderError,
    LLMTimeoutError,
)
from adaptive_framework.rag.generation.llm_response import LLMResponse
from adaptive_framework.rag.generation.prompt_builder import BuiltPrompt
from adaptive_framework.rag.interfaces.i_llm_provider import ILLMProvider

logger = logging.getLogger(__name__)

GEMINI_DEFAULT_MODEL = "gemini-3.6-flash"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

OPENAI_DEFAULT_MODEL = "gpt-4o-mini"
OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"


class RealLLMProvider(ILLMProvider):
    """Production LLM provider supporting Gemini and OpenAI-compatible REST endpoints.

    Integrates with live models behind the ILLMProvider interface using httpx.
    Credentials are read exclusively from environment variables and never logged.
    """

    def __init__(
        self,
        backend: str = "gemini",
        model_name: str | None = None,
        api_key: str | None = None,
        api_key_env_var: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout_seconds: float = 30.0,
        base_url: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        """Initialise RealLLMProvider.

        Args:
            backend: Target LLM backend ("gemini", "openai", "real").
            model_name: Model identifier string.
            api_key: Optional explicit API key. If None, loaded from env var.
            api_key_env_var: Name of env var holding API key.
            temperature: Sampling temperature for generation.
            max_tokens: Maximum tokens in generated completion.
            timeout_seconds: HTTP request timeout in seconds.
            base_url: Custom base URL for OpenAI-compatible proxies/backends.
            client: Optional pre-configured httpx.Client for testing/reuse.
        """
        self._backend = backend.lower()
        if self._backend in ("real", "gemini"):
            self._backend = "gemini"
            default_env = "GEMINI_API_KEY"
            default_model = GEMINI_DEFAULT_MODEL
        elif self._backend in ("openai", "groq", "ollama"):
            self._backend = "openai"
            default_env = "OPENAI_API_KEY"
            default_model = OPENAI_DEFAULT_MODEL
        else:
            raise LLMProviderError(f"Unsupported LLM provider backend: '{backend}'.")

        self._api_key_env_var = api_key_env_var or default_env
        self._api_key = api_key
        self._model_name = model_name or default_model
        self._temperature = float(temperature)
        self._max_tokens = int(max_tokens)
        self._timeout_seconds = float(timeout_seconds)
        self._base_url = base_url or (
            OPENAI_DEFAULT_BASE_URL if self._backend == "openai" else None
        )

        self._client = client or httpx.Client(timeout=self._timeout_seconds)
        self._owns_client = client is None

        self._call_count = 0
        self._total_generation_time_s = 0.0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0

    def _resolve_api_key(self) -> str:
        """Resolve API key from memory or environment without logging secrets."""
        if self._api_key and self._api_key.strip():
            return self._api_key.strip()

        key = os.environ.get(self._api_key_env_var)
        if not key or not key.strip():
            raise LLMProviderError(
                f"API key missing for provider '{self._backend}'. "
                f"Please set environment variable '{self._api_key_env_var}'."
            )
        return key.strip()

    def generate(self, prompt: BuiltPrompt, **kwargs: Any) -> LLMResponse:
        """Generate an answer using the configured real LLM backend.

        Args:
            prompt: Structured BuiltPrompt with system and user messages.
            **kwargs: Generation overrides (e.g. temperature, max_tokens).

        Returns:
            LLMResponse containing response text, metrics, and token usage.
        """
        api_key = self._resolve_api_key()
        start_time = time.perf_counter()

        temp = kwargs.get("temperature", self._temperature)
        max_tok = kwargs.get("max_tokens", self._max_tokens)

        try:
            if self._backend == "gemini":
                response = self._generate_gemini(
                    prompt=prompt, api_key=api_key, temperature=temp, max_tokens=max_tok
                )
            else:
                response = self._generate_openai(
                    prompt=prompt, api_key=api_key, temperature=temp, max_tokens=max_tok
                )
        except (LLMProviderError, LLMTimeoutError, LLMGenerationError):
            raise
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"LLM generation timed out after {self._timeout_seconds}s with {self._model_name}."
            ) from exc
        except httpx.NetworkError as exc:
            raise LLMProviderError(
                f"Network failure communicating with {self._backend} provider: {type(exc).__name__}."
            ) from exc
        except Exception as exc:
            raise LLMGenerationError(
                f"Unexpected error during {self._backend} generation: {type(exc).__name__}: {exc}"
            ) from exc

        duration = time.perf_counter() - start_time
        self._call_count += 1
        self._total_generation_time_s += duration
        if response.prompt_tokens:
            self._total_prompt_tokens += response.prompt_tokens
        if response.completion_tokens:
            self._total_completion_tokens += response.completion_tokens

        return LLMResponse(
            text=response.text,
            provider_name=self.get_provider_name(),
            model_name=self.get_model_name(),
            generation_time_s=duration,
            finish_reason=response.finish_reason,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            metadata=response.metadata,
        )

    def _generate_gemini(
        self, prompt: BuiltPrompt, api_key: str, temperature: float, max_tokens: int
    ) -> LLMResponse:
        """Call Google Gemini REST API."""
        url = GEMINI_API_URL.format(model=self._model_name)
        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "system_instruction": {
                "parts": [{"text": prompt.system_message}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt.user_message}],
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        http_resp = self._client.post(url, headers=headers, json=payload)
        self._handle_http_status(http_resp)

        try:
            data = http_resp.json()
        except Exception as exc:
            raise LLMGenerationError("Malformed JSON response from Gemini API.") from exc

        candidates = data.get("candidates", [])
        if not candidates:
            feedback = data.get("promptFeedback", {})
            block_reason = feedback.get("blockReason")
            if block_reason:
                raise LLMGenerationError(
                    f"Gemini API blocked request: {block_reason}."
                )
            raise LLMGenerationError("Gemini API returned no candidates.")

        first_candidate = candidates[0]
        content = first_candidate.get("content", {})
        parts = content.get("parts", [])
        text = "".join(
            part.get("text", "")
            for part in parts
            if not part.get("thought", False)
        ).strip()

        if not text:
            raise LLMGenerationError("Gemini API returned empty completion text.")

        finish_reason = first_candidate.get("finishReason", "STOP")

        usage = data.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount")
        completion_tokens = usage.get("candidatesTokenCount")

        return LLMResponse(
            text=text,
            provider_name=self.get_provider_name(),
            model_name=self.get_model_name(),
            generation_time_s=0.0,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            metadata={"safety_ratings": first_candidate.get("safetyRatings", [])},
        )

    def _generate_openai(
        self, prompt: BuiltPrompt, api_key: str, temperature: float, max_tokens: int
    ) -> LLMResponse:
        """Call OpenAI-compatible REST API."""
        base_url = (self._base_url or OPENAI_DEFAULT_BASE_URL).rstrip("/")
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": prompt.system_message},
                {"role": "user", "content": prompt.user_message},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        http_resp = self._client.post(url, headers=headers, json=payload)
        self._handle_http_status(http_resp)

        try:
            data = http_resp.json()
        except Exception as exc:
            raise LLMGenerationError("Malformed JSON response from OpenAI API.") from exc

        choices = data.get("choices", [])
        if not choices:
            raise LLMGenerationError("OpenAI API returned no completion choices.")

        first_choice = choices[0]
        message = first_choice.get("message", {})
        text = message.get("content", "").strip()

        if not text:
            raise LLMGenerationError("OpenAI API returned empty completion text.")

        finish_reason = first_choice.get("finish_reason", "stop")

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

        return LLMResponse(
            text=text,
            provider_name=self.get_provider_name(),
            model_name=self.get_model_name(),
            generation_time_s=0.0,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            metadata={"system_fingerprint": data.get("system_fingerprint")},
        )

    def _handle_http_status(self, response: httpx.Response) -> None:
        """Inspect HTTP status code and raise appropriate framework exceptions."""
        if response.status_code == 200:
            return

        status = response.status_code
        if status in (401, 403):
            raise LLMProviderError(
                f"Authentication failed for {self._backend} (HTTP {status}). "
                "Please verify the configured API key."
            )
        if status == 429:
            raise LLMProviderError(
                f"Rate limit exceeded for {self._backend} (HTTP 429). "
                "Please back off and retry later."
            )
        if status >= 500:
            raise LLMGenerationError(
                f"Provider server error from {self._backend} (HTTP {status})."
            )

        raise LLMGenerationError(
            f"Provider request failed with HTTP {status}: {response.text[:200]}"
        )

    def get_provider_name(self) -> str:
        """Return provider identifier string."""
        return self._backend

    def get_model_name(self) -> str:
        """Return active model identifier."""
        return self._model_name

    def get_metrics(self) -> dict[str, Any]:
        """Return telemetry metrics for generation requests."""
        return {
            "call_count": self._call_count,
            "total_generation_time_s": self._total_generation_time_s,
            "average_generation_time_s": (
                self._total_generation_time_s / self._call_count
                if self._call_count > 0
                else 0.0
            ),
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
        }

    def shutdown(self) -> None:
        """Close HTTP client and clean up connections."""
        if self._owns_client and self._client:
            self._client.close()
