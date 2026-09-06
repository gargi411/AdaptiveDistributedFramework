"""Deterministic Fake LLM Provider for testing and offline RAG verification."""

from __future__ import annotations

import re
import time
from typing import Any

from adaptive_framework.rag.generation.llm_response import LLMResponse
from adaptive_framework.rag.generation.prompt_builder import BuiltPrompt
from adaptive_framework.rag.interfaces.i_llm_provider import ILLMProvider


class FakeLLMProvider(ILLMProvider):
    """Deterministic, offline LLM provider simulating grounded RAG responses.

    Inspects evidence present within the prompt user message and produces
    predictable, structured answers citing discovered sources.
    """

    def __init__(
        self,
        model_name: str = "fake-llm-v1",
        fixed_latency_s: float = 0.001,
        force_error: bool = False,
    ) -> None:
        """Initialise FakeLLMProvider.

        Args:
            model_name: Model identifier string.
            fixed_latency_s: Synthetic delay to simulate network/generation latency.
            force_error: If True, raises RuntimeError during generate calls.
        """
        self._model_name = model_name
        self._fixed_latency_s = fixed_latency_s
        self._force_error = force_error
        self._call_count = 0
        self._total_generation_time_s = 0.0

    def generate(self, prompt: BuiltPrompt, **kwargs: Any) -> LLMResponse:
        """Generate a deterministic answer based on the prompt's evidence.

        Args:
            prompt: BuiltPrompt instance.
            **kwargs: Extra arguments.

        Returns:
            LLMResponse containing synthesized grounded answer.
        """
        if self._force_error:
            raise RuntimeError("FakeLLMProvider configured to force error.")

        start_time = time.perf_counter()
        if self._fixed_latency_s > 0:
            time.sleep(self._fixed_latency_s)

        self._call_count += 1

        if not prompt.has_evidence:
            text = "No relevant documents were found in the knowledge base to answer the question."
        else:
            # Parse evidence blocks from prompt user message
            # Format in PromptBuilder: [Source: {doc_id} | File: {file} | Pages: {pages} | Rank: {rank}]
            source_matches = re.findall(
                r"\[Source:\s*([^\|]+)\s*\|\s*File:\s*([^\|]+)\s*\|\s*Pages:\s*([^\|]+?)(?:\s*\|\s*Section:[^\|]+)?\s*\|\s*Rank:\s*(\d+)\]",
                prompt.user_message,
            )

            citations: list[str] = []
            for match in source_matches:
                doc_id = match[0].strip()
                pages = match[2].strip()
                citations.append(f"({doc_id}, p. {pages})")

            citations_str = ", ".join(citations) if citations else "(unknown source)"

            # Extract snippet from evidence
            evidence_match = re.search(
                r"--- BEGIN UNTRUSTED RETRIEVED EVIDENCE ---\s*\n.*?\n(.*?)--- END UNTRUSTED RETRIEVED EVIDENCE ---",
                prompt.user_message,
                re.DOTALL,
            )
            snippet = ""
            evidence_text = ""
            if evidence_match:
                evidence_text = evidence_match.group(1).lower()
                raw_lines = [
                    line.strip()
                    for line in evidence_match.group(1).splitlines()
                    if line.strip() and not line.startswith("[Source:")
                ]
                if raw_lines:
                    snippet = raw_lines[0][:150]

            # Check if key content words in user question are present in evidence
            query_str = getattr(prompt, "query", "") or ""
            if not query_str:
                q_match = re.search(r"User Question:\s*(.+?)(?:\n|$)", prompt.user_message)
                if q_match:
                    query_str = q_match.group(1)

            is_unsupported = False
            if query_str:
                q_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", query_str.lower()))
                stopwords = {
                    "what", "which", "where", "when", "does", "were", "patient", "clinical",
                    "documented", "undergo", "have", "with", "this", "that", "from", "report",
                }
                content_words = q_words - stopwords
                if content_words and not any(w in evidence_text for w in content_words):
                    is_unsupported = True

            if is_unsupported:
                text = (
                    "Insufficient clinical evidence found in the retrieved documentation to answer this question. "
                    "The requested topic is not supported by the medical records on file."
                )
            else:
                text = (
                    f"Based on the retrieved evidence: {snippet}... "
                    f"Sources: {citations_str}."
                )

        duration = time.perf_counter() - start_time
        self._total_generation_time_s += duration

        return LLMResponse(
            text=text,
            provider_name=self.get_provider_name(),
            model_name=self.get_model_name(),
            generation_time_s=duration,
            finish_reason="stop",
            prompt_tokens=len(prompt.full_prompt_text.split()),
            completion_tokens=len(text.split()),
        )

    def get_provider_name(self) -> str:
        """Return provider name."""
        return "fake"

    def get_model_name(self) -> str:
        """Return model name."""
        return self._model_name

    def get_metrics(self) -> dict[str, Any]:
        """Return provider telemetry metrics."""
        return {
            "call_count": self._call_count,
            "total_generation_time_s": self._total_generation_time_s,
            "average_generation_time_s": (
                self._total_generation_time_s / self._call_count
                if self._call_count > 0
                else 0.0
            ),
        }

    def shutdown(self) -> None:
        """Shutdown provider (no-op for fake)."""
        pass
