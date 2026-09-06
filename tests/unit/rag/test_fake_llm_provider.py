"""Unit tests for FakeLLMProvider (Phase 4.5)."""

from __future__ import annotations

import pytest

from adaptive_framework.rag.generation.context_builder import (
    BuiltContext,
    EvidenceBlock,
)
from adaptive_framework.rag.generation.fake_llm_provider import FakeLLMProvider
from adaptive_framework.rag.generation.llm_response import LLMResponse
from adaptive_framework.rag.generation.prompt_builder import (
    BuiltPrompt,
    PromptBuilder,
)
from adaptive_framework.rag.retrieval.retrieval_result import RetrievalResult


def _build_test_prompt(has_evidence: bool = True) -> BuiltPrompt:
    if not has_evidence:
        context = BuiltContext(
            evidence_blocks=(),
            total_chars=0,
            truncated=False,
            retained_results=(),
        )
    else:
        block = EvidenceBlock(
            rank=1,
            document_id="doc_heart_01",
            source_file="cardio.pdf",
            page_numbers=(3,),
            chunk_id="chk_c01",
            score=0.94,
            section_heading="Cardiology Findings",
            text="Patient exhibits resting heart rate of 72 bpm with normal sinus rhythm.",
        )
        result = RetrievalResult(
            chunk_id="chk_c01",
            document_id="doc_heart_01",
            source_file="cardio.pdf",
            page_numbers=(3,),
            text="Patient exhibits resting heart rate of 72 bpm with normal sinus rhythm.",
            section_heading="Cardiology Findings",
            document_type="report",
            chunk_index=0,
            score=0.94,
            rank=1,
        )
        context = BuiltContext(
            evidence_blocks=(block,),
            total_chars=len(block.text),
            truncated=False,
            retained_results=(result,),
        )
    return PromptBuilder().build(context, "What was the resting heart rate?")


class TestFakeLLMProvider:
    """Test suite for FakeLLMProvider."""

    def test_generate_returns_llm_response(self) -> None:
        provider = FakeLLMProvider(model_name="fake-v1", fixed_latency_s=0.0)
        prompt = _build_test_prompt(has_evidence=True)
        response = provider.generate(prompt)

        assert isinstance(response, LLMResponse)
        assert response.provider_name == "fake"
        assert response.model_name == "fake-v1"
        assert response.finish_reason == "stop"
        assert response.generation_time_s >= 0.0
        assert "72 bpm" in response.text
        assert "doc_heart_01" in response.text
        assert "p. 3" in response.text

    def test_deterministic_same_input_same_output(self) -> None:
        provider = FakeLLMProvider(model_name="fake-v1", fixed_latency_s=0.0)
        prompt = _build_test_prompt(has_evidence=True)

        resp1 = provider.generate(prompt)
        resp2 = provider.generate(prompt)

        assert resp1.text == resp2.text

    def test_no_evidence_response(self) -> None:
        provider = FakeLLMProvider()
        prompt = _build_test_prompt(has_evidence=False)
        response = provider.generate(prompt)

        assert "No relevant documents were found" in response.text

    def test_provider_and_model_name_accessors(self) -> None:
        provider = FakeLLMProvider(model_name="custom-fake-model")
        assert provider.get_provider_name() == "fake"
        assert provider.get_model_name() == "custom-fake-model"

    def test_metrics_tracked(self) -> None:
        provider = FakeLLMProvider(fixed_latency_s=0.0)
        prompt = _build_test_prompt(has_evidence=True)

        assert provider.get_metrics()["call_count"] == 0
        provider.generate(prompt)
        assert provider.get_metrics()["call_count"] == 1
        provider.generate(prompt)
        metrics = provider.get_metrics()
        assert metrics["call_count"] == 2
        assert metrics["total_generation_time_s"] >= 0.0

    def test_force_error_behavior(self) -> None:
        provider = FakeLLMProvider(force_error=True)
        prompt = _build_test_prompt(has_evidence=True)
        with pytest.raises(RuntimeError, match="configured to force error"):
            provider.generate(prompt)

    def test_shutdown_noop(self) -> None:
        provider = FakeLLMProvider()
        provider.shutdown()  # Should execute cleanly without error
