"""Unit tests for RAGAnswer and SourceReference models (Phase 4.5)."""

from __future__ import annotations

import pytest

from adaptive_framework.rag.generation.answer import RAGAnswer, SourceReference
from adaptive_framework.rag.retrieval.retrieval_result import RetrievalResult


def _make_sample_answer(status: str = "ok") -> RAGAnswer:
    res = RetrievalResult(
        chunk_id="chk_01",
        document_id="doc_abc",
        source_file="doc_abc.pdf",
        page_numbers=(1, 2),
        text="Normal range for serum sodium is 135 to 145 mEq/L.",
        section_heading="Electrolytes",
        document_type="lab",
        chunk_index=0,
        score=0.93,
        rank=1,
    )
    src = SourceReference(
        document_id="doc_abc",
        source_file="doc_abc.pdf",
        page_numbers=(1, 2),
    )
    return RAGAnswer(
        query="What is normal sodium range?",
        answer_text="Based on evidence: Normal range is 135 to 145 mEq/L.",
        evidence=(res,),
        source_references=(src,),
        retrieval_metrics={"engine_latency_ms": 4.5},
        provider_name="fake",
        model_name="fake-llm-v1",
        retrieval_latency_s=0.005,
        context_latency_s=0.001,
        prompt_latency_s=0.001,
        generation_latency_s=0.010,
        total_latency_s=0.017,
        context_chars=50,
        num_retrieved=1,
        generation_status=status,
    )


class TestAnswerModel:
    """Test suite for RAGAnswer and SourceReference serialization and properties."""

    def test_all_fields_present(self) -> None:
        answer = _make_sample_answer()
        assert answer.query == "What is normal sodium range?"
        assert "135 to 145" in answer.answer_text
        assert len(answer.evidence) == 1
        assert len(answer.source_references) == 1
        assert answer.provider_name == "fake"
        assert answer.model_name == "fake-llm-v1"
        assert answer.generation_status == "ok"
        assert answer.num_retrieved == 1

    def test_to_dict_serialisable(self) -> None:
        answer = _make_sample_answer()
        data = answer.to_dict()

        assert isinstance(data, dict)
        assert data["query"] == answer.query
        assert data["generation_status"] == "ok"
        assert len(data["evidence"]) == 1
        assert data["evidence"][0]["document_id"] == "doc_abc"
        assert len(data["source_references"]) == 1
        assert data["source_references"][0]["source_file"] == "doc_abc.pdf"
        assert data["source_references"][0]["page_numbers"] == [1, 2]

    def test_source_reference_to_dict(self) -> None:
        src = SourceReference("doc_1", "file.pdf", (1, 2, 3))
        d = src.to_dict()
        assert d["document_id"] == "doc_1"
        assert d["source_file"] == "file.pdf"
        assert d["page_numbers"] == [1, 2, 3]

    def test_latencies_non_negative(self) -> None:
        answer = _make_sample_answer()
        assert answer.retrieval_latency_s >= 0.0
        assert answer.context_latency_s >= 0.0
        assert answer.prompt_latency_s >= 0.0
        assert answer.generation_latency_s >= 0.0
        assert answer.total_latency_s >= 0.0

    def test_generation_status_values(self) -> None:
        for status in ("ok", "no_results", "llm_error", "empty_query", "retrieval_error", "context_error", "prompt_error"):
            answer = _make_sample_answer(status=status)
            assert answer.generation_status == status
