"""Unit tests for RAGService orchestration (Phase 4.5)."""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from adaptive_framework.rag.generation.answer import RAGAnswer
from adaptive_framework.rag.generation.context_builder import ContextBuilder
from adaptive_framework.rag.generation.fake_llm_provider import FakeLLMProvider
from adaptive_framework.rag.generation.prompt_builder import PromptBuilder
from adaptive_framework.rag.generation.rag_service import RAGService
from adaptive_framework.rag.interfaces.i_retrieval_engine import IRetrievalEngine
from adaptive_framework.rag.retrieval.retrieval_result import RetrievalResult


def _mock_result(
    rank: int = 1,
    score: float = 0.92,
    doc_id: str = "doc_resp_01",
    text: str = "Patients diagnosed with asthma frequently exhibit wheezing and dyspnea.",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"chk_{rank}",
        document_id=doc_id,
        source_file="pulmonology.pdf",
        page_numbers=(5,),
        text=text,
        section_heading="Asthma Symptoms",
        document_type="clinical",
        chunk_index=rank - 1,
        score=score,
        rank=rank,
    )


class TestRAGService:
    """Test suite for RAGService end-to-end orchestration and edge case handling."""

    @pytest.fixture
    def mock_retrieval_engine(self) -> MagicMock:
        engine = MagicMock(spec=IRetrievalEngine)
        engine.retrieve.return_value = [_mock_result()]
        engine.get_metrics.return_value = {"engine_latency_ms": 1.2}
        return engine

    @pytest.fixture
    def rag_service(self, mock_retrieval_engine: MagicMock) -> RAGService:
        return RAGService(
            retrieval_engine=mock_retrieval_engine,
            context_builder=ContextBuilder(),
            prompt_builder=PromptBuilder(),
            llm_provider=FakeLLMProvider(fixed_latency_s=0.0),
        )

    def test_happy_path(self, rag_service: RAGService, mock_retrieval_engine: MagicMock) -> None:
        answer = rag_service.answer("What are asthma symptoms?")

        assert isinstance(answer, RAGAnswer)
        assert answer.generation_status == "ok"
        assert answer.query == "What are asthma symptoms?"
        assert len(answer.evidence) == 1
        assert len(answer.source_references) == 1
        assert answer.source_references[0].document_id == "doc_resp_01"
        assert "wheezing" in answer.answer_text
        assert mock_retrieval_engine.retrieve.called

    def test_empty_query_returns_empty_query_status(self, rag_service: RAGService, mock_retrieval_engine: MagicMock) -> None:
        answer = rag_service.answer("   ")
        assert answer.generation_status == "empty_query"
        assert len(answer.evidence) == 0
        assert not mock_retrieval_engine.retrieve.called

    def test_none_query_returns_empty_query_status(self, rag_service: RAGService, mock_retrieval_engine: MagicMock) -> None:
        answer = rag_service.answer("")
        assert answer.generation_status == "empty_query"
        assert not mock_retrieval_engine.retrieve.called

    def test_no_retrieval_results_returns_no_results_status(
        self, mock_retrieval_engine: MagicMock
    ) -> None:
        mock_retrieval_engine.retrieve.return_value = []
        llm = FakeLLMProvider()
        service = RAGService(
            retrieval_engine=mock_retrieval_engine,
            context_builder=ContextBuilder(),
            prompt_builder=PromptBuilder(),
            llm_provider=llm,
        )

        answer = service.answer("What is the rare disease X?")
        assert answer.generation_status == "no_results"
        assert len(answer.evidence) == 0
        assert "No relevant documents" in answer.answer_text
        # LLM generate should not have been called
        assert llm.get_metrics()["call_count"] == 0

    def test_retrieval_engine_exception_handled(self, mock_retrieval_engine: MagicMock) -> None:
        mock_retrieval_engine.retrieve.side_effect = RuntimeError("FAISS search failed")
        service = RAGService(
            retrieval_engine=mock_retrieval_engine,
            context_builder=ContextBuilder(),
            prompt_builder=PromptBuilder(),
            llm_provider=FakeLLMProvider(),
        )

        answer = service.answer("Query that fails retrieval")
        assert answer.generation_status == "retrieval_error"
        assert len(answer.evidence) == 0

    def test_llm_exception_handled(self, mock_retrieval_engine: MagicMock) -> None:
        service = RAGService(
            retrieval_engine=mock_retrieval_engine,
            context_builder=ContextBuilder(),
            prompt_builder=PromptBuilder(),
            llm_provider=FakeLLMProvider(force_error=True),
        )

        answer = service.answer("Query where LLM fails")
        assert answer.generation_status == "llm_error"
        assert len(answer.evidence) == 1  # Evidence was retrieved
        assert "error occurred during answer generation" in answer.answer_text

    def test_latencies_measured_and_positive(self, rag_service: RAGService) -> None:
        answer = rag_service.answer("What is asthma?")
        assert answer.retrieval_latency_s >= 0.0
        assert answer.context_latency_s >= 0.0
        assert answer.prompt_latency_s >= 0.0
        assert answer.generation_latency_s >= 0.0
        assert answer.total_latency_s >= 0.0

    def test_provenance_preserved_end_to_end(self, rag_service: RAGService) -> None:
        answer = rag_service.answer("What is asthma?")
        ev = answer.evidence[0]
        assert ev.document_id == "doc_resp_01"
        assert ev.source_file == "pulmonology.pdf"
        assert ev.page_numbers == (5,)
        assert ev.section_heading == "Asthma Symptoms"

    def test_source_references_deduplicated(self, mock_retrieval_engine: MagicMock) -> None:
        # Two chunks from the same document and same page
        r1 = _mock_result(rank=1, doc_id="doc_1", text="Part 1")
        r2 = _mock_result(rank=2, doc_id="doc_1", text="Part 2")
        mock_retrieval_engine.retrieve.return_value = [r1, r2]

        service = RAGService(
            retrieval_engine=mock_retrieval_engine,
            context_builder=ContextBuilder(),
            prompt_builder=PromptBuilder(),
            llm_provider=FakeLLMProvider(),
        )

        answer = service.answer("Query")
        assert len(answer.evidence) == 2
        # source_references must deduplicate identical (doc_id, source_file, page_numbers)
        assert len(answer.source_references) == 1

    def test_to_dict_full_pipeline_output(self, rag_service: RAGService) -> None:
        answer = rag_service.answer("What is asthma?")
        d = answer.to_dict()
        assert d["generation_status"] == "ok"
        assert "evidence" in d
        assert "source_references" in d
        assert "retrieval_latency_s" in d
