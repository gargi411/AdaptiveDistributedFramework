"""Unit tests for ClinicalRAGOrchestrator and Clinical Models (Phase 4.11).

Tests ingestion, validation, provenance preservation, grounding, error handling,
and structured response generation with offline mocks.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from adaptive_framework.core.exceptions import (
    LLMProviderError,
    RerankerError,
    RetrievalEngineError,
)
from adaptive_framework.rag.generation.clinical_rag_orchestrator import (
    ClinicalRAGOrchestrator,
)
from adaptive_framework.rag.generation.context_builder import ContextBuilder
from adaptive_framework.rag.generation.fake_llm_provider import FakeLLMProvider
from adaptive_framework.rag.generation.llm_response import LLMResponse
from adaptive_framework.rag.generation.prompt_builder import PromptBuilder
from adaptive_framework.rag.interfaces.i_retrieval_engine import IRetrievalEngine
from adaptive_framework.rag.models.clinical_models import (
    CLINICAL_DISCLAIMER,
    ClinicalDocumentUpload,
    ClinicalQueryRequest,
    DoctorClinicalResponse,
    DoctorEvidenceItem,
    DocumentIngestionResult,
)
from adaptive_framework.rag.reranking.rerank_result import RerankedRetrievalResult
from adaptive_framework.rag.retrieval.retrieval_result import RetrievalResult


def _make_reranked_result(
    chunk_id: str = "c1",
    document_id: str = "doc_101",
    score: float = 2.45,
    rank: int = 1,
    page_numbers: tuple[int, ...] = (1, 2),
    text: str = "Patient has elevated HbA1c of 8.4% and fasting blood sugar of 165 mg/dL.",
    section_heading: str = "Laboratory Findings",
) -> RerankedRetrievalResult:
    return RerankedRetrievalResult(
        chunk_id=chunk_id,
        document_id=document_id,
        source_file="patient_101.pdf",
        page_numbers=page_numbers,
        text=text,
        section_heading=section_heading,
        document_type="clinical_note",
        chunk_index=0,
        score=score,
        rank=rank,
        dense_rank=1,
        sparse_rank=2,
        dense_score=0.88,
        sparse_score=11.5,
        rrf_score=0.032,
        reranker_score=score,
        reranker_rank=rank,
    )


class TestClinicalModels:
    """Tests for structured clinical request and response models."""

    def test_clinical_query_request_validation(self) -> None:
        # Valid request
        req = ClinicalQueryRequest(query="What is the HbA1c level?", top_k=5, candidate_top_k=15)
        assert req.query == "What is the HbA1c level?"
        assert req.top_k == 5
        assert req.candidate_top_k == 15
        assert req.to_dict()["query"] == "What is the HbA1c level?"

        # Empty query validation
        with pytest.raises(ValueError, match="Clinical query must not be empty"):
            ClinicalQueryRequest(query="")
        with pytest.raises(ValueError, match="Clinical query must not be empty"):
            ClinicalQueryRequest(query="   ")

        # Invalid top_k
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            ClinicalQueryRequest(query="valid query", top_k=0)

        # Invalid candidate_top_k
        with pytest.raises(ValueError, match="candidate_top_k must be >= 1"):
            ClinicalQueryRequest(query="valid query", candidate_top_k=0)

    def test_clinical_document_upload_validation(self) -> None:
        # Valid upload with path
        up1 = ClinicalDocumentUpload(filename="note.pdf", file_path="/path/note.pdf")
        assert up1.filename == "note.pdf"

        # Valid upload with bytes
        up2 = ClinicalDocumentUpload(filename="note.pdf", file_bytes=b"%PDF-1.4...")
        assert up2.file_bytes == b"%PDF-1.4..."

        # Empty filename
        with pytest.raises(ValueError, match="Filename must not be empty"):
            ClinicalDocumentUpload(filename="", file_bytes=b"data")

        # Missing both file_path and file_bytes
        with pytest.raises(ValueError, match="Either file_path or file_bytes must be provided"):
            ClinicalDocumentUpload(filename="note.pdf")


class TestClinicalRAGOrchestrator:
    """Tests for ClinicalRAGOrchestrator operations and error handling."""

    @pytest.fixture
    def mock_retrieval_engine(self) -> MagicMock:
        engine = MagicMock(spec=IRetrievalEngine)
        engine.retrieve.return_value = [_make_reranked_result()]
        return engine

    @pytest.fixture
    def fake_llm(self) -> FakeLLMProvider:
        return FakeLLMProvider()

    @pytest.fixture
    def orchestrator(
        self, mock_retrieval_engine: MagicMock, fake_llm: FakeLLMProvider, tmp_path: Path
    ) -> ClinicalRAGOrchestrator:
        return ClinicalRAGOrchestrator(
            retrieval_engine=mock_retrieval_engine,
            llm_provider=fake_llm,
            index_storage_dir=tmp_path / "clinical_index",
        )

    def test_query_end_to_end_success(
        self, orchestrator: ClinicalRAGOrchestrator, mock_retrieval_engine: MagicMock
    ) -> None:
        req = ClinicalQueryRequest(
            query="What was the HbA1c value?",
            clinical_context="Patient reports fatigue and polyuria.",
            top_k=5,
            candidate_top_k=15,
        )

        resp = orchestrator.query(req)

        assert isinstance(resp, DoctorClinicalResponse)
        assert resp.generation_status == "success"
        assert resp.query == "What was the HbA1c value?"
        assert resp.clinical_context == "Patient reports fatigue and polyuria."
        assert "HbA1c of 8.4%" in resp.answer_text
        assert resp.disclaimer == CLINICAL_DISCLAIMER

        # Verify provenance preservation
        assert len(resp.evidence) == 1
        ev = resp.evidence[0]
        assert isinstance(ev, DoctorEvidenceItem)
        assert ev.chunk_id == "c1"
        assert ev.document_id == "doc_101"
        assert ev.page_numbers == (1, 2)
        assert ev.section_heading == "Laboratory Findings"
        assert ev.dense_rank == 1
        assert ev.sparse_rank == 2
        assert ev.reranker_rank == 1
        assert ev.score == 2.45

        # Verify source references
        assert len(resp.source_references) == 1
        assert resp.source_references[0].document_id == "doc_101"
        assert resp.source_references[0].page_numbers == (1, 2)

        # Verify stage latencies
        assert "retrieval_ms" in resp.stage_latencies_ms
        assert "prompt_ms" in resp.stage_latencies_ms
        assert "generation_ms" in resp.stage_latencies_ms
        assert "total_ms" in resp.stage_latencies_ms
        assert resp.stage_latencies_ms["total_ms"] >= 0.0

        # Verify serialization
        d = resp.to_dict()
        assert d["generation_status"] == "success"
        assert len(d["evidence"]) == 1

    def test_query_explicit_symptom_prompt_separation(
        self, mock_retrieval_engine: MagicMock, fake_llm: FakeLLMProvider, tmp_path: Path
    ) -> None:
        mock_prompt_builder = MagicMock(spec=PromptBuilder)
        mock_prompt_builder.build.return_value = MagicMock(full_prompt_text="mock prompt")

        orch = ClinicalRAGOrchestrator(
            retrieval_engine=mock_retrieval_engine,
            llm_provider=fake_llm,
            prompt_builder=mock_prompt_builder,
            index_storage_dir=tmp_path / "clinical_index",
        )

        req = ClinicalQueryRequest(
            query="Check creatinine levels.",
            clinical_context="Patient reports mild edema.",
        )
        orch.query(req)

        mock_prompt_builder.build.assert_called_once()
        passed_query = mock_prompt_builder.build.call_args[1]["query"]
        assert "USER QUESTION:\nCheck creatinine levels." in passed_query
        assert "PATIENT CONTEXT (SYMPTOMS REPORTED BY USER - NOT VERIFIED RECORD FINDINGS):" in passed_query
        assert "Patient reports mild edema." in passed_query
        assert "Distinguish documented findings in the uploaded record from user-reported symptoms" in passed_query

    def test_query_empty_query_handled(self, orchestrator: ClinicalRAGOrchestrator) -> None:
        # Create request bypassing __post_init__ or query with whitespace
        req = ClinicalQueryRequest.__new__(ClinicalQueryRequest)
        object.__setattr__(req, "query", "   ")
        object.__setattr__(req, "document_id", None)
        object.__setattr__(req, "clinical_context", None)
        object.__setattr__(req, "top_k", 5)
        object.__setattr__(req, "candidate_top_k", 15)
        object.__setattr__(req, "metadata_filters", None)

        resp = orchestrator.query(req)
        assert resp.generation_status == "empty_query"
        assert "Empty query" in resp.answer_text
        assert resp.evidence == ()

    def test_query_insufficient_evidence_handled(
        self, orchestrator: ClinicalRAGOrchestrator, mock_retrieval_engine: MagicMock
    ) -> None:
        mock_retrieval_engine.retrieve.return_value = []

        req = ClinicalQueryRequest(query="Find unknown condition.")
        resp = orchestrator.query(req)

        assert resp.generation_status == "insufficient_evidence"
        assert "Insufficient evidence" in resp.answer_text
        assert resp.evidence == ()

    def test_query_retrieval_failure_handled(
        self, orchestrator: ClinicalRAGOrchestrator, mock_retrieval_engine: MagicMock
    ) -> None:
        mock_retrieval_engine.retrieve.side_effect = RetrievalEngineError("FAISS index unavailable")

        req = ClinicalQueryRequest(query="Check lab values.")
        resp = orchestrator.query(req)

        assert resp.generation_status == "retrieval_failure"
        assert "Retrieval error" in resp.answer_text
        assert resp.evidence == ()

    def test_query_reranker_failure_handled(
        self, orchestrator: ClinicalRAGOrchestrator, mock_retrieval_engine: MagicMock
    ) -> None:
        mock_retrieval_engine.retrieve.side_effect = RerankerError("Cross-encoder device memory error")

        req = ClinicalQueryRequest(query="Check lab values.")
        resp = orchestrator.query(req)

        assert resp.generation_status == "reranker_failure"
        assert "Retrieval error" in resp.answer_text

    def test_query_llm_failure_handled_preserves_evidence(
        self, orchestrator: ClinicalRAGOrchestrator, fake_llm: FakeLLMProvider
    ) -> None:
        fake_llm._force_error = True

        req = ClinicalQueryRequest(query="Check lab values.")
        resp = orchestrator.query(req)

        assert resp.generation_status == "llm_failure"
        assert "LLM generation failed" in resp.answer_text
        # Evidence should still be preserved so doctor can inspect the retrieved passages!
        assert len(resp.evidence) == 1
        assert resp.evidence[0].chunk_id == "c1"

    def test_document_ingestion_text_flow(
        self, orchestrator: ClinicalRAGOrchestrator, tmp_path: Path
    ) -> None:
        text_content = (
            "CLINICAL DISCHARGE SUMMARY\n\n"
            "PATIENT: John Doe\n"
            "DIAGNOSIS: Type 2 Diabetes Mellitus with peripheral neuropathy.\n\n"
            "LABORATORY FINDINGS:\n"
            "HbA1c was measured at 8.9% on admission and stabilized to 7.8%.\n\n"
            "MEDICATIONS:\n"
            "Metformin 500mg BD, Glimepiride 1mg OD.\n"
        )
        sample_file = tmp_path / "sample_note.txt"
        sample_file.write_text(text_content, encoding="utf-8")

        upload = ClinicalDocumentUpload(
            filename="sample_note.txt",
            file_path=str(sample_file),
            document_id="doc_test_001",
        )

        res = orchestrator.index_document(upload)

        assert isinstance(res, DocumentIngestionResult)
        assert res.status == "success"
        assert res.document_id == "doc_test_001"
        assert res.page_count >= 1
        assert res.chunk_count >= 1
        assert res.ingestion_latency_ms >= 0.0

        indexed = orchestrator.get_indexed_documents()
        assert len(indexed) == 1
        assert indexed[0].document_id == "doc_test_001"

    def test_document_ingestion_empty_file_fails_gracefully(
        self, orchestrator: ClinicalRAGOrchestrator, tmp_path: Path
    ) -> None:
        empty_file = tmp_path / "empty_note.txt"
        empty_file.write_text("", encoding="utf-8")

        upload = ClinicalDocumentUpload(
            filename="empty_note.txt",
            file_path=str(empty_file),
        )

        res = orchestrator.index_document(upload)
        assert res.status == "failed"
        assert res.chunk_count == 0
        assert "no extractable text" in (res.error_message or "").lower()
