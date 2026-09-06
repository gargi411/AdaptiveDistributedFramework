"""Final End-to-End Clinical RAG Validation Suite (Phase 4.12).

Validates the complete pipeline across all 12 formal integration scenarios:
- TEST 1: Digital PDF -> direct text extraction -> RAG
- TEST 2: Scanned PDF -> adaptive OCR -> RAG
- TEST 3: Multi-page document -> page decomposition -> RAG
- TEST 4: Multiple questions against the same indexed document (index/query separation)
- TEST 5: Evidence provenance verification (Answer -> Evidence -> Chunk -> Document -> Page)
- TEST 6: GPU failure -> CPU fallback -> successful completion
- TEST 7: Empty / whitespace question handling
- TEST 8: Missing / empty document handling
- TEST 9: LLM provider unavailable / exception handling
- TEST 10: Fake LLM end-to-end validation
- TEST 11: Real LLM end-to-end validation (conditional on GEMINI_API_KEY)
- TEST 12: Dashboard UI import and compilation validation
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from adaptive_framework.core.exceptions import LLMProviderError
from adaptive_framework.document_processing.processing_strategy import (
    PageExtractionResult,
    ProcessingStrategyFactory,
)
from adaptive_framework.models.chunk import Chunk
from adaptive_framework.models.page import BoundingBox, LayoutElement, Page, PageType, ProcessingMethod
from adaptive_framework.rag.chunker import SemanticChunker
from adaptive_framework.rag.embedding.bge_embedder import BGEEmbedder
from adaptive_framework.rag.generation.clinical_rag_orchestrator import (
    ClinicalRAGOrchestrator,
)
from adaptive_framework.rag.generation.context_builder import ContextBuilder
from adaptive_framework.rag.generation.fake_llm_provider import FakeLLMProvider
from adaptive_framework.rag.generation.prompt_builder import PromptBuilder
from adaptive_framework.rag.generation.real_llm_provider import RealLLMProvider
from adaptive_framework.rag.models.clinical_models import (
    CLINICAL_DISCLAIMER,
    ClinicalDocumentUpload,
    ClinicalQueryRequest,
    DoctorClinicalResponse,
)
from adaptive_framework.rag.models.embedding import Embedding
from adaptive_framework.rag.reranking.cross_encoder_reranker import CrossEncoderReranker
from adaptive_framework.rag.retrieval.bm25_retriever import BM25Retriever
from adaptive_framework.rag.retrieval.hybrid_retriever import HybridRetriever
from adaptive_framework.rag.retrieval.reranked_retriever import RerankedRetriever
from adaptive_framework.rag.retrieval.retrieval_engine import RetrievalEngine
from adaptive_framework.rag.vector_store.faiss_manager import FAISSManager


def _mock_deterministic_vector(text: str, dim: int = 1024) -> list[float]:
    """Deterministic normalized pseudo-embedding for fast integration validation."""
    vec = np.zeros(dim, dtype=np.float32)
    for i, char in enumerate(text.lower()):
        vec[i % dim] += ord(char)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


class TestFinalValidationE2E:
    """12 Comprehensive End-to-End Validation Test Cases for Phase 4.12."""

    @pytest.fixture
    def e2e_orchestrator(self, tmp_path: Path) -> ClinicalRAGOrchestrator:
        dim = 1024
        clinical_index_dir = tmp_path / "outputs" / "final_validation" / "clinical_index"
        clinical_index_dir.mkdir(parents=True, exist_ok=True)

        embedder = MagicMock(spec=BGEEmbedder)
        embedder.embedding_dim = dim
        embedder.model_name = "BAAI/bge-large-en-v1.5"
        embedder.embed_query.side_effect = lambda q: np.array(_mock_deterministic_vector(q, dim), dtype=np.float32)
        embedder.embed_chunks.side_effect = lambda chunks: [
            Embedding(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                vector=_mock_deterministic_vector(c.text, dim),
                model_name="BAAI/bge-large-en-v1.5",
                embedding_time_seconds=0.001,
                created_at="2026-09-06T12:00:00+00:00",
            )
            for c in chunks
        ]

        manager = FAISSManager(dim=dim, index_type="flat")
        dense_engine = RetrievalEngine(
            embedding_provider=embedder,
            faiss_manager=manager,
            default_top_k=20,
        )
        bm25 = BM25Retriever(k1=1.5, b=0.75)
        hybrid_retriever = HybridRetriever(
            dense_engine=dense_engine,
            sparse_retriever=bm25,
            dense_top_k=20,
            sparse_top_k=20,
            final_top_k=15,
            rrf_k=60,
        )

        mock_ce = MagicMock()
        mock_ce.predict.side_effect = lambda pairs, batch_size=32: np.array(
            [3.5 - 0.2 * i for i in range(len(pairs))], dtype=np.float32
        )
        reranker = CrossEncoderReranker(model=mock_ce, batch_size=32)

        reranked_retriever = RerankedRetriever(
            base_retriever=hybrid_retriever,
            reranker=reranker,
            candidate_top_k=15,
            final_top_k=5,
        )

        context_builder = ContextBuilder(max_chunks=5, max_context_chars=4000)
        prompt_builder = PromptBuilder()
        fake_llm = FakeLLMProvider()

        return ClinicalRAGOrchestrator(
            retrieval_engine=reranked_retriever,
            llm_provider=fake_llm,
            context_builder=context_builder,
            prompt_builder=prompt_builder,
            embedder=embedder,
            vector_store=manager,
            sparse_retriever=bm25,
            index_storage_dir=clinical_index_dir,
        )

    # TEST 1: Digital PDF -> direct extraction -> RAG
    def test_01_digital_pdf_direct_extraction_to_rag(
        self, e2e_orchestrator: ClinicalRAGOrchestrator, tmp_path: Path
    ) -> None:
        import fitz
        doc_path = tmp_path / "digital_biomedical.pdf"
        doc = fitz.open()
        p1 = doc.new_page()
        p1.insert_text(
            (50, 72),
            "Clinical Protocol: Randomized evaluation of Empagliflozin 10mg daily in chronic kidney disease.\n"
            "Primary endpoint: reduction in sustained decline of eGFR by at least 50% or cardiovascular death.\n"
            "Secondary endpoints include hospitalisation for heart failure and all-cause mortality."
        )
        doc.save(str(doc_path))
        doc.close()

        upload = ClinicalDocumentUpload(
            filename="digital_biomedical.pdf",
            file_path=str(doc_path),
            document_id="DOC-DIGITAL-01",
        )
        ingest_res = e2e_orchestrator.index_document(upload)
        assert ingest_res.status == "success"
        assert ingest_res.page_count == 1
        assert ingest_res.chunk_count >= 1

        req = ClinicalQueryRequest(
            query="What is the primary endpoint for the Empagliflozin evaluation?",
            clinical_context="Reviewing protocol inclusion criteria.",
        )
        resp = e2e_orchestrator.query(req)
        assert resp.generation_status == "success"
        assert len(resp.evidence) > 0
        assert resp.evidence[0].document_id == "DOC-DIGITAL-01"
        assert "Empagliflozin" in resp.evidence[0].snippet or "eGFR" in resp.evidence[0].snippet

    # TEST 2: Scanned PDF -> adaptive OCR -> RAG
    def test_02_scanned_pdf_adaptive_ocr_to_rag(
        self, e2e_orchestrator: ClinicalRAGOrchestrator, tmp_path: Path
    ) -> None:
        import fitz
        # Create a PDF without digital text
        doc_path = tmp_path / "scanned_record.pdf"
        doc = fitz.open()
        p = doc.new_page()
        # Draw a rectangle to simulate a scanned image box without text
        p.draw_rect(fitz.Rect(50, 50, 400, 600), color=(0, 0, 0))
        doc.save(str(doc_path))
        doc.close()

        # Mock strategy factory and OCR strategy
        mock_factory = MagicMock(spec=ProcessingStrategyFactory)
        mock_ocr_strat = MagicMock()
        mock_ocr_strat.process.return_value = PageExtractionResult(
            text="Patient chart note: Elevated AST 84 U/L and ALT 92 U/L with acute RUQ abdominal tenderness.",
            ocr_confidence=0.96,
            processing_method="ocr",
            ocr_engine="openvino",
            ocr_device="GPU",
            ocr_time_s=0.045,
            layout_elements=[
                LayoutElement(
                    element_type="paragraph",
                    text="Patient chart note: Elevated AST 84 U/L and ALT 92 U/L with acute RUQ abdominal tenderness.",
                    bbox=BoundingBox(50, 50, 400, 600),
                    reading_order=1,
                )
            ],
        )
        mock_factory.get_strategy.return_value = mock_ocr_strat
        e2e_orchestrator.strategy_factory = mock_factory

        upload = ClinicalDocumentUpload(
            filename="scanned_record.pdf",
            file_path=str(doc_path),
            document_id="DOC-SCANNED-01",
        )
        ingest_res = e2e_orchestrator.index_document(upload)
        assert ingest_res.status == "success"
        assert ingest_res.chunk_count >= 1
        mock_factory.get_strategy.assert_called_with(PageType.SCANNED)

        req = ClinicalQueryRequest(query="What were the liver function lab results?")
        resp = e2e_orchestrator.query(req)
        assert resp.generation_status == "success"
        assert len(resp.evidence) > 0
        assert "AST 84" in resp.evidence[0].snippet or "ALT 92" in resp.evidence[0].snippet

    # TEST 3: Multi-page document -> distributed processing -> RAG
    def test_03_multi_page_document_processing_to_rag(
        self, e2e_orchestrator: ClinicalRAGOrchestrator, tmp_path: Path
    ) -> None:
        import fitz
        doc_path = tmp_path / "multipage_record.pdf"
        doc = fitz.open()
        p1 = doc.new_page()
        p1.insert_text((50, 72), "Page 1: Admission Assessment. Patient CR-8819 presented with acute dyspnea.")
        p2 = doc.new_page()
        p2.insert_text((50, 72), "Page 2: Diagnostic Imaging. Chest CT revealed bilateral pulmonary infiltrates.")
        p3 = doc.new_page()
        p3.insert_text((50, 72), "Page 3: Treatment Course. Initiated IV Methylprednisolone 40mg q12h.")
        doc.save(str(doc_path))
        doc.close()

        upload = ClinicalDocumentUpload(
            filename="multipage_record.pdf",
            file_path=str(doc_path),
            document_id="DOC-MULTI-01",
        )
        ingest_res = e2e_orchestrator.index_document(upload)
        assert ingest_res.status == "success"
        assert ingest_res.page_count == 3
        assert ingest_res.chunk_count >= 1

        req = ClinicalQueryRequest(query="What was the diagnostic chest CT finding?")
        resp = e2e_orchestrator.query(req)
        assert resp.generation_status == "success"
        assert len(resp.evidence) > 0
        # Check page provenance
        page_nums = [p for ev in resp.evidence for p in ev.page_numbers]
        assert 2 in page_nums

    # TEST 4: Multiple questions against the same indexed document
    def test_04_multiple_queries_against_same_index(
        self, e2e_orchestrator: ClinicalRAGOrchestrator, tmp_path: Path
    ) -> None:
        note_text = (
            "Subject: Clinic Follow-up Note\n"
            "History: 45-year-old male with persistent cough and nocturnal wheezing.\n"
            "Diagnosis: Moderate Persistent Asthma.\n"
            "Medications: Fluticasone/Salmeterol 250/50 mcg 1 inhalation BID.\n"
            "Allergies: Penicillin (severe urticaria and bronchospasm)."
        )
        f = tmp_path / "asthma_followup.txt"
        f.write_text(note_text, encoding="utf-8")

        # Ingest once
        ingest = e2e_orchestrator.index_document(
            ClinicalDocumentUpload(filename="asthma_followup.txt", file_path=str(f), document_id="DOC-ASTHMA")
        )
        assert ingest.status == "success"

        # Query 1
        q1 = ClinicalQueryRequest(query="What is the patient's asthma medication regimen?")
        r1 = e2e_orchestrator.query(q1)
        assert r1.generation_status == "success"
        assert len(r1.evidence) > 0

        # Query 2 (reusing existing index)
        q2 = ClinicalQueryRequest(query="What known drug allergies does the patient have?")
        r2 = e2e_orchestrator.query(q2)
        assert r2.generation_status == "success"
        assert len(r2.evidence) > 0
        assert "Penicillin" in r2.evidence[0].snippet

    # TEST 5: Evidence provenance verification
    def test_05_evidence_provenance_verification(
        self, e2e_orchestrator: ClinicalRAGOrchestrator, tmp_path: Path
    ) -> None:
        note_text = (
            "OPERATIVE REPORT\n"
            "PROCEDURE: Laparoscopic Appendectomy\n"
            "SURGEON: Dr. R. Vance, MD\n"
            "FINDINGS: Suppurative, non-perforated retrocecal appendix with mild surrounding serositis."
        )
        f = tmp_path / "op_report.txt"
        f.write_text(note_text, encoding="utf-8")

        e2e_orchestrator.index_document(
            ClinicalDocumentUpload(filename="op_report.txt", file_path=str(f), document_id="DOC-OP-01")
        )

        resp = e2e_orchestrator.query(ClinicalQueryRequest(query="What were the operative findings of the appendix?"))
        assert resp.generation_status == "success"
        assert len(resp.evidence) > 0
        ev = resp.evidence[0]
        # Verify full provenance chain
        assert ev.document_id == "DOC-OP-01"
        assert ev.source_file == str(f)
        assert ev.chunk_id is not None
        assert 1 in ev.page_numbers
        assert ev.rank == 1
        assert ev.score > 0.0

    # TEST 6: GPU failure -> CPU fallback -> successful completion
    def test_06_gpu_failure_cpu_fallback(
        self, e2e_orchestrator: ClinicalRAGOrchestrator, tmp_path: Path
    ) -> None:
        import fitz
        doc_path = tmp_path / "fallback_scanned.pdf"
        doc = fitz.open()
        p = doc.new_page()
        p.draw_rect(fitz.Rect(10, 10, 200, 200), color=(0, 0, 0))
        doc.save(str(doc_path))
        doc.close()

        mock_factory = MagicMock()
        mock_strategy = MagicMock()
        mock_strategy.process.return_value = PageExtractionResult(
            text="Fallback Recovered Note: Blood pressure 128/82 mmHg, resting heart rate 72 bpm.",
            ocr_confidence=0.91,
            processing_method="ocr",
            ocr_device="CPU_FALLBACK",
            warnings=["GPU failed; executed CPU fallback"],
        )
        mock_factory.get_strategy.return_value = mock_strategy
        e2e_orchestrator.strategy_factory = mock_factory

        ingest = e2e_orchestrator.index_document(
            ClinicalDocumentUpload(filename="fallback_scanned.pdf", file_path=str(doc_path), document_id="DOC-FB-01")
        )
        assert ingest.status == "success"
        resp = e2e_orchestrator.query(ClinicalQueryRequest(query="What is the blood pressure and resting heart rate?"))
        assert resp.generation_status == "success"
        assert "128/82" in resp.evidence[0].snippet

    # TEST 7: Empty/invalid question handling
    def test_07_empty_question_handling(self, e2e_orchestrator: ClinicalRAGOrchestrator) -> None:
        with pytest.raises(ValueError, match="Clinical query must not be empty"):
            ClinicalQueryRequest(query="")
        with pytest.raises(ValueError, match="Clinical query must not be empty"):
            ClinicalQueryRequest(query="   \t\n  ")

    # TEST 8: Missing/empty document handling
    def test_08_empty_document_handling(
        self, e2e_orchestrator: ClinicalRAGOrchestrator, tmp_path: Path
    ) -> None:
        empty_file = tmp_path / "empty_doc.txt"
        empty_file.write_text("   \n   ", encoding="utf-8")
        ingest = e2e_orchestrator.index_document(
            ClinicalDocumentUpload(filename="empty_doc.txt", file_path=str(empty_file))
        )
        assert ingest.status == "failed"
        assert ingest.page_count == 0

    # TEST 9: LLM unavailable -> graceful failure
    def test_09_llm_unavailable_graceful_failure(
        self, e2e_orchestrator: ClinicalRAGOrchestrator, tmp_path: Path
    ) -> None:
        note_text = "Progress note: Routine postoperative follow-up uncomplicated."
        f = tmp_path / "postop.txt"
        f.write_text(note_text, encoding="utf-8")
        e2e_orchestrator.index_document(ClinicalDocumentUpload(filename="postop.txt", file_path=str(f)))

        broken_llm = MagicMock()
        broken_llm.generate.side_effect = LLMProviderError("Remote LLM connection timeout")
        e2e_orchestrator.llm_provider = broken_llm

        resp = e2e_orchestrator.query(ClinicalQueryRequest(query="Was the follow-up uncomplicated?"))
        assert resp.generation_status == "llm_failure"
        assert "Remote LLM connection timeout" in resp.answer_text
        # Evidence was still retrieved prior to LLM failure
        assert len(resp.evidence) > 0

    # TEST 10: Fake LLM end-to-end validation
    def test_10_fake_llm_end_to_end_validation(
        self, e2e_orchestrator: ClinicalRAGOrchestrator, tmp_path: Path
    ) -> None:
        assert isinstance(e2e_orchestrator.llm_provider, FakeLLMProvider)
        note_text = "Diagnostic Pathology: Benign thyroid follicular adenoma, clear margins."
        f = tmp_path / "pathology.txt"
        f.write_text(note_text, encoding="utf-8")
        e2e_orchestrator.index_document(ClinicalDocumentUpload(filename="pathology.txt", file_path=str(f)))

        resp = e2e_orchestrator.query(ClinicalQueryRequest(query="What is the pathology diagnosis?"))
        assert resp.generation_status == "success"
        assert resp.provider_name == "fake"
        assert resp.disclaimer == CLINICAL_DISCLAIMER

    # TEST 11: Real LLM end-to-end validation (conditional on GEMINI_API_KEY)
    @pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY environment variable required for live Gemini test",
    )
    def test_11_real_llm_end_to_end_validation(
        self, e2e_orchestrator: ClinicalRAGOrchestrator, tmp_path: Path
    ) -> None:
        real_llm = RealLLMProvider(
            api_key=os.environ["GEMINI_API_KEY"],
            model_name="gemini-2.5-flash",
            temperature=0.0,
        )
        e2e_orchestrator.llm_provider = real_llm
        note_text = "Discharge Summary: Metformin titrated to 1000mg twice daily with meals."
        f = tmp_path / "diabetic_note.txt"
        f.write_text(note_text, encoding="utf-8")
        e2e_orchestrator.index_document(ClinicalDocumentUpload(filename="diabetic_note.txt", file_path=str(f)))

        resp = e2e_orchestrator.query(ClinicalQueryRequest(query="What dose of Metformin was prescribed?"))
        assert resp.generation_status == "success"
        assert resp.provider_name == "gemini"
        assert "1000" in resp.answer_text or "Metformin" in resp.answer_text

    # TEST 12: Dashboard / application import and execution validation
    def test_12_dashboard_import_and_compilation(self) -> None:
        # Verify clinical dashboard imports cleanly
        dashboard_module = importlib.import_module("dashboard.clinical_rag_app")
        assert hasattr(dashboard_module, "get_orchestrator")
        assert hasattr(dashboard_module, "main")

        # Verify cluster engineering dashboard imports cleanly
        eng_app = importlib.import_module("dashboard.app")
        assert hasattr(eng_app, "main")
