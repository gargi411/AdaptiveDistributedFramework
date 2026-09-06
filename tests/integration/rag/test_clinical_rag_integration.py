"""End-to-end integration tests for Doctor-Facing RAG Pipeline (Phase 4.11).

Tests the full path:
Document Upload (PDF/EHR) -> Document Processing -> UnifiedDocument
-> Semantic Chunking -> Clinical Index (FAISS + BM25)
-> Doctor Question + Symptoms -> Hybrid Retrieval + Cross-Encoder Reranking
-> ContextBuilder -> PromptBuilder -> LLM -> DoctorClinicalResponse
-> Answer + Evidence + Full Page/Source Provenance.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest

from adaptive_framework.models.chunk import Chunk
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
    """Generate normalized pseudo-embedding based on text hash for fast integration tests."""
    vec = np.zeros(dim, dtype=np.float32)
    for i, char in enumerate(text.lower()):
        vec[i % dim] += ord(char)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


class TestClinicalRAGIntegration:
    """Integration test suite for the complete doctor-facing clinical pipeline."""

    @pytest.fixture
    def integration_stack(self, tmp_path: Path) -> tuple[ClinicalRAGOrchestrator, FakeLLMProvider]:
        dim = 1024
        clinical_index_dir = tmp_path / "outputs" / "rag" / "clinical_index"
        clinical_index_dir.mkdir(parents=True, exist_ok=True)

        # 1. Embedder mock
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

        # 2. FAISS Vector Store
        manager = FAISSManager(dim=dim, index_type="flat")

        # 3. Dense Engine
        dense_engine = RetrievalEngine(
            embedding_provider=embedder,
            faiss_manager=manager,
            default_top_k=20,
        )

        # 4. Sparse Engine
        bm25 = BM25Retriever(k1=1.5, b=0.75)

        # 5. Stage 1 Hybrid Engine (RRF k=60, candidate_top_k=15)
        hybrid_retriever = HybridRetriever(
            dense_engine=dense_engine,
            sparse_retriever=bm25,
            dense_top_k=20,
            sparse_top_k=20,
            final_top_k=15,
            rrf_k=60,
        )

        # 6. Stage 2 Cross-Encoder Reranker (mocked predict for fast offline integration)
        mock_ce_model = MagicMock()
        mock_ce_model.predict.side_effect = lambda pairs, batch_size=32: np.array(
            [2.5 - 0.1 * i for i in range(len(pairs))], dtype=np.float32
        )
        reranker = CrossEncoderReranker(
            model=mock_ce_model,
            batch_size=32,
        )

        # 7. Two-Stage Reranked Retriever (frozen Phase 4.10 parameters)
        reranked_retriever = RerankedRetriever(
            base_retriever=hybrid_retriever,
            reranker=reranker,
            candidate_top_k=15,
            final_top_k=5,
        )

        # 8. Generation Components
        context_builder = ContextBuilder(max_chunks=5, max_context_chars=4000)
        prompt_builder = PromptBuilder()
        fake_llm = FakeLLMProvider()

        # 9. Orchestrator
        orchestrator = ClinicalRAGOrchestrator(
            retrieval_engine=reranked_retriever,
            llm_provider=fake_llm,
            context_builder=context_builder,
            prompt_builder=prompt_builder,
            embedder=embedder,
            vector_store=manager,
            sparse_retriever=bm25,
            index_storage_dir=clinical_index_dir,
        )

        return orchestrator, fake_llm

    def test_end_to_end_clinical_workflow(
        self, integration_stack: tuple[ClinicalRAGOrchestrator, FakeLLMProvider], tmp_path: Path
    ) -> None:
        orchestrator, _ = integration_stack

        # Step 1: Doctor uploads clinical discharge note
        clinical_note = (
            "PATIENT DISCHARGE SUMMARY\n"
            "PATIENT ID: CR-49201\n"
            "ADMISSION DIAGNOSIS: Acute Coronary Syndrome, Unstable Angina\n"
            "DISCHARGE DIAGNOSIS: Non-ST Elevation Myocardial Infarction (NSTEMI), Hypertension, Dyslipidemia\n\n"
            "HOSPITAL COURSE:\n"
            "62-year-old male admitted with crushing substernal chest pain radiating to left jaw.\n"
            "ECG showed ST depressions in leads V4-V6. Troponin I elevated at 2.4 ng/mL.\n"
            "Coronary angiography revealed 85% stenosis in mid-LAD, successfully stented with DES.\n\n"
            "DISCHARGE MEDICATIONS:\n"
            "1. Aspirin 75mg once daily\n"
            "2. Ticagrelor 90mg twice daily\n"
            "3. Atorvastatin 80mg once daily at night\n"
            "4. Ramipril 2.5mg once daily\n"
            "5. Metoprolol Succinate 25mg once daily\n\n"
            "DISCHARGE INSTRUCTIONS & WARNINGS:\n"
            "Strict compliance with Dual Antiplatelet Therapy (DAPT) for minimum 12 months.\n"
            "Follow up in Cardiology Clinic in 2 weeks with repeat lipid profile and ECG.\n"
        )
        note_file = tmp_path / "CR-49201_discharge.txt"
        note_file.write_text(clinical_note, encoding="utf-8")

        upload = ClinicalDocumentUpload(
            filename="CR-49201_discharge.txt",
            file_path=str(note_file),
            document_id="CR-49201",
        )

        # Ingestion & Indexing
        ingest_res = orchestrator.index_document(upload)
        assert ingest_res.status == "success"
        assert ingest_res.document_id == "CR-49201"
        assert ingest_res.chunk_count >= 1

        # Step 2: Doctor submits clinical question with symptoms
        request = ClinicalQueryRequest(
            query="What discharge medications and DAPT duration were prescribed for this patient?",
            clinical_context="Patient reports mild epigastric burning after starting medications.",
            top_k=5,
            candidate_top_k=15,
        )

        response = orchestrator.query(request)

        # Step 3: Verify structured response and clinical provenance
        assert isinstance(response, DoctorClinicalResponse)
        assert response.generation_status == "success"
        assert response.query == request.query
        assert response.clinical_context == request.clinical_context
        assert response.disclaimer == CLINICAL_DISCLAIMER

        # Check evidence citations and provenance
        assert len(response.evidence) > 0
        top_ev = response.evidence[0]
        assert top_ev.document_id == "CR-49201"
        assert top_ev.source_file == str(note_file)
        assert 1 in top_ev.page_numbers
        assert top_ev.rank == 1
        assert top_ev.score > 0.0

        # Check source references
        assert len(response.source_references) == 1
        assert response.source_references[0].document_id == "CR-49201"

        # Check latency tracking across stages
        lats = response.stage_latencies_ms
        assert lats["retrieval_ms"] >= 0.0
        assert lats["context_ms"] >= 0.0
        assert lats["prompt_ms"] >= 0.0
        assert lats["generation_ms"] >= 0.0
        assert lats["total_ms"] >= 0.0

        # Step 4: Verify separation of indexing from querying (subsequent query does NOT re-index)
        second_request = ClinicalQueryRequest(
            query="What were the coronary angiogram and stenting findings?",
            top_k=3,
        )
        second_resp = orchestrator.query(second_request)
        assert second_resp.generation_status == "success"
        assert len(second_resp.evidence) > 0
        assert second_resp.evidence[0].document_id == "CR-49201"

    @pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY environment variable required for live Gemini test",
    )
    def test_live_gemini_clinical_query_if_key_available(
        self, integration_stack: tuple[ClinicalRAGOrchestrator, FakeLLMProvider], tmp_path: Path
    ) -> None:
        """Verify that RealLLMProvider seamlessly integrates with the doctor-facing orchestrator."""
        real_llm = RealLLMProvider(
            api_key=os.environ["GEMINI_API_KEY"],
            model_name="gemini-2.5-flash",
            temperature=0.0,
        )

        orchestrator, _ = integration_stack
        # Swap in real LLM provider
        orchestrator.llm_provider = real_llm

        clinical_note = (
            "PATIENT: Jane Smith, 54F\n"
            "LABS: Serum Potassium: 5.8 mEq/L (Critical High). BUN: 34 mg/dL. Creatinine: 1.8 mg/dL.\n"
            "ACTION PLAN: Discontinue Spironolactone immediately. Recheck BMP in 48 hours.\n"
        )
        f = tmp_path / "jane_smith_labs.txt"
        f.write_text(clinical_note, encoding="utf-8")

        orchestrator.index_document(ClinicalDocumentUpload(filename="jane_smith_labs.txt", file_path=str(f)))

        req = ClinicalQueryRequest(
            query="What was the serum potassium level and what medication was discontinued?",
        )
        resp = orchestrator.query(req)

        assert resp.generation_status == "success"
        assert "5.8" in resp.answer_text or "Spironolactone" in resp.answer_text
        assert resp.provider_name == "gemini"
        assert len(resp.evidence) > 0
