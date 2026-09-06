"""End-to-end integration test for Phase 4.5 RAG Answer Generation pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock
import numpy as np
import pytest

from adaptive_framework.models.chunk import Chunk
from adaptive_framework.rag.embedding.bge_embedder import BGEEmbedder
from adaptive_framework.rag.generation.context_builder import ContextBuilder
from adaptive_framework.rag.generation.fake_llm_provider import FakeLLMProvider
from adaptive_framework.rag.generation.prompt_builder import PromptBuilder
from adaptive_framework.rag.generation.rag_service import RAGService
from adaptive_framework.rag.models.embedding import Embedding
from adaptive_framework.rag.retrieval.retrieval_engine import RetrievalEngine
from adaptive_framework.rag.vector_store.faiss_manager import FAISSManager


def _make_deterministic_vector(text: str, dim: int = 64) -> np.ndarray:
    """Create a reproducible pseudo-embedding vector for text."""
    seed = abs(hash(text)) % (2**32)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


@pytest.mark.integration
class TestRAGPipelineIntegration:
    """Integration test verifying full Query -> Retrieval -> Context -> Prompt -> LLM -> Answer pipeline."""

    @pytest.fixture
    def mock_embedder(self) -> MagicMock:
        dim = 64
        embedder = MagicMock(spec=BGEEmbedder)
        embedder.embedding_dim = dim
        embedder.get_model_name.return_value = "BAAI/bge-large-en-v1.5"

        def embed_query_side_effect(text: str) -> list[float]:
            if "diabetes" in text.lower():
                vec = _make_deterministic_vector(
                    "Type 2 diabetes mellitus is managed with metformin as first-line pharmacotherapy alongside lifestyle modifications.",
                    dim=dim,
                )
            else:
                vec = _make_deterministic_vector(text, dim=dim)
            return [float(x) for x in vec]

        embedder.embed_query.side_effect = embed_query_side_effect
        return embedder

    @pytest.fixture
    def populated_faiss_manager(self) -> tuple[FAISSManager, list[Chunk]]:
        """Index a set of synthetic medical documents into FAISS."""
        dim = 64
        manager = FAISSManager(dim=dim, model_name="BAAI/bge-large-en-v1.5")

        docs = [
            {
                "chunk_id": "c1",
                "document_id": "cardio_guideline",
                "source_file": "cardio_guideline.pdf",
                "page_numbers": (1, 2),
                "text": "Hypertension is defined as systolic blood pressure of 130 mmHg or higher or diastolic BP of 80 mmHg or higher.",
                "section_heading": "Diagnostic Criteria",
                "document_type": "clinical_guideline",
                "chunk_index": 0,
            },
            {
                "chunk_id": "c2",
                "document_id": "respiratory_guideline",
                "source_file": "respiratory_guideline.pdf",
                "page_numbers": (3,),
                "text": "Chronic obstructive pulmonary disease is characterized by persistent airflow limitation and respiratory symptoms.",
                "section_heading": "Pathophysiology",
                "document_type": "clinical_guideline",
                "chunk_index": 0,
            },
            {
                "chunk_id": "c3",
                "document_id": "endocrine_report",
                "source_file": "endocrine_report.pdf",
                "page_numbers": (7, 8),
                "text": "Type 2 diabetes mellitus is managed with metformin as first-line pharmacotherapy alongside lifestyle modifications.",
                "section_heading": "Management",
                "document_type": "treatment_protocol",
                "chunk_index": 0,
            },
        ]

        chunks = [
            Chunk(
                chunk_id=d["chunk_id"],
                document_id=d["document_id"],
                source_file=d["source_file"],
                chunk_index=d["chunk_index"],
                page_numbers=d["page_numbers"],
                text=d["text"],
                section_heading=d["section_heading"],
                document_type=d["document_type"],
                character_count=len(d["text"]),
                word_count=len(d["text"].split()),
                start_char_in_page=0,
                end_char_in_page=len(d["text"]),
            )
            for d in docs
        ]

        embeddings = [
            Embedding(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                vector=tuple(float(x) for x in _make_deterministic_vector(c.text, dim=dim)),
                model_name="BAAI/bge-large-en-v1.5",
                embedding_time_seconds=0.001,
                created_at="2026-09-06T12:00:00Z",
            )
            for c in chunks
        ]

        manager.add_embeddings_with_chunks(embeddings, chunks)
        return manager, chunks

    def test_end_to_end_rag_pipeline(
        self,
        mock_embedder: MagicMock,
        populated_faiss_manager: tuple[FAISSManager, list[Chunk]],
    ) -> None:
        manager, chunks = populated_faiss_manager

        retrieval_engine = RetrievalEngine(
            embedding_provider=mock_embedder,
            faiss_manager=manager,
            default_top_k=3,
        )
        context_builder = ContextBuilder(max_chunks=3, max_context_chars=3000)
        prompt_builder = PromptBuilder()
        llm_provider = FakeLLMProvider(model_name="fake-integration-v1")

        rag_service = RAGService(
            retrieval_engine=retrieval_engine,
            context_builder=context_builder,
            prompt_builder=prompt_builder,
            llm_provider=llm_provider,
        )

        # Execute query
        query = "What is the first-line medication for type 2 diabetes?"
        answer = rag_service.answer(query)

        # Assertions
        assert answer.generation_status == "ok"
        assert answer.query == query
        assert len(answer.evidence) > 0
        assert len(answer.source_references) > 0

        # Provenance verification
        for ref in answer.source_references:
            assert ref.document_id in {"cardio_guideline", "respiratory_guideline", "endocrine_report"}
            assert ref.source_file.endswith(".pdf")
            assert len(ref.page_numbers) > 0

        # LLM response must reflect evidence
        assert "Based on the retrieved evidence" in answer.answer_text
        assert "Sources:" in answer.answer_text

        # All latencies recorded and non-negative
        assert answer.retrieval_latency_s >= 0.0
        assert answer.context_latency_s >= 0.0
        assert answer.prompt_latency_s >= 0.0
        assert answer.generation_latency_s >= 0.0
        assert answer.total_latency_s >= 0.0

        # Verify serialization
        data = answer.to_dict()
        assert data["generation_status"] == "ok"
        assert len(data["evidence"]) > 0
        assert len(data["source_references"]) > 0

    def test_end_to_end_empty_results_scenario(
        self,
        mock_embedder: MagicMock,
    ) -> None:
        dim = 64
        empty_manager = FAISSManager(dim=dim, model_name="BAAI/bge-large-en-v1.5")

        retrieval_engine = RetrievalEngine(
            embedding_provider=mock_embedder,
            faiss_manager=empty_manager,
            default_top_k=3,
        )
        rag_service = RAGService(
            retrieval_engine=retrieval_engine,
            context_builder=ContextBuilder(),
            prompt_builder=PromptBuilder(),
            llm_provider=FakeLLMProvider(),
        )

        answer = rag_service.answer("A query with no index entries")
        assert answer.generation_status == "no_results"
        assert len(answer.evidence) == 0
        assert len(answer.source_references) == 0
        assert "No relevant documents" in answer.answer_text
