"""Integration tests for Reranked Retrieval with Hybrid Retriever, Cross-Encoder, and RAG service (Phase 4.9)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from adaptive_framework.models.chunk import Chunk
from adaptive_framework.rag.embedding.bge_embedder import BGEEmbedder
from adaptive_framework.rag.evaluation.dataset import DEFAULT_GROUND_TRUTH_CASES
from adaptive_framework.rag.evaluation.evaluator import RAGEvaluator
from adaptive_framework.rag.generation.context_builder import ContextBuilder
from adaptive_framework.rag.generation.fake_llm_provider import FakeLLMProvider
from adaptive_framework.rag.generation.prompt_builder import PromptBuilder
from adaptive_framework.rag.generation.rag_service import RAGService
from adaptive_framework.rag.models.embedding import Embedding
from adaptive_framework.rag.reranking.cross_encoder_reranker import CrossEncoderReranker
from adaptive_framework.rag.retrieval.bm25_retriever import BM25Retriever
from adaptive_framework.rag.retrieval.hybrid_retriever import HybridRetriever
from adaptive_framework.rag.retrieval.reranked_retriever import RerankedRetriever
from adaptive_framework.rag.retrieval.retrieval_engine import RetrievalEngine
from adaptive_framework.rag.vector_store.faiss_manager import FAISSManager


def _make_vector(text: str, dim: int = 1024) -> tuple[float, ...]:
    seed = abs(hash(text)) % (2**32)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return tuple(float(x) for x in vec)


def _make_chunk(chunk_id: str, doc_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        source_file="synthetic_notes.json",
        chunk_index=0,
        page_numbers=(1,),
        text=text,
        section_heading="Discharge Medications",
        document_type="discharge_summary",
        character_count=len(text),
        word_count=len(text.split()),
        start_char_in_page=0,
        end_char_in_page=len(text),
    )


class TestRerankedRetrievalIntegration:
    """Integration test suite for cross-encoder reranked search and downstream RAG pipeline."""

    @pytest.fixture
    def indexed_reranked_retriever(self) -> RerankedRetriever:
        dim = 1024
        chunks = [
            _make_chunk(
                "chk_synth_0001",
                "clinical_note_0001",
                "Discharge Medications: Metformin 500mg BD, Glimepiride 1mg OD. Patient diagnosed with T2DM.",
            ),
            _make_chunk(
                "chk_synth_0002",
                "clinical_note_0002",
                "Discharge summary for CAD with STEMI. Patient prescribed Aspirin 75mg and Clopidogrel 75mg.",
            ),
            _make_chunk(
                "chk_synth_0003",
                "clinical_note_0003",
                "Patient with acute appendicitis underwent laparoscopic appendectomy. Prescribed Cefixime 200mg.",
            ),
        ]

        # 1. FAISS Manager + RetrievalEngine
        manager = FAISSManager(dim=dim, index_type="flat")
        embeddings = [
            Embedding(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                vector=_make_vector(c.text, dim=dim),
                model_name="BAAI/bge-large-en-v1.5",
                embedding_time_seconds=0.001,
                created_at="2026-09-06T12:00:00+00:00",
            )
            for c in chunks
        ]
        manager.add_embeddings(embeddings)
        manager.register_chunks(chunks)

        embedder = MagicMock(spec=BGEEmbedder)
        embedder.embedding_dim = dim
        embedder.embed_query.side_effect = lambda q: np.array(_make_vector(q, dim=dim), dtype=np.float32)

        dense_engine = RetrievalEngine(embedding_provider=embedder, faiss_manager=manager, default_top_k=3)

        # 2. BM25
        bm25 = BM25Retriever(k1=1.5, b=0.75)
        bm25.index_chunks(chunks)

        # 3. Hybrid retriever
        hybrid = HybridRetriever(
            dense_engine=dense_engine,
            sparse_retriever=bm25,
            dense_top_k=3,
            sparse_top_k=3,
            final_top_k=3,
            rrf_k=60,
        )

        # 4. Cross-Encoder reranker (mocked model scoring by keyword relevance for testing)
        mock_ce = MagicMock()
        def _mock_predict(pairs: list[tuple[str, str]]) -> np.ndarray:
            scores = []
            for query, text in pairs:
                # Assign higher score if any word matches
                q_words = set(query.lower().split())
                t_words = set(text.lower().split())
                overlap = len(q_words & t_words)
                scores.append(float(overlap) * 2.0)
            return np.array(scores, dtype=np.float32)

        mock_ce.predict.side_effect = _mock_predict
        reranker = CrossEncoderReranker(model=mock_ce, batch_size=4)

        # 5. RerankedRetriever
        return RerankedRetriever(
            base_retriever=hybrid,
            reranker=reranker,
            candidate_top_k=3,
            final_top_k=2,
        )

    def test_end_to_end_reranked_evaluation(self, indexed_reranked_retriever: RerankedRetriever) -> None:
        """Verify RerankedRetriever integrates seamlessly with RAGEvaluator."""
        evaluator = RAGEvaluator(retrieval_engine=indexed_reranked_retriever, k_values=(1, 2))
        cases = DEFAULT_GROUND_TRUTH_CASES[:2]

        report = evaluator.evaluate(cases)
        assert report.query_count == 2
        assert 1 in report.doc_metrics["precision"]
        assert 2 in report.doc_metrics["precision"]
        assert report.latency_stats_ms["mean"] > 0.0

    def test_rag_pipeline_compatibility(self, indexed_reranked_retriever: RerankedRetriever) -> None:
        """Verify reranked results flow through ContextBuilder, PromptBuilder, and RAGService."""
        context_builder = ContextBuilder(max_chunks=2, max_context_chars=4000)
        prompt_builder = PromptBuilder()
        fake_llm = FakeLLMProvider()

        rag_service = RAGService(
            retrieval_engine=indexed_reranked_retriever,
            context_builder=context_builder,
            prompt_builder=prompt_builder,
            llm_provider=fake_llm,
        )

        answer = rag_service.answer("What medications are prescribed for T2DM diabetes?")
        assert answer.answer_text is not None
        assert len(answer.source_references) > 0
        assert answer.model_name == "fake-llm-v1"
        assert answer.num_retrieved <= 2
        assert answer.generation_status == "ok"
