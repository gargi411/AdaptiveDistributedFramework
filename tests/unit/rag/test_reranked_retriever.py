"""Unit tests for RerankedRetriever and retrieval factory (Phase 4.9)."""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from adaptive_framework.rag.interfaces.i_retrieval_engine import IRetrievalEngine
from adaptive_framework.rag.interfaces.i_sparse_retriever import ISparseRetriever
from adaptive_framework.rag.reranking.i_reranker import IReranker
from adaptive_framework.rag.reranking.rerank_result import RerankedRetrievalResult
from adaptive_framework.rag.retrieval.reranked_retriever import RerankedRetriever
from adaptive_framework.rag.retrieval.retrieval_factory import create_retrieval_engine
from adaptive_framework.rag.retrieval.retrieval_result import RetrievalResult


def _make_res(chunk_id: str, rank: int, score: float = 0.8) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        source_file="test.pdf",
        page_numbers=(1,),
        text=f"Sample text {chunk_id}",
        section_heading="Section",
        document_type="clinical_note",
        chunk_index=0,
        score=score,
        rank=rank,
    )


def _make_reranked_res(chunk_id: str, rank: int, score: float = 2.0) -> RerankedRetrievalResult:
    return RerankedRetrievalResult(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        source_file="test.pdf",
        page_numbers=(1,),
        text=f"Sample text {chunk_id}",
        section_heading="Section",
        document_type="clinical_note",
        chunk_index=0,
        score=score,
        rank=rank,
        dense_score=0.8,
        dense_rank=rank,
        reranker_score=score,
        reranker_rank=rank,
    )


class TestRerankedRetriever:
    """Test orchestration, graceful fallback, and metrics of RerankedRetriever."""

    def test_init_validation(self) -> None:
        mock_base = MagicMock(spec=IRetrievalEngine)
        mock_reranker = MagicMock(spec=IReranker)

        with pytest.raises(TypeError, match="base_retriever must not be None"):
            RerankedRetriever(base_retriever=None, reranker=mock_reranker)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="reranker must not be None"):
            RerankedRetriever(base_retriever=mock_base, reranker=None)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="candidate_top_k must be >= 1"):
            RerankedRetriever(mock_base, mock_reranker, candidate_top_k=0)
        with pytest.raises(ValueError, match="final_top_k must be >= 1"):
            RerankedRetriever(mock_base, mock_reranker, final_top_k=0)

    def test_query_validation(self) -> None:
        mock_base = MagicMock(spec=IRetrievalEngine)
        mock_reranker = MagicMock(spec=IReranker)
        retriever = RerankedRetriever(mock_base, mock_reranker)

        with pytest.raises(ValueError, match="empty"):
            retriever.retrieve("")
        with pytest.raises(ValueError, match="empty"):
            retriever.retrieve("   ")
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            retriever.retrieve("valid query", top_k=0)

    def test_successful_two_stage_retrieval(self) -> None:
        mock_base = MagicMock(spec=IRetrievalEngine)
        mock_reranker = MagicMock(spec=IReranker)

        c1 = _make_res("c1", rank=1, score=0.9)
        c2 = _make_res("c2", rank=2, score=0.8)
        mock_base.retrieve.return_value = [c1, c2]

        r1 = _make_reranked_res("c2", rank=1, score=3.5)
        r2 = _make_reranked_res("c1", rank=2, score=1.2)
        mock_reranker.rerank.return_value = [r1, r2]

        retriever = RerankedRetriever(
            base_retriever=mock_base,
            reranker=mock_reranker,
            candidate_top_k=10,
            final_top_k=2,
        )

        results = retriever.retrieve("clinical query", top_k=2)

        mock_base.retrieve.assert_called_once_with(
            query="clinical query",
            top_k=10,
            min_score=0.0,
            metadata_filters=None,
        )
        mock_reranker.rerank.assert_called_once_with(
            query="clinical query",
            candidates=[c1, c2],
            top_k=2,
        )
        assert len(results) == 2
        assert results[0].chunk_id == "c2"
        assert results[1].chunk_id == "c1"

        # Check metrics
        metrics = retriever.get_metrics()
        assert metrics["total_queries"] == 1
        assert metrics["candidate_top_k"] == 10
        assert metrics["final_top_k"] == 2
        assert "base_metrics" in metrics
        assert "reranker_metrics" in metrics

    def test_empty_stage1_candidates(self) -> None:
        mock_base = MagicMock(spec=IRetrievalEngine)
        mock_reranker = MagicMock(spec=IReranker)
        mock_base.retrieve.return_value = []

        retriever = RerankedRetriever(mock_base, mock_reranker)
        results = retriever.retrieve("no match query")

        assert results == []
        mock_reranker.rerank.assert_not_called()

    def test_graceful_fallback_on_reranker_error(self) -> None:
        mock_base = MagicMock(spec=IRetrievalEngine)
        mock_reranker = MagicMock(spec=IReranker)

        c1 = _make_res("c1", rank=1, score=0.9)
        c2 = _make_res("c2", rank=2, score=0.8)
        mock_base.retrieve.return_value = [c1, c2]
        mock_reranker.rerank.side_effect = RuntimeError("CUDA OOM or model crash")

        retriever = RerankedRetriever(
            base_retriever=mock_base,
            reranker=mock_reranker,
        )

        results = retriever.retrieve("clinical query", top_k=2)
        assert len(results) == 2
        assert results[0].chunk_id == "c1"
        assert results[1].chunk_id == "c2"


class TestRetrievalFactoryPhase49:
    """Test retrieval factory with hybrid_reranked strategy."""

    def test_create_hybrid_reranked_engine(self) -> None:
        mock_dense = MagicMock(spec=IRetrievalEngine)
        mock_sparse = MagicMock(spec=ISparseRetriever)
        mock_reranker = MagicMock(spec=IReranker)

        engine = create_retrieval_engine(
            strategy="hybrid_reranked",
            dense_engine=mock_dense,
            sparse_retriever=mock_sparse,
            reranker=mock_reranker,
            candidate_top_k=20,
            final_top_k=5,
        )

        assert isinstance(engine, RerankedRetriever)
        assert engine.candidate_top_k == 20
        assert engine.final_top_k == 5
        assert engine.reranker is mock_reranker

    def test_create_hybrid_reranked_without_sparse_raises(self) -> None:
        mock_dense = MagicMock(spec=IRetrievalEngine)
        mock_reranker = MagicMock(spec=IReranker)

        with pytest.raises(ValueError, match="sparse_retriever must be provided"):
            create_retrieval_engine(
                strategy="hybrid_reranked",
                dense_engine=mock_dense,
                sparse_retriever=None,
                reranker=mock_reranker,
            )
