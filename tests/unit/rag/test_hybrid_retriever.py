"""Unit tests for HybridRetriever and retrieval factory (Phase 4.8)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from adaptive_framework.rag.interfaces.i_retrieval_engine import IRetrievalEngine
from adaptive_framework.rag.interfaces.i_sparse_retriever import ISparseRetriever
from adaptive_framework.rag.retrieval.hybrid_retriever import HybridRetriever
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


class TestHybridRetriever:
    """Test orchestration, graceful failure, and metrics of HybridRetriever."""

    def test_init_validation(self) -> None:
        mock_dense = MagicMock(spec=IRetrievalEngine)
        mock_sparse = MagicMock(spec=ISparseRetriever)

        with pytest.raises(TypeError, match="dense_engine must not be None"):
            HybridRetriever(dense_engine=None, sparse_retriever=mock_sparse)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="sparse_retriever must not be None"):
            HybridRetriever(dense_engine=mock_dense, sparse_retriever=None)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="dense_top_k must be >= 1"):
            HybridRetriever(mock_dense, mock_sparse, dense_top_k=0)
        with pytest.raises(ValueError, match="sparse_top_k must be >= 1"):
            HybridRetriever(mock_dense, mock_sparse, sparse_top_k=0)
        with pytest.raises(ValueError, match="final_top_k must be >= 1"):
            HybridRetriever(mock_dense, mock_sparse, final_top_k=0)
        with pytest.raises(ValueError, match="rrf_k must be >= 1"):
            HybridRetriever(mock_dense, mock_sparse, rrf_k=0)

    def test_query_validation(self) -> None:
        mock_dense = MagicMock(spec=IRetrievalEngine)
        mock_sparse = MagicMock(spec=ISparseRetriever)
        retriever = HybridRetriever(mock_dense, mock_sparse)

        with pytest.raises(ValueError, match="empty"):
            retriever.retrieve("")
        with pytest.raises(ValueError, match="empty"):
            retriever.retrieve("   ")
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            retriever.retrieve("valid query", top_k=0)

    def test_successful_hybrid_retrieval(self) -> None:
        mock_dense = MagicMock(spec=IRetrievalEngine)
        mock_sparse = MagicMock(spec=ISparseRetriever)

        mock_dense.retrieve.return_value = [_make_res("c1", rank=1, score=0.9)]
        mock_sparse.retrieve.return_value = [_make_res("c2", rank=1, score=5.0)]

        retriever = HybridRetriever(
            dense_engine=mock_dense,
            sparse_retriever=mock_sparse,
            dense_top_k=10,
            sparse_top_k=10,
            final_top_k=5,
            rrf_k=60,
        )

        results = retriever.retrieve("cardiac troponin")
        assert len(results) == 2
        mock_dense.retrieve.assert_called_once_with(
            query="cardiac troponin", top_k=10, min_score=0.0, metadata_filters=None
        )
        mock_sparse.retrieve.assert_called_once_with(
            query="cardiac troponin", top_k=10, min_score=0.0
        )

    def test_dense_failure_fallback(self) -> None:
        mock_dense = MagicMock(spec=IRetrievalEngine)
        mock_sparse = MagicMock(spec=ISparseRetriever)

        mock_dense.retrieve.side_effect = RuntimeError("FAISS search fault")
        mock_sparse.retrieve.return_value = [_make_res("c2", rank=1, score=4.0)]

        retriever = HybridRetriever(mock_dense, mock_sparse)
        results = retriever.retrieve("hypertension")

        # Graceful degradation: sparse results should still be returned
        assert len(results) == 1
        assert results[0].chunk_id == "c2"

    def test_sparse_failure_fallback(self) -> None:
        mock_dense = MagicMock(spec=IRetrievalEngine)
        mock_sparse = MagicMock(spec=ISparseRetriever)

        mock_dense.retrieve.return_value = [_make_res("c1", rank=1, score=0.88)]
        mock_sparse.retrieve.side_effect = RuntimeError("BM25 error")

        retriever = HybridRetriever(mock_dense, mock_sparse)
        results = retriever.retrieve("diabetes")

        assert len(results) == 1
        assert results[0].chunk_id == "c1"

    def test_both_empty(self) -> None:
        mock_dense = MagicMock(spec=IRetrievalEngine)
        mock_sparse = MagicMock(spec=ISparseRetriever)

        mock_dense.retrieve.return_value = []
        mock_sparse.retrieve.return_value = []

        retriever = HybridRetriever(mock_dense, mock_sparse)
        results = retriever.retrieve("nonexistent condition")
        assert results == []

    def test_metrics_aggregation(self) -> None:
        mock_dense = MagicMock(spec=IRetrievalEngine)
        mock_sparse = MagicMock(spec=ISparseRetriever)

        mock_dense.retrieve.return_value = []
        mock_sparse.retrieve.return_value = []
        mock_dense.get_metrics.return_value = {"dense_queries": 1}
        mock_sparse.get_metrics.return_value = {"sparse_queries": 1}

        retriever = HybridRetriever(mock_dense, mock_sparse)
        retriever.retrieve("query")

        metrics = retriever.get_metrics()
        assert metrics["total_queries"] == 1
        assert "dense_metrics" in metrics
        assert "sparse_metrics" in metrics


class TestRetrievalFactory:
    """Test retrieval factory strategy switching."""

    def test_create_dense(self) -> None:
        mock_dense = MagicMock(spec=IRetrievalEngine)
        engine = create_retrieval_engine(strategy="dense", dense_engine=mock_dense)
        assert engine is mock_dense

    def test_create_hybrid(self) -> None:
        mock_dense = MagicMock(spec=IRetrievalEngine)
        mock_sparse = MagicMock(spec=ISparseRetriever)
        engine = create_retrieval_engine(
            strategy="hybrid",
            dense_engine=mock_dense,
            sparse_retriever=mock_sparse,
        )
        assert isinstance(engine, HybridRetriever)

    def test_create_hybrid_without_sparse_raises(self) -> None:
        mock_dense = MagicMock(spec=IRetrievalEngine)
        with pytest.raises(ValueError, match="sparse_retriever must be provided"):
            create_retrieval_engine(strategy="hybrid", dense_engine=mock_dense, sparse_retriever=None)

    def test_unknown_strategy_raises(self) -> None:
        mock_dense = MagicMock(spec=IRetrievalEngine)
        with pytest.raises(ValueError, match="Unsupported retrieval strategy"):
            create_retrieval_engine(strategy="quantum", dense_engine=mock_dense)
