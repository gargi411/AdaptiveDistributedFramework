"""Unit tests for Phase 4.10 Reranking Optimization and Adaptive Policy."""

from __future__ import annotations

from unittest.mock import MagicMock
import numpy as np
import pytest

from adaptive_framework.rag.interfaces.i_retrieval_engine import IRetrievalEngine
from adaptive_framework.rag.reranking.adaptive_policy import AdaptiveCandidateDepthPolicy
from adaptive_framework.rag.reranking.cross_encoder_reranker import CrossEncoderReranker
from adaptive_framework.rag.retrieval.hybrid_result import HybridRetrievalResult
from adaptive_framework.rag.retrieval.reranked_retriever import RerankedRetriever
from adaptive_framework.rag.retrieval.retrieval_result import RetrievalResult


def _make_res(chunk_id: str, rank: int, score: float = 0.8) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        source_file="test.pdf",
        page_numbers=(1,),
        text=f"Sample text for {chunk_id}",
        section_heading="Section",
        document_type="clinical_note",
        chunk_index=0,
        score=score,
        rank=rank,
    )


def _make_hybrid_res(
    chunk_id: str,
    rank: int,
    rrf_score: float,
    dense_rank: int | None = None,
    sparse_rank: int | None = None,
) -> HybridRetrievalResult:
    return HybridRetrievalResult(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        source_file="test.pdf",
        page_numbers=(1,),
        text=f"Sample text for {chunk_id}",
        section_heading="Section",
        document_type="clinical_note",
        chunk_index=0,
        score=rrf_score,
        rank=rank,
        dense_rank=dense_rank,
        sparse_rank=sparse_rank,
        dense_score=0.9 if dense_rank else None,
        sparse_score=10.0 if sparse_rank else None,
        rrf_score=rrf_score,
    )


class TestCandidateDepthAndBatchOverrides:
    """Tests for dynamic candidate depth and batch size parameter overrides."""

    def test_candidate_depth_override(self) -> None:
        mock_base = MagicMock(spec=IRetrievalEngine)
        mock_reranker = MagicMock()
        mock_base.retrieve.return_value = [_make_res(f"c{i}", i + 1) for i in range(10)]
        mock_reranker.rerank.return_value = []

        retriever = RerankedRetriever(
            base_retriever=mock_base,
            reranker=mock_reranker,
            candidate_top_k=20,
            final_top_k=5,
        )

        retriever.retrieve("clinical query", candidate_top_k=10)
        mock_base.retrieve.assert_called_once_with(
            query="clinical query",
            top_k=10,
            min_score=0.0,
            metadata_filters=None,
        )

    def test_candidate_depth_invalid(self) -> None:
        mock_base = MagicMock(spec=IRetrievalEngine)
        mock_reranker = MagicMock()
        retriever = RerankedRetriever(base_retriever=mock_base, reranker=mock_reranker)

        with pytest.raises(ValueError, match="candidate_top_k must be >= 1"):
            retriever.retrieve("clinical query", candidate_top_k=0)

    def test_batch_size_override(self) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1.0, 2.0])

        reranker = CrossEncoderReranker(model=mock_model, batch_size=16)
        candidates = [_make_res("c1", 1), _make_res("c2", 2)]

        reranker.rerank("query", candidates, batch_size=4)
        mock_model.predict.assert_called_once_with(
            [("query", "Sample text for c1"), ("query", "Sample text for c2")],
            batch_size=4,
        )

    def test_batch_size_invalid(self) -> None:
        mock_model = MagicMock()
        reranker = CrossEncoderReranker(model=mock_model, batch_size=16)
        candidates = [_make_res("c1", 1)]

        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            reranker.rerank("query", candidates, batch_size=0)

    def test_warmup(self) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5])

        reranker = CrossEncoderReranker(model=mock_model)
        cold_start_ms = reranker.warmup()

        assert cold_start_ms >= 0.0
        assert reranker.is_initialized
        mock_model.predict.assert_called_once_with(
            [("warmup query", "warmup passage text")],
            batch_size=1,
        )


class TestAdaptiveCandidateDepthPolicy:
    """Tests for signal computation, decision boundaries, and ground-truth isolation."""

    def test_init_validation(self) -> None:
        with pytest.raises(ValueError, match="min_depth must be >= 1"):
            AdaptiveCandidateDepthPolicy(min_depth=0)
        with pytest.raises(ValueError, match="default_depth.*must be >= min_depth"):
            AdaptiveCandidateDepthPolicy(min_depth=10, default_depth=5)
        with pytest.raises(ValueError, match="max_depth.*must be >= default_depth"):
            AdaptiveCandidateDepthPolicy(default_depth=15, max_depth=10)

    def test_empty_candidates_handling(self) -> None:
        policy = AdaptiveCandidateDepthPolicy(min_depth=5, default_depth=10, max_depth=20)
        assert policy.determine_depth([]) == 5
        signals = policy.compute_signals([])
        assert signals["candidate_count"] == 0
        assert signals["dense_sparse_agreement"] is False

    def test_high_confidence_selects_min_depth(self) -> None:
        # Candidate 1 has high dual-engine consensus (dense=1, sparse=1) and large score gap
        candidates = [
            _make_hybrid_res("c1", 1, rrf_score=0.032, dense_rank=1, sparse_rank=1),
            _make_hybrid_res("c2", 2, rrf_score=0.025, dense_rank=5, sparse_rank=6),
            _make_hybrid_res("c3", 3, rrf_score=0.020, dense_rank=8, sparse_rank=10),
            _make_hybrid_res("c4", 4, rrf_score=0.018, dense_rank=12, sparse_rank=12),
            _make_hybrid_res("c5", 5, rrf_score=0.015, dense_rank=15, sparse_rank=15),
        ]
        policy = AdaptiveCandidateDepthPolicy(
            min_depth=5,
            default_depth=10,
            max_depth=20,
            high_confidence_gap_threshold=0.005,
        )
        signals = policy.compute_signals(candidates)
        assert signals["has_consensus_top1"] is True
        assert signals["score_gap_1_5"] >= 0.005
        assert policy.determine_depth(candidates) == 5

    def test_moderate_confidence_selects_default_depth(self) -> None:
        # Candidate 1 lacks top-1 consensus, but Candidate 2 has consensus in top-3
        candidates = [
            _make_hybrid_res("c1", 1, rrf_score=0.030, dense_rank=1, sparse_rank=15),
            _make_hybrid_res("c2", 2, rrf_score=0.028, dense_rank=2, sparse_rank=2),
            _make_hybrid_res("c3", 3, rrf_score=0.026, dense_rank=10, sparse_rank=10),
            _make_hybrid_res("c4", 4, rrf_score=0.025, dense_rank=12, sparse_rank=12),
            _make_hybrid_res("c5", 5, rrf_score=0.024, dense_rank=15, sparse_rank=15),
        ] + [_make_hybrid_res(f"c{i}", i, 0.02) for i in range(6, 16)]
        policy = AdaptiveCandidateDepthPolicy(min_depth=5, default_depth=10, max_depth=20)
        signals = policy.compute_signals(candidates)
        assert signals["has_consensus_top1"] is False
        assert signals["dense_sparse_agreement"] is True
        assert policy.determine_depth(candidates) == 10

    def test_low_confidence_selects_max_depth(self) -> None:
        # None of top candidates have dual consensus (disagreement between dense and sparse)
        candidates = [
            _make_hybrid_res("c1", 1, rrf_score=0.016, dense_rank=1, sparse_rank=None),
            _make_hybrid_res("c2", 2, rrf_score=0.016, dense_rank=None, sparse_rank=1),
            _make_hybrid_res("c3", 3, rrf_score=0.015, dense_rank=2, sparse_rank=None),
            _make_hybrid_res("c4", 4, rrf_score=0.015, dense_rank=None, sparse_rank=2),
            _make_hybrid_res("c5", 5, rrf_score=0.014, dense_rank=3, sparse_rank=None),
        ] + [_make_hybrid_res(f"c{i}", i, 0.01) for i in range(6, 21)]

        policy = AdaptiveCandidateDepthPolicy(min_depth=5, default_depth=10, max_depth=20)
        signals = policy.compute_signals(candidates)
        assert signals["has_consensus_top1"] is False
        assert signals["dense_sparse_agreement"] is False
        assert policy.determine_depth(candidates) == 20

    def test_ground_truth_isolation(self) -> None:
        # Ensure policy operates purely on candidate instances without ground-truth parameters
        policy = AdaptiveCandidateDepthPolicy()
        candidates = [_make_res(f"c{i}", i) for i in range(1, 10)]
        depth = policy.determine_depth(candidates)
        assert isinstance(depth, int)
        assert 5 <= depth <= 20
