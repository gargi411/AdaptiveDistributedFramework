"""Unit tests for Reciprocal Rank Fusion (Phase 4.8)."""

from __future__ import annotations

import pytest

from adaptive_framework.rag.retrieval.hybrid_result import HybridRetrievalResult
from adaptive_framework.rag.retrieval.retrieval_result import RetrievalResult
from adaptive_framework.rag.retrieval.rrf import reciprocal_rank_fusion


def _make_res(chunk_id: str, rank: int, score: float = 0.8) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        source_file="test.pdf",
        page_numbers=(1,),
        text=f"Text for {chunk_id}",
        section_heading="Section",
        document_type="note",
        chunk_index=0,
        score=score,
        rank=rank,
    )


class TestReciprocalRankFusion:
    """Mathematical and structural tests for RRF."""

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="rrf_k must be >= 1"):
            reciprocal_rank_fusion([], [], rrf_k=0)
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            reciprocal_rank_fusion([], [], top_k=0)

    def test_empty_inputs(self) -> None:
        assert reciprocal_rank_fusion([], [], rrf_k=60, top_k=5) == []

    def test_dense_only(self) -> None:
        dense = [_make_res("A", rank=1, score=0.9), _make_res("B", rank=2, score=0.8)]
        fused = reciprocal_rank_fusion(dense, [], rrf_k=60, top_k=5)

        assert len(fused) == 2
        assert fused[0].chunk_id == "A"
        assert fused[0].rank == 1
        assert pytest.approx(fused[0].score, 1e-6) == 1.0 / 61.0
        assert fused[0].dense_rank == 1
        assert fused[0].sparse_rank is None
        assert fused[0].dense_score == 0.9
        assert fused[0].sparse_score is None

        assert fused[1].chunk_id == "B"
        assert fused[1].rank == 2
        assert pytest.approx(fused[1].score, 1e-6) == 1.0 / 62.0

    def test_sparse_only(self) -> None:
        sparse = [_make_res("X", rank=1, score=5.4), _make_res("Y", rank=2, score=3.2)]
        fused = reciprocal_rank_fusion([], sparse, rrf_k=60, top_k=5)

        assert len(fused) == 2
        assert fused[0].chunk_id == "X"
        assert fused[0].rank == 1
        assert pytest.approx(fused[0].score, 1e-6) == 1.0 / 61.0
        assert fused[0].sparse_rank == 1
        assert fused[0].dense_rank is None
        assert fused[0].sparse_score == 5.4

    def test_manual_calculation_verification(self) -> None:
        """Verify exact mathematical values from Phase 4.8 specification.

        Dense:  A (rank 1), B (rank 2), C (rank 3)
        Sparse: C (rank 1), A (rank 2), D (rank 3)
        rrf_k = 60
        Expected:
            RRF(A) = 1/61 + 1/62 = 0.0163934 + 0.0161290 = 0.0325224
            RRF(B) = 1/62        = 0.0161290
            RRF(C) = 1/63 + 1/61 = 0.0158730 + 0.0163934 = 0.0322664
            RRF(D) = 1/63        = 0.0158730
        Order: A (1), C (2), B (3), D (4)
        """
        dense = [
            _make_res("A", rank=1, score=0.95),
            _make_res("B", rank=2, score=0.85),
            _make_res("C", rank=3, score=0.75),
        ]
        sparse = [
            _make_res("C", rank=1, score=12.0),
            _make_res("A", rank=2, score=8.0),
            _make_res("D", rank=3, score=4.0),
        ]

        fused = reciprocal_rank_fusion(dense, sparse, rrf_k=60, top_k=10)

        assert len(fused) == 4
        assert [f.chunk_id for f in fused] == ["A", "C", "B", "D"]

        expected_a = (1.0 / 61.0) + (1.0 / 62.0)
        expected_c = (1.0 / 63.0) + (1.0 / 61.0)
        expected_b = 1.0 / 62.0
        expected_d = 1.0 / 63.0

        assert pytest.approx(fused[0].score, 1e-6) == expected_a
        assert pytest.approx(fused[1].score, 1e-6) == expected_c
        assert pytest.approx(fused[2].score, 1e-6) == expected_b
        assert pytest.approx(fused[3].score, 1e-6) == expected_d

        assert fused[0].dense_rank == 1
        assert fused[0].sparse_rank == 2
        assert fused[1].dense_rank == 3
        assert fused[1].sparse_rank == 1

    def test_top_k_truncation(self) -> None:
        dense = [_make_res(f"c_{i}", rank=i) for i in range(1, 10)]
        sparse = [_make_res(f"c_{i}", rank=i) for i in range(1, 10)]

        fused = reciprocal_rank_fusion(dense, sparse, rrf_k=60, top_k=3)
        assert len(fused) == 3
        assert [f.rank for f in fused] == [1, 2, 3]

    def test_configurable_k(self) -> None:
        dense = [_make_res("A", rank=1)]
        fused_60 = reciprocal_rank_fusion(dense, [], rrf_k=60)
        fused_20 = reciprocal_rank_fusion(dense, [], rrf_k=20)

        assert pytest.approx(fused_60[0].score, 1e-6) == 1.0 / 61.0
        assert pytest.approx(fused_20[0].score, 1e-6) == 1.0 / 21.0
        assert fused_20[0].score > fused_60[0].score
