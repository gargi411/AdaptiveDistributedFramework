"""Unit tests for CrossEncoderReranker (Phase 4.9)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from adaptive_framework.rag.reranking.cross_encoder_reranker import CrossEncoderReranker
from adaptive_framework.rag.reranking.rerank_result import RerankedRetrievalResult
from adaptive_framework.rag.retrieval.hybrid_result import HybridRetrievalResult
from adaptive_framework.rag.retrieval.retrieval_result import RetrievalResult


def _make_res(chunk_id: str, rank: int, score: float = 0.8, text: str = "") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        source_file="test.pdf",
        page_numbers=(1,),
        text=text or f"Sample text for {chunk_id}",
        section_heading="Section",
        document_type="clinical_note",
        chunk_index=0,
        score=score,
        rank=rank,
    )


def _make_hybrid_res(
    chunk_id: str,
    rank: int,
    rrf_score: float = 0.03,
    dense_rank: int | None = 1,
    sparse_rank: int | None = 2,
    text: str = "",
) -> HybridRetrievalResult:
    return HybridRetrievalResult(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        source_file="test.pdf",
        page_numbers=(1,),
        text=text or f"Hybrid sample text for {chunk_id}",
        section_heading="Section",
        document_type="clinical_note",
        chunk_index=0,
        score=rrf_score,
        rank=rank,
        dense_rank=dense_rank,
        sparse_rank=sparse_rank,
        dense_score=0.85 if dense_rank else None,
        sparse_score=12.4 if sparse_rank else None,
        rrf_score=rrf_score,
    )


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker initialization, scoring, ranking, and validation."""

    def test_init_defaults(self) -> None:
        mock_model = MagicMock()
        reranker = CrossEncoderReranker(model=mock_model)
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert reranker.device in ("cpu", "cuda")
        assert reranker.batch_size == 16
        assert reranker.is_initialized

    def test_init_validation(self) -> None:
        with pytest.raises(ValueError, match="model_name must not be empty"):
            CrossEncoderReranker(model_name="")
        with pytest.raises(ValueError, match="model_name must not be empty"):
            CrossEncoderReranker(model_name="   ")
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            CrossEncoderReranker(batch_size=0)

    def test_rerank_empty_query(self) -> None:
        reranker = CrossEncoderReranker(model=MagicMock())
        c1 = _make_res("c1", rank=1)
        with pytest.raises(ValueError, match="Query string must not be empty"):
            reranker.rerank("", [c1])
        with pytest.raises(ValueError, match="Query string must not be empty"):
            reranker.rerank("   ", [c1])

    def test_rerank_top_k_validation(self) -> None:
        reranker = CrossEncoderReranker(model=MagicMock())
        c1 = _make_res("c1", rank=1)
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            reranker.rerank("query", [c1], top_k=0)

    def test_rerank_empty_candidates(self) -> None:
        mock_model = MagicMock()
        reranker = CrossEncoderReranker(model=mock_model)
        results = reranker.rerank("query", [])
        assert results == []
        mock_model.predict.assert_not_called()

    def test_rerank_sorting_and_scores(self) -> None:
        mock_model = MagicMock()
        # 3 candidates: c1, c2, c3. Model assigns scores: 0.2, 0.9, 0.5
        mock_model.predict.return_value = np.array([0.2, 0.9, 0.5])

        reranker = CrossEncoderReranker(model=mock_model)
        c1 = _make_res("c1", rank=1, score=0.8)
        c2 = _make_res("c2", rank=2, score=0.7)
        c3 = _make_res("c3", rank=3, score=0.6)

        results = reranker.rerank("my clinical query", [c1, c2, c3], top_k=3)

        assert len(results) == 3
        # Should be ordered descending: c2 (0.9), c3 (0.5), c1 (0.2)
        assert results[0].chunk_id == "c2"
        assert results[0].rank == 1
        assert pytest.approx(results[0].score, abs=1e-4) == 0.9
        assert pytest.approx(results[0].reranker_score, abs=1e-4) == 0.9
        assert results[0].reranker_rank == 1

        assert results[1].chunk_id == "c3"
        assert results[1].rank == 2
        assert pytest.approx(results[1].score, abs=1e-4) == 0.5

        assert results[2].chunk_id == "c1"
        assert results[2].rank == 3
        assert pytest.approx(results[2].score, abs=1e-4) == 0.2

    def test_rerank_preserves_hybrid_provenance(self) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([2.5])

        reranker = CrossEncoderReranker(model=mock_model)
        cand = _make_hybrid_res("c1", rank=1, rrf_score=0.032, dense_rank=2, sparse_rank=1)

        results = reranker.rerank("query", [cand], top_k=1)
        assert len(results) == 1
        r = results[0]

        assert isinstance(r, RerankedRetrievalResult)
        assert r.chunk_id == "c1"
        assert r.dense_rank == 2
        assert r.sparse_rank == 1
        assert pytest.approx(r.rrf_score, abs=1e-4) == 0.032
        assert pytest.approx(r.score, abs=1e-4) == 2.5
        assert r.rank == 1
        assert pytest.approx(r.reranker_score, abs=1e-4) == 2.5
        assert r.reranker_rank == 1

    def test_rerank_batching(self) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        reranker = CrossEncoderReranker(model=mock_model, batch_size=2)
        candidates = [_make_res(f"c{i}", rank=i + 1) for i in range(5)]

        results = reranker.rerank("query", candidates, top_k=5)
        assert len(results) == 5
        mock_model.predict.assert_called_once()
        _, kwargs = mock_model.predict.call_args
        assert kwargs.get("batch_size") == 2
        # Highest score was 5.0 (c4)
        assert results[0].chunk_id == "c4"
        assert results[0].rank == 1

    def test_rerank_top_k_truncation(self) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.1, 0.4, 0.9, 0.2])

        reranker = CrossEncoderReranker(model=mock_model)
        candidates = [_make_res(f"c{i}", rank=i + 1) for i in range(4)]

        results = reranker.rerank("query", candidates, top_k=2)
        assert len(results) == 2
        assert results[0].chunk_id == "c2"  # score 0.9
        assert results[1].chunk_id == "c1"  # score 0.4

    def test_lazy_loading(self) -> None:
        with patch("sentence_transformers.CrossEncoder") as mock_ce_cls:
            mock_inst = MagicMock()
            mock_ce_cls.return_value = mock_inst
            mock_inst.predict.return_value = np.array([1.0])

            reranker = CrossEncoderReranker(model_name="test-model", device="cpu")
            assert not reranker.is_initialized

            # Calling rerank triggers _ensure_model
            c = _make_res("c1", rank=1)
            results = reranker.rerank("query", [c])
            assert reranker.is_initialized
            mock_ce_cls.assert_called_once_with("test-model", device="cpu")
            assert len(results) == 1
