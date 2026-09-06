"""RerankedRetrievalResult -- enriched retrieval result for second-stage reranking (Phase 4.9).

Inherits from RetrievalResult, preserving all chunk metadata and stage-1 retrieval scores
(dense similarity, sparse BM25, RRF fusion), while adding cross-encoder reranking provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from adaptive_framework.rag.retrieval.retrieval_result import RetrievalResult


@dataclass(frozen=True)
class RerankedRetrievalResult(RetrievalResult):
    """RetrievalResult with multi-stage provenance including cross-encoder reranking.

    Attributes:
        dense_rank: 1-based rank in stage-1 dense retrieval, or None if not retrieved by dense.
        sparse_rank: 1-based rank in stage-1 sparse retrieval, or None if not retrieved by sparse.
        dense_score: Raw dense similarity score, or None.
        sparse_score: Raw BM25 lexical score, or None.
        rrf_score: Stage-1 Reciprocal Rank Fusion score, or None.
        reranker_score: Raw relevance score from the cross-encoder model.
        reranker_rank: 1-based rank after second-stage reranking.
    """

    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float | None = None
    reranker_score: float = 0.0
    reranker_rank: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary including reranking fields."""
        d = super().to_dict()
        d.update(
            {
                "dense_rank": self.dense_rank,
                "sparse_rank": self.sparse_rank,
                "dense_score": self.dense_score,
                "sparse_score": self.sparse_score,
                "rrf_score": self.rrf_score,
                "reranker_score": self.reranker_score,
                "reranker_rank": self.reranker_rank,
            }
        )
        return d

    def __repr__(self) -> str:
        return (
            f"RerankedRetrievalResult("
            f"rank={self.rank}, "
            f"reranker_score={self.reranker_score:.4f}, "
            f"chunk_id='{self.chunk_id}', "
            f"doc='{self.document_id}', "
            f"rrf_score={self.rrf_score})"
        )
