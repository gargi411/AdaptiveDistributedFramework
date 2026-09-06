"""HybridRetrievalResult -- enriched retrieval result for hybrid search (Phase 4.8).

Inherits from RetrievalResult, preserving all chunk metadata and fields,
while adding explicit provenance for dense vs sparse ranking and fusion scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from adaptive_framework.rag.retrieval.retrieval_result import RetrievalResult


@dataclass(frozen=True)
class HybridRetrievalResult(RetrievalResult):
    """RetrievalResult with additional dense, sparse, and fusion provenance.

    Attributes:
        dense_rank: 1-based rank in the dense retrieval list, or None if not present.
        sparse_rank: 1-based rank in the sparse retrieval list, or None if not present.
        dense_score: Raw dense similarity score, or None if not retrieved by dense.
        sparse_score: Raw BM25 lexical score, or None if not retrieved by sparse.
        rrf_score: Computed Reciprocal Rank Fusion score.
    """

    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary including hybrid fields."""
        d = super().to_dict()
        d.update(
            {
                "dense_rank": self.dense_rank,
                "sparse_rank": self.sparse_rank,
                "dense_score": self.dense_score,
                "sparse_score": self.sparse_score,
                "rrf_score": self.rrf_score,
            }
        )
        return d

    def __repr__(self) -> str:
        return (
            f"HybridRetrievalResult("
            f"rank={self.rank}, "
            f"rrf_score={self.rrf_score:.5f}, "
            f"chunk_id='{self.chunk_id}', "
            f"dense_rank={self.dense_rank}, "
            f"sparse_rank={self.sparse_rank})"
        )
