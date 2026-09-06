"""IHybridRetriever -- abstract interface for hybrid retrieval (Phase 4.8).

Combines dense semantic retrieval and sparse lexical retrieval using
rank-level fusion strategies like Reciprocal Rank Fusion (RRF).
Extends IRetrievalEngine for direct drop-in compatibility across the RAG pipeline.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Optional

from adaptive_framework.rag.interfaces.i_retrieval_engine import IRetrievalEngine

if TYPE_CHECKING:
    from adaptive_framework.rag.retrieval.hybrid_result import HybridRetrievalResult


class IHybridRetriever(IRetrievalEngine):
    """Abstract interface for hybrid retrievers.

    Extends IRetrievalEngine with hybrid-specific retrieval capabilities
    and detailed provenance tracking across dense and sparse stages.
    """

    @abstractmethod
    def retrieve_hybrid(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: float = 0.0,
        metadata_filters: Optional[dict[str, Any]] = None,
    ) -> list[HybridRetrievalResult]:
        """Retrieve top-K chunks with explicit hybrid provenance metrics.

        Args:
            query: Natural-language query string.
            top_k: Number of fused results to return.
            min_score: Minimum fusion score threshold.
            metadata_filters: Optional metadata filters.

        Returns:
            List of HybridRetrievalResult objects ordered by descending RRF score.
        """
