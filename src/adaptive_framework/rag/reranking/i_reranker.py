"""IReranker -- abstract interface for candidate reranking (Phase 4.9).

Defines the contract for second-stage passage rerankers (such as cross-encoders)
operating on candidate lists retrieved by dense, sparse, or hybrid retrievers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Sequence

if TYPE_CHECKING:
    from adaptive_framework.rag.reranking.rerank_result import RerankedRetrievalResult
    from adaptive_framework.rag.retrieval.retrieval_result import RetrievalResult


class IReranker(ABC):
    """Abstract interface for passage rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> list[RerankedRetrievalResult]:
        """Rerank a sequence of candidate chunks for a query using cross-attention.

        Args:
            query: Non-empty query string.
            candidates: Sequence of RetrievalResult instances to rescore.
            top_k: Number of reranked results to return. If None, returns all candidates.
            batch_size: Optional batch size override for this rerank call.

        Returns:
            List of RerankedRetrievalResult instances sorted in descending score order.
        """

    @abstractmethod
    def get_metrics(self) -> dict[str, Any]:
        """Return operational metrics for the reranker."""
