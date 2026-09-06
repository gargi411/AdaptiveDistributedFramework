"""IRetrievalEngine — abstract retrieval interface for Phase 4.4.

Defines the contract for retrieval engines in the Phase 4 Knowledge Layer.
The concrete implementation is RetrievalEngine (Phase 4.4 Milestone 1).

Future implementations (Phase 4.4 Milestone 2+) may implement:
    BM25RetrievalEngine
    HybridRetrievalEngine (vector + BM25)

This interface does NOT cover:
    - Context building
    - Prompt building
    - LLM integration
    - Answer generation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from adaptive_framework.rag.retrieval.retrieval_result import RetrievalResult


class IRetrievalEngine(ABC):
    """Abstract interface for retrieval engines.

    Concrete implementations take a natural-language query and return
    a ranked list of RetrievalResult objects sourced from indexed documents.

    Example:
        >>> engine: IRetrievalEngine = RetrievalEngine(provider, manager)
        >>> results = engine.retrieve("troponin elevation in MI patients", top_k=5)
        >>> for r in results:
        ...     print(r.rank, r.score, r.text[:60])
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: float = 0.0,
        metadata_filters: Optional[dict[str, Any]] = None,
    ) -> list[RetrievalResult]:
        """Retrieve the top-K most relevant chunks for a natural-language query.

        Args:
            query: Natural-language query string.  Must be non-empty.
            top_k: Maximum number of results to return.  If None, uses the
                implementation's default.  Must be >= 1.
            min_score: Minimum similarity score threshold.  Results below
                this score are excluded.  Default: 0.0 (no filter).
            metadata_filters: Optional implementation-specific filters.

        Returns:
            List of RetrievalResult objects ordered by descending score.
            Empty list if no results match.

        Raises:
            ValueError: If query is empty or whitespace-only.
            ValueError: If top_k < 1 or exceeds the implementation maximum.
            RuntimeError: If the underlying embedding provider is not ready.
        """

    @abstractmethod
    def get_metrics(self) -> dict[str, Any]:
        """Return a snapshot of retrieval metrics.

        Returns:
            Dictionary with at least: total_queries, total_results_returned,
            avg_retrieval_time_ms.
        """
