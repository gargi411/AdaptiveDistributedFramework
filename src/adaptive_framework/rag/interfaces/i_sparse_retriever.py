"""ISparseRetriever -- abstract interface for sparse lexical retrieval (Phase 4.8).

Defines the contract for sparse retrieval algorithms (such as BM25Okapi).
Sparse retrievers index text chunks and compute lexical relevance scores
for natural-language queries.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Sequence

from adaptive_framework.models.chunk import Chunk

if TYPE_CHECKING:
    from adaptive_framework.rag.retrieval.retrieval_result import RetrievalResult


class ISparseRetriever(ABC):
    """Abstract interface for sparse lexical retrievers.

    Concrete implementations index Chunk objects and provide lexical
    search capability returning standard RetrievalResult instances.
    """

    @abstractmethod
    def index_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Index a sequence of chunks into the sparse inverted index.

        Args:
            chunks: Sequence of Chunk objects. Must preserve chunk_id,
                document_id, and text.
        """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[RetrievalResult]:
        """Retrieve top-K chunks matching query terms using lexical scoring.

        Args:
            query: Non-empty query string.
            top_k: Number of ranked results to return. Must be >= 1.
            min_score: Minimum lexical score threshold.

        Returns:
            List of RetrievalResult instances sorted in descending score order.
        """

    @abstractmethod
    def get_metrics(self) -> dict[str, Any]:
        """Return operational metrics for the sparse retriever."""
