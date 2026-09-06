"""FAISSMetrics — indexing and search statistics for Phase 4.3.

Tracks:
    - total_chunks_indexed   Total chunks in the FAISS index.
    - total_documents        Unique document_ids in the index.
    - index_build_time_s     Cumulative time adding vectors to FAISS.
    - total_searches         Number of search() calls.
    - total_search_time_s    Cumulative search time (seconds).
    - avg_search_time_ms     Mean per-query search latency (milliseconds).
    - deletions              Number of delete_document() calls.
    - rebuilds               Number of full index rebuilds after deletion.

No retrieval pipeline metrics (BM25, hybrid) are included here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FAISSMetrics:
    """Mutable counter/timer object for the FAISS vector store.

    One instance is owned by FAISSManager and updated during
    add_embeddings(), search(), and delete_document() calls.

    Example:
        >>> m = FAISSMetrics()
        >>> m.record_add(n_chunks=64, n_documents=3, elapsed_s=0.5)
        >>> m.record_search(elapsed_s=0.001)
        >>> print(m.avg_search_time_ms)
        1.0
    """

    total_chunks_indexed: int = 0
    total_documents: int = 0
    index_build_time_s: float = 0.0
    total_searches: int = 0
    total_search_time_s: float = 0.0
    deletions: int = 0
    rebuilds: int = 0

    def record_add(
        self,
        n_chunks: int,
        n_documents: int,
        elapsed_s: float,
    ) -> None:
        """Record metrics for a completed add_embeddings() call.

        Args:
            n_chunks: Number of new chunks added.
            n_documents: Number of unique documents now in the index.
            elapsed_s: Wall-clock seconds for the insertion.
        """
        self.total_chunks_indexed += n_chunks
        self.total_documents = n_documents
        self.index_build_time_s += elapsed_s

    def record_search(self, elapsed_s: float) -> None:
        """Record metrics for one search() call.

        Args:
            elapsed_s: Wall-clock seconds for this search.
        """
        self.total_searches += 1
        self.total_search_time_s += elapsed_s

    def record_deletion(self, n_chunks_removed: int, rebuilt: bool) -> None:
        """Record a delete_document() operation.

        Args:
            n_chunks_removed: Number of chunks removed from the index.
            rebuilt: Whether a full index rebuild was performed.
        """
        self.total_chunks_indexed -= n_chunks_removed
        self.deletions += 1
        if rebuilt:
            self.rebuilds += 1

    @property
    def avg_search_time_ms(self) -> float:
        """Mean search latency in milliseconds.

        Returns:
            Mean latency, or 0.0 if no searches have been performed.
        """
        if self.total_searches == 0:
            return 0.0
        return (self.total_search_time_s / self.total_searches) * 1000.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise current metrics to a plain dictionary.

        Returns:
            Dictionary with all metric fields as primitive types.
        """
        return {
            "total_chunks_indexed": self.total_chunks_indexed,
            "total_documents": self.total_documents,
            "index_build_time_s": round(self.index_build_time_s, 4),
            "total_searches": self.total_searches,
            "total_search_time_s": round(self.total_search_time_s, 4),
            "avg_search_time_ms": round(self.avg_search_time_ms, 3),
            "deletions": self.deletions,
            "rebuilds": self.rebuilds,
        }

    def __repr__(self) -> str:
        return (
            f"FAISSMetrics("
            f"chunks={self.total_chunks_indexed}, "
            f"docs={self.total_documents}, "
            f"searches={self.total_searches}, "
            f"avg_search={self.avg_search_time_ms:.2f}ms)"
        )
