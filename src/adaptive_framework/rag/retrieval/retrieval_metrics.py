"""RetrievalMetrics — per-session retrieval counters and latency (Phase 4.4).

Tracks statistics for RetrievalEngine.retrieve() calls:
    total_queries            Number of retrieve() invocations.
    total_results_returned   Cumulative count of RetrievalResult objects returned.
    total_retrieval_time_s   Cumulative wall-clock seconds across all queries.
    empty_results_count      Number of queries that returned zero results.
    avg_retrieval_time_ms    Mean latency per query in milliseconds (property).

One RetrievalMetrics instance is owned by a RetrievalEngine and updated
after every retrieve() call completes successfully.

Failed calls (raised exceptions) are NOT recorded so that the metrics
reflect only the retrieval path, not error handling overhead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievalMetrics:
    """Mutable counter and timer for RetrievalEngine.

    Example:
        >>> m = RetrievalMetrics()
        >>> m.record_query(n_results=3, elapsed_s=0.012)
        >>> m.record_query(n_results=0, elapsed_s=0.008)
        >>> m.total_queries
        2
        >>> m.empty_results_count
        1
        >>> round(m.avg_retrieval_time_ms, 1)
        10.0
    """

    total_queries: int = 0
    total_results_returned: int = 0
    total_retrieval_time_s: float = 0.0
    empty_results_count: int = 0

    def record_query(self, n_results: int, elapsed_s: float) -> None:
        """Record one completed retrieve() call.

        Args:
            n_results: Number of RetrievalResult objects returned.
            elapsed_s: Wall-clock seconds for the full retrieval call.
        """
        self.total_queries += 1
        self.total_results_returned += n_results
        self.total_retrieval_time_s += elapsed_s
        if n_results == 0:
            self.empty_results_count += 1

    @property
    def avg_retrieval_time_ms(self) -> float:
        """Mean retrieval latency in milliseconds.

        Returns:
            Mean latency, or 0.0 if no queries have been recorded.
        """
        if self.total_queries == 0:
            return 0.0
        return (self.total_retrieval_time_s / self.total_queries) * 1000.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise current metrics to a plain dictionary.

        Returns:
            Dictionary with all metric fields as primitive types.
        """
        return {
            "total_queries": self.total_queries,
            "total_results_returned": self.total_results_returned,
            "total_retrieval_time_s": round(self.total_retrieval_time_s, 4),
            "empty_results_count": self.empty_results_count,
            "avg_retrieval_time_ms": round(self.avg_retrieval_time_ms, 3),
        }

    def __repr__(self) -> str:
        return (
            f"RetrievalMetrics("
            f"queries={self.total_queries}, "
            f"results={self.total_results_returned}, "
            f"empty={self.empty_results_count}, "
            f"avg={self.avg_retrieval_time_ms:.2f}ms)"
        )
