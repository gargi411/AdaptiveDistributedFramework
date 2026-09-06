"""Embedding metrics — counters and timers for Phase 4.3 embedding pipeline.

Tracks:
    - chunks_embedded        Total chunks embedded so far (including cache hits).
    - batches_processed      Number of model forward passes executed.
    - cache_hits             Number of chunks served from the embedding cache.
    - cache_misses           Number of chunks requiring model inference.
    - total_embedding_time_s Wall-clock seconds spent in model inference only.
    - avg_batch_time_s       Mean seconds per batch (total / batches).
    - throughput_chunks_per_s Chunks per second including cache hits.

All public fields are primitive types so metrics can be serialised to
JSON without any transformation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EmbeddingMetrics:
    """Mutable counter/timer object for the embedding layer.

    One instance is typically owned by a BGEEmbedder and updated
    during embed_chunks / embed_query calls.

    Attributes:
        chunks_embedded: Total chunks embedded (cache hits + misses).
        batches_processed: Number of model forward-pass batches executed.
        cache_hits: Chunks returned from disk cache (no model call).
        cache_misses: Chunks that required a model forward pass.
        total_embedding_time_s: Total seconds spent in model inference.
        _start_wall_time: Internal; set by the first batch to compute
            throughput relative to session start.

    Example:
        >>> m = EmbeddingMetrics()
        >>> m.record_batch(batch_size=64, elapsed_s=1.2, cache_hits=10, cache_misses=54)
        >>> print(m.throughput_chunks_per_s)
    """

    chunks_embedded: int = 0
    batches_processed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_embedding_time_s: float = 0.0
    _session_start_s: float = field(default=0.0, repr=False)
    _session_elapsed_s: float = field(default=0.0, repr=False)

    def record_batch(
        self,
        batch_size: int,
        elapsed_s: float,
        cache_hits: int,
        cache_misses: int,
    ) -> None:
        """Record metrics for one completed embedding batch.

        Args:
            batch_size: Total chunks in the batch (hits + misses).
            elapsed_s: Wall-clock seconds for this batch.
            cache_hits: Chunks returned from cache in this batch.
            cache_misses: Chunks computed by model in this batch.
        """
        self.chunks_embedded += batch_size
        self.batches_processed += 1
        self.cache_hits += cache_hits
        self.cache_misses += cache_misses
        self.total_embedding_time_s += elapsed_s
        self._session_elapsed_s += elapsed_s

    @property
    def avg_batch_time_s(self) -> float:
        """Mean seconds per processed batch.

        Returns:
            Average batch time, or 0.0 if no batches have been processed.
        """
        if self.batches_processed == 0:
            return 0.0
        return self.total_embedding_time_s / self.batches_processed

    @property
    def throughput_chunks_per_s(self) -> float:
        """Chunks per second (model inference time only, misses only).

        Returns:
            Throughput, or 0.0 if no time has elapsed.
        """
        if self.total_embedding_time_s <= 0.0:
            return 0.0
        return self.cache_misses / self.total_embedding_time_s

    def to_dict(self) -> dict[str, Any]:
        """Serialise current metrics to a plain dictionary.

        Returns:
            Dictionary with all metric fields as primitive types.
        """
        return {
            "chunks_embedded": self.chunks_embedded,
            "batches_processed": self.batches_processed,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_embedding_time_s": round(self.total_embedding_time_s, 4),
            "avg_batch_time_s": round(self.avg_batch_time_s, 4),
            "throughput_chunks_per_s": round(self.throughput_chunks_per_s, 2),
        }

    def __repr__(self) -> str:
        return (
            f"EmbeddingMetrics("
            f"chunks={self.chunks_embedded}, "
            f"cache_hits={self.cache_hits}, "
            f"cache_misses={self.cache_misses}, "
            f"throughput={self.throughput_chunks_per_s:.1f} ch/s)"
        )
