"""Embedding package — Phase 4.3.

Exports:
    BGEEmbedder:          BAAI/bge-large-en-v1.5 concrete embedder.
    EmbeddingCache:       SHA-256-keyed SQLite persistent cache.
    EmbeddingMetrics:     Counter/timer for embedding throughput tracking.
    EmbeddingProvider:    Abstract base class (for subclassing only).
    embed_batch_local:    Local (non-Ray) batch embedding function.
    embed_batch_ray:      Ray-scheduled batch embedding function.
"""

from adaptive_framework.rag.embedding.bge_embedder import BGEEmbedder
from adaptive_framework.rag.embedding.batch_embedding_worker import (
    embed_batch_local,
    embed_batch_ray,
)
from adaptive_framework.rag.embedding.embedding_cache import EmbeddingCache
from adaptive_framework.rag.embedding.embedding_metrics import EmbeddingMetrics
from adaptive_framework.rag.embedding.embedding_provider import EmbeddingProvider

__all__ = [
    "BGEEmbedder",
    "EmbeddingCache",
    "EmbeddingMetrics",
    "EmbeddingProvider",
    "embed_batch_local",
    "embed_batch_ray",
]
