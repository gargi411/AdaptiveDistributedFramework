"""IEmbeddingProvider — abstract embedding interface for Phase 4.3.

Extends the existing Phase 1 IEmbedder concept to operate on the richer
Chunk objects produced by Phase 4.2 SemanticChunker, and returns the
Embedding objects consumed by Phase 4.3 FAISSManager.

ASYMMETRIC ENCODING RULE (BGE-specific, enforced here as a contract):
    embed_query(query):  MUST prepend the instruction prefix.
    embed_chunks(chunks): MUST NOT prepend any prefix.

This asymmetry is a property of BAAI/bge-large-en-v1.5 and must be
preserved in every concrete implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from adaptive_framework.models.chunk import Chunk
from adaptive_framework.rag.models.embedding import Embedding


class IEmbeddingProvider(ABC):
    """Abstract interface for embedding models used in Phase 4.3.

    Concrete implementations (e.g. BGEEmbedder) must respect the
    asymmetric encoding rule documented in the module docstring.

    Example:
        >>> provider: IEmbeddingProvider = BGEEmbedder(model_name, device, batch_size)
        >>> provider.initialize()
        >>> embeddings = provider.embed_chunks([chunk1, chunk2])
        >>> q_vec = provider.embed_query("What is CRISPR?")
        >>> print(len(q_vec))
        1024
    """

    @abstractmethod
    def initialize(self) -> None:
        """Load the embedding model into memory.

        Raises:
            RuntimeError: If the model fails to load.
        """

    @abstractmethod
    def embed_chunks(self, chunks: list[Chunk]) -> list[Embedding]:
        """Compute embeddings for a list of Chunk objects.

        Chunks are encoded WITHOUT any instruction prefix.  This matches the
        BGE asymmetric encoding requirement for document-side passages.

        Args:
            chunks: Non-empty list of Chunk objects to embed.

        Returns:
            List of Embedding objects in the same order as the input chunks.
            Each vector has length == embedding_dim (1024 for BGE-large).

        Raises:
            RuntimeError: If embedding fails.
            ValueError: If chunks is empty.
        """

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Compute a query embedding with the BGE instruction prefix.

        The prefix is:
            "Represent this sentence for searching relevant passages: "

        This is the asymmetric query-side encoding required by
        BAAI/bge-large-en-v1.5 for retrieval tasks.

        Args:
            query: Raw natural language query string.

        Returns:
            Float list of length embedding_dim (1024 for BGE-large).

        Raises:
            RuntimeError: If embedding fails.
            ValueError: If query is empty or None.
        """

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the output embedding dimensionality.

        Returns:
            Integer dimension; 1024 for BAAI/bge-large-en-v1.5.
        """

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the embedding model identifier.

        Returns:
            Model name string, e.g. 'BAAI/bge-large-en-v1.5'.
        """

    @abstractmethod
    def get_metrics(self) -> dict[str, Any]:
        """Return a snapshot of embedding metrics.

        Returns:
            Dictionary with keys: chunks_embedded, batches_processed,
            cache_hits, cache_misses, total_embedding_time_s,
            avg_batch_time_s, throughput_chunks_per_s.
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Release model weights and free resources."""
