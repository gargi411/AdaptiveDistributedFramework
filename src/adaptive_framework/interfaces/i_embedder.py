"""IEmbedder — Abstract text embedding interface (RAG pipeline)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from adaptive_framework.interfaces.i_chunker import TextChunk


class IEmbedder(ABC):
    """Abstract interface for text embedding models.

    Converts TextChunk objects into fixed-dimensional float vectors
    for storage in the vector store.

    Example:
        >>> embedder: IEmbedder = SentenceTransformerEmbedder(cfg.rag.embedder, logger)
        >>> embedder.initialize()
        >>> vectors = embedder.embed([chunk1, chunk2])
        >>> print(len(vectors[0]))  # embedding dimension
        384
    """

    @abstractmethod
    def initialize(self) -> None:
        """Load the embedding model into memory.

        Raises:
            PipelineError: If the model fails to load.
        """

    @abstractmethod
    def embed(self, chunks: list[TextChunk]) -> list[list[float]]:
        """Compute embedding vectors for a batch of TextChunks.

        Args:
            chunks: List of TextChunk objects to embed.

        Returns:
            List of embedding vectors. Each vector has length == embedding_dim.
                Order matches the input chunks list.

        Raises:
            PipelineError: If embedding fails.
        """

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the dimensionality of embedding vectors.

        Returns:
            Integer embedding dimension (e.g., 384, 768, 1536).
        """

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the embedding model identifier string.

        Returns:
            Model name (e.g., 'sentence-transformers/all-MiniLM-L6-v2').
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Release model weights and free resources."""
