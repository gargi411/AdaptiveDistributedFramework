"""IVectorStore — Abstract vector store interface (RAG pipeline)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from adaptive_framework.interfaces.i_chunker import TextChunk


@dataclass
class SearchResult:
    """A single result returned by a vector store similarity search.

    Attributes:
        chunk: The matched TextChunk.
        score: Similarity score (higher = more similar, in [0.0, 1.0]).
        document_id: Parent document identifier.
    """

    chunk: TextChunk
    score: float
    document_id: str


class IVectorStore(ABC):
    """Abstract interface for vector stores in the RAG pipeline.

    Stores (chunk, embedding) pairs and supports similarity search.
    Implementations may use ChromaDB, FAISS, or Qdrant without
    any change to the RAG pipeline calling code.

    Example:
        >>> store: IVectorStore = ChromaDBVectorStore(cfg.rag.vector_store, logger)
        >>> store.initialize()
        >>> store.upsert(chunks=[chunk1], embeddings=[[0.1, 0.2, ...]])
        >>> results = store.search(query_embedding=[0.15, 0.22, ...], top_k=5)
    """

    @abstractmethod
    def initialize(self) -> None:
        """Connect to or create the vector store collection.

        Raises:
            PipelineError: If initialization fails.
        """

    @abstractmethod
    def upsert(
        self,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Insert or update (chunk, embedding) pairs in the store.

        Args:
            chunks: List of TextChunk objects.
            embeddings: Corresponding embedding vectors. Must have the
                same length as chunks.

        Raises:
            PipelineError: If the upsert fails.
        """

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[SearchResult]:
        """Perform a similarity search against stored embeddings.

        Args:
            query_embedding: Query vector (must match store embedding_dim).
            top_k: Maximum number of results to return.

        Returns:
            List of SearchResult ordered by descending similarity score.

        Raises:
            PipelineError: If the search fails.
        """

    @abstractmethod
    def get_document_count(self) -> int:
        """Return the total number of chunks stored.

        Returns:
            Total chunk count in the collection.
        """

    @abstractmethod
    def clear(self) -> None:
        """Delete all entries from the collection.

        Raises:
            PipelineError: If clearing fails.
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Persist and close the vector store connection."""
