"""Embedding — immutable dense-vector representation of a Chunk (Phase 4.3).

Architecture position:

    list[Chunk]
        -> BGEEmbedder           (rag/embedding/bge_embedder.py)
        -> list[Embedding]       <- this module
        -> FAISSManager          (rag/vector_store/faiss_manager.py)
        -> FAISS index

Design rules:
    - Frozen dataclass, consistent with Chunk, Page, and UnifiedDocument.
    - vector is stored as a plain Python tuple[float, ...] so the object is
      picklable without NumPy.  The FAISSManager converts to np.ndarray on
      insertion.
    - embedding_time_seconds measures only the forward pass for this
      particular chunk (or its proportional share of a batch).
    - created_at is an ISO 8601 UTC timestamp set at embedding time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Embedding:
    """Dense vector representation of a single Chunk.

    Produced by BGEEmbedder.  Stored in FAISSManager alongside the
    originating Chunk.

    Attributes:
        chunk_id: Matches Chunk.chunk_id exactly (16-char hex digest).
        document_id: Matches Chunk.document_id.
        vector: 1024-dimensional float tuple (BAAI/bge-large-en-v1.5).
            Stored as a tuple to guarantee immutability and picklability.
        model_name: Embedding model identifier, e.g. 'BAAI/bge-large-en-v1.5'.
        embedding_time_seconds: Wall-clock seconds for this chunk's embedding.
        created_at: ISO 8601 UTC timestamp of creation.

    Example:
        >>> import numpy as np
        >>> vec = tuple(float(x) for x in np.random.randn(1024))
        >>> emb = Embedding(
        ...     chunk_id="a1b2c3d4e5f60718",
        ...     document_id="doc-001",
        ...     vector=vec,
        ...     model_name="BAAI/bge-large-en-v1.5",
        ...     embedding_time_seconds=0.012,
        ...     created_at="2026-08-31T00:00:00+00:00",
        ... )
        >>> len(emb.vector)
        1024
    """

    chunk_id: str
    document_id: str
    vector: tuple[float, ...]
    model_name: str
    embedding_time_seconds: float
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary.

        Note: the vector is converted to a list for JSON compatibility.

        Returns:
            Dictionary containing all embedding fields.
        """
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "vector": list(self.vector),
            "model_name": self.model_name,
            "embedding_time_seconds": self.embedding_time_seconds,
            "created_at": self.created_at,
        }

    def __repr__(self) -> str:
        return (
            f"Embedding(chunk_id='{self.chunk_id}', "
            f"doc='{self.document_id}', "
            f"dim={len(self.vector)}, "
            f"model='{self.model_name}')"
        )
