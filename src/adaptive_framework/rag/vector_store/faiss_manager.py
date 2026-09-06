"""FAISSManager — in-memory FAISS index with full metadata mapping (Phase 4.3).

Responsibilities:
    - Initialize an empty FAISS index via FAISSIndexBuilder.
    - Add Embedding objects:
        * L2-normalize vectors before insertion (so that inner-product ==
          cosine similarity).
        * Maintain dict[int, str]: faiss_id -> chunk_id.
        * Maintain dict[str, Chunk]: chunk_id -> Chunk.
    - Search:
        * Normalize query vector.
        * Return (Chunk, score) pairs ordered by descending score.
        * Support metadata_filters dict for future Phase 4.4 filtering.
    - delete_document(document_id):
        * Remove all chunks belonging to a document.
        * Rebuild the FAISS index from remaining vectors.
    - Track statistics via FAISSMetrics.

This module does NOT implement:
    - BM25 retrieval
    - Hybrid search
    - Query planning
    - Context building
    - LLM integration

Vector normalization:
    IndexFlatIP computes the dot product of two vectors.  When both the
    stored vectors and the query vector are L2-normalized (unit length),
    the inner product equals the cosine similarity.  Normalization is
    applied inside this manager so callers do not need to worry about it.

Metadata mapping:
    Two complementary dictionaries are maintained:
        _faiss_id_to_chunk_id: maps FAISS integer row index -> chunk_id.
        _chunk_id_to_chunk:    maps chunk_id -> full Chunk object.
    When a chunk is deleted and the index is rebuilt, both dictionaries are
    rebuilt from scratch with consecutive FAISS IDs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from adaptive_framework.models.chunk import Chunk
from adaptive_framework.rag.models.embedding import Embedding
from adaptive_framework.rag.vector_store.faiss_index_builder import FAISSIndexBuilder
from adaptive_framework.rag.vector_store.faiss_metrics import FAISSMetrics

log = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single result from FAISSManager.search().

    Attributes:
        chunk: The matched Chunk object.
        score: Cosine similarity score in [0, 1] (higher = more similar).
        document_id: Parent document identifier.
        faiss_rank: 1-indexed rank within the FAISS result set.
    """

    chunk: Chunk
    score: float
    document_id: str
    faiss_rank: int


class FAISSManager:
    """Manages a FAISS index for dense vector search over Chunk objects.

    Args:
        dim: Embedding dimension.  Must be 1024 for BAAI/bge-large-en-v1.5.
        index_type: FAISS index type: 'flat', 'ivf', or 'hnsw'.
            Default: 'flat' (IndexFlatIP, exact cosine similarity).
        n_clusters: IVF cluster count.  Ignored for flat/hnsw.
        model_name: Embedding model identifier stored in index_stats.

    Example:
        >>> manager = FAISSManager(dim=1024)
        >>> manager.add_embeddings(embeddings)
        >>> results = manager.search(query_vec, top_k=5)
    """

    def __init__(
        self,
        dim: int = 1024,
        index_type: str = "flat",
        n_clusters: int = 100,
        model_name: str = "BAAI/bge-large-en-v1.5",
    ) -> None:
        self._dim = dim
        self._index_type = index_type
        self._n_clusters = n_clusters
        self._model_name = model_name

        # Core metadata maps.
        self._faiss_id_to_chunk_id: dict[int, str] = {}
        self._chunk_id_to_chunk: dict[str, Chunk] = {}

        # Stored vectors for rebuild (dict faiss_id -> np.ndarray row).
        # We keep these so we can rebuild the FAISS index after deletions.
        self._stored_vectors: dict[str, np.ndarray] = {}  # chunk_id -> vector (dim,)

        self._metrics = FAISSMetrics()

        builder = FAISSIndexBuilder(
            dim=dim,
            index_type=index_type,
            n_clusters=n_clusters,
        )
        self._index: object = builder.build()
        log.info(
            "FAISSManager initialized: dim=%d, index_type=%s.",
            dim,
            index_type,
        )

    # ------------------------------------------------------------------
    # Adding embeddings
    # ------------------------------------------------------------------

    def add_embeddings(self, embeddings: list[Embedding]) -> None:
        """Add a list of Embedding objects to the FAISS index.

        Vectors are L2-normalized before insertion so that inner product
        equals cosine similarity.  Duplicate chunk_ids are silently skipped.

        Args:
            embeddings: Non-empty list of Embedding objects.

        Raises:
            ValueError: If any embedding has the wrong vector dimension.
            RuntimeError: If faiss is not installed.
        """
        if not embeddings:
            return

        import faiss  # type: ignore[import-untyped]

        t0 = time.perf_counter()
        unique_embeddings = [
            e for e in embeddings if e.chunk_id not in self._chunk_id_to_chunk
        ]
        if not unique_embeddings:
            log.debug("FAISSManager.add_embeddings: all %d chunks already indexed.", len(embeddings))
            return

        # Validate dimensions.
        for emb in unique_embeddings:
            if len(emb.vector) != self._dim:
                raise ValueError(
                    f"FAISSManager: embedding dim mismatch for chunk '{emb.chunk_id}': "
                    f"expected {self._dim}, got {len(emb.vector)}."
                )

        # Build matrix.
        matrix = np.array(
            [emb.vector for emb in unique_embeddings], dtype=np.float32
        )  # shape (N, dim)

        # L2-normalize.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        matrix /= norms

        # For IVF: train if not yet trained.
        if hasattr(self._index, "is_trained") and not self._index.is_trained:  # type: ignore[union-attr]
            if matrix.shape[0] >= self._n_clusters:
                log.info("FAISSManager: training IVF index on %d vectors.", matrix.shape[0])
                self._index.train(matrix)  # type: ignore[union-attr]
            else:
                log.warning(
                    "FAISSManager: IVF index requires at least %d vectors for training; "
                    "got %d.  Adding to flat fallback behavior.",
                    self._n_clusters,
                    matrix.shape[0],
                )

        # Current FAISS size = next faiss_id.
        start_id: int = self._index.ntotal  # type: ignore[union-attr]
        self._index.add(matrix)  # type: ignore[union-attr]

        # Update metadata maps.
        for i, emb in enumerate(unique_embeddings):
            faiss_id = start_id + i
            self._faiss_id_to_chunk_id[faiss_id] = emb.chunk_id
            self._stored_vectors[emb.chunk_id] = matrix[i]

        # We do NOT have the Chunk objects here — they are stored separately.
        # Callers must also call register_chunks() or ensure they added via
        # add_embeddings_with_chunks().

        elapsed = time.perf_counter() - t0
        unique_docs = len({e.document_id for e in unique_embeddings})
        self._metrics.record_add(
            n_chunks=len(unique_embeddings),
            n_documents=len(self._get_unique_document_ids()),
            elapsed_s=elapsed,
        )
        log.debug(
            "FAISSManager: added %d embeddings in %.3fs (total=%d).",
            len(unique_embeddings),
            elapsed,
            self._index.ntotal,  # type: ignore[union-attr]
        )

    def register_chunks(self, chunks: list[Chunk]) -> None:
        """Register Chunk objects so they can be returned by search().

        This method is called after add_embeddings() to populate the
        chunk_id -> Chunk lookup.  Only chunks not already registered
        are added.

        Args:
            chunks: List of Chunk objects to register.
        """
        for chunk in chunks:
            if chunk.chunk_id not in self._chunk_id_to_chunk:
                self._chunk_id_to_chunk[chunk.chunk_id] = chunk

    def add_embeddings_with_chunks(
        self,
        embeddings: list[Embedding],
        chunks: list[Chunk],
    ) -> None:
        """Convenience method: add embeddings and register chunks together.

        Args:
            embeddings: Embedding objects to add to the FAISS index.
            chunks: Corresponding Chunk objects to register.
                Must have the same length as embeddings.

        Raises:
            ValueError: If lengths do not match.
        """
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"FAISSManager.add_embeddings_with_chunks: "
                f"embeddings ({len(embeddings)}) and chunks ({len(chunks)}) must have equal length."
            )
        self.add_embeddings(embeddings)
        self.register_chunks(chunks)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        metadata_filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """Search the FAISS index for the most similar chunks.

        The query vector is L2-normalized before the search so that the
        returned scores are cosine similarities.

        Args:
            query_vector: Float list of length == dim.
            top_k: Maximum number of results to return.
            metadata_filters: Reserved for Phase 4.4 metadata filtering.
                Currently a no-op; future versions will filter results by
                document_id, section, page_number, etc.

        Returns:
            List of SearchResult ordered by descending score.
            May be shorter than top_k if the index has fewer entries.

        Raises:
            RuntimeError: If faiss is not installed.
            ValueError: If query_vector has wrong dimension.
        """
        if len(query_vector) != self._dim:
            raise ValueError(
                f"FAISSManager.search: query_vector dim mismatch: "
                f"expected {self._dim}, got {len(query_vector)}."
            )
        n_total: int = self._index.ntotal  # type: ignore[union-attr]
        if n_total == 0:
            return []

        t0 = time.perf_counter()
        k = min(top_k, n_total)

        # Normalize query vector.
        q = np.array([query_vector], dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q /= norm

        scores, faiss_ids = self._index.search(q, k)  # type: ignore[union-attr]
        elapsed = time.perf_counter() - t0
        self._metrics.record_search(elapsed_s=elapsed)

        results: list[SearchResult] = []
        rank = 1
        for score, fid in zip(scores[0], faiss_ids[0]):
            if fid < 0:
                # FAISS returns -1 for invalid results.
                continue
            chunk_id = self._faiss_id_to_chunk_id.get(int(fid))
            if chunk_id is None:
                continue
            chunk = self._chunk_id_to_chunk.get(chunk_id)
            if chunk is None:
                continue

            # Apply metadata filters (Phase 4.4 hook — currently pass-through).
            if metadata_filters and not self._apply_filters(chunk, metadata_filters):
                continue

            results.append(
                SearchResult(
                    chunk=chunk,
                    score=float(score),
                    document_id=chunk.document_id,
                    faiss_rank=rank,
                )
            )
            rank += 1

        return results

    # ------------------------------------------------------------------
    # Document deletion and rebuild
    # ------------------------------------------------------------------

    def delete_document(self, document_id: str) -> int:
        """Remove all chunks belonging to a document from the index.

        After removal, the FAISS index is rebuilt from the remaining vectors.
        Both metadata maps are updated accordingly.

        Args:
            document_id: The document_id to remove.

        Returns:
            Number of chunks removed.  0 if the document was not in the index.
        """
        chunks_to_remove = [
            chunk_id
            for chunk_id, chunk in self._chunk_id_to_chunk.items()
            if chunk.document_id == document_id
        ]
        if not chunks_to_remove:
            return 0

        n_removed = len(chunks_to_remove)
        for chunk_id in chunks_to_remove:
            del self._chunk_id_to_chunk[chunk_id]
            self._stored_vectors.pop(chunk_id, None)

        # Rebuild FAISS index and faiss_id_to_chunk_id from remaining vectors.
        self._rebuild_index()

        self._metrics.record_deletion(n_chunks_removed=n_removed, rebuilt=True)
        log.info(
            "FAISSManager.delete_document: removed %d chunks for doc '%s'; "
            "index rebuilt (total=%d).",
            n_removed,
            document_id,
            self._index.ntotal,  # type: ignore[union-attr]
        )
        return n_removed

    def _rebuild_index(self) -> None:
        """Rebuild the FAISS index from the currently stored vectors.

        Called after delete_document() to reconstruct a consistent index
        from the surviving chunk vectors.  Also regenerates faiss_id_to_chunk_id.
        """
        import faiss  # type: ignore[import-untyped]

        builder = FAISSIndexBuilder(
            dim=self._dim,
            index_type=self._index_type,
            n_clusters=self._n_clusters,
        )
        self._index = builder.build()
        self._faiss_id_to_chunk_id = {}
        self._metrics.rebuilds += 1

        remaining_items = list(self._stored_vectors.items())
        if not remaining_items:
            return

        chunk_ids = [item[0] for item in remaining_items]
        vectors = np.stack([item[1] for item in remaining_items], axis=0)

        if hasattr(self._index, "is_trained") and not self._index.is_trained:  # type: ignore[union-attr]
            if vectors.shape[0] >= self._n_clusters:
                self._index.train(vectors)  # type: ignore[union-attr]

        self._index.add(vectors)  # type: ignore[union-attr]
        for i, chunk_id in enumerate(chunk_ids):
            self._faiss_id_to_chunk_id[i] = chunk_id

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def _get_unique_document_ids(self) -> set[str]:
        """Return the set of unique document_ids currently in the index."""
        return {chunk.document_id for chunk in self._chunk_id_to_chunk.values()}

    @staticmethod
    def _apply_filters(chunk: Chunk, filters: dict[str, Any]) -> bool:
        """Check whether a Chunk passes the given metadata filters.

        This is a Phase 4.4 hook.  Currently supports:
            document_id: str or list[str]

        Args:
            chunk: Chunk to check.
            filters: Dict of filter criteria.

        Returns:
            True if the chunk passes all filters, False otherwise.
        """
        if "document_id" in filters:
            allowed = filters["document_id"]
            if isinstance(allowed, str):
                if chunk.document_id != allowed:
                    return False
            elif isinstance(allowed, list):
                if chunk.document_id not in allowed:
                    return False
        return True

    # ------------------------------------------------------------------
    # Information
    # ------------------------------------------------------------------

    def get_index_stats(self) -> dict[str, Any]:
        """Return human-readable index statistics.

        Returns:
            Dictionary with total_chunks, total_documents, embedding_dimension,
            index_type, model_name.
        """
        return {
            "total_chunks": int(self._index.ntotal),  # type: ignore[union-attr]
            "total_documents": len(self._get_unique_document_ids()),
            "embedding_dimension": self._dim,
            "index_type": self._index_type,
            "model_name": self._model_name,
            "metrics": self._metrics.to_dict(),
        }

    @property
    def metrics(self) -> FAISSMetrics:
        """Return the FAISSMetrics instance for this manager."""
        return self._metrics

    @property
    def total_chunks(self) -> int:
        """Number of vectors currently in the FAISS index."""
        return int(self._index.ntotal)  # type: ignore[union-attr]

    def __repr__(self) -> str:
        return (
            f"FAISSManager("
            f"index_type='{self._index_type}', "
            f"dim={self._dim}, "
            f"chunks={self.total_chunks}, "
            f"docs={len(self._get_unique_document_ids())})"
        )
