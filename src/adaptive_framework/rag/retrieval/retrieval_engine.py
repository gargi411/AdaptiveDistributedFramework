"""RetrievalEngine — semantic retrieval over FAISS with BGE embeddings (Phase 4.4).

This module implements the first retrieval layer of the Phase 4 Knowledge Layer:

    natural-language query
        -> BGEEmbedder.embed_query()          (asymmetric prefix applied inside)
        -> 1024-dim L2-normalized query vector
        -> FAISSManager.search()              (exact cosine via IndexFlatIP)
        -> list[SearchResult]                 (raw, FAISS-internal result objects)
        -> list[RetrievalResult]              (flat, public-facing result objects)

The RetrievalEngine does NOT:
    - Load embedding models (callers must call provider.initialize() beforehand).
    - Build or persist FAISS indices (callers supply a populated FAISSManager).
    - Perform BM25, hybrid, or re-ranking.
    - Generate answers or call any LLM.
    - Use GPU acceleration.

Validation:
    - Empty or whitespace-only query -> ValueError.
    - top_k < 1 -> ValueError.
    - top_k > max_top_k -> ValueError (prevents unbounded result sets).
    - Embedding provider not initialized -> RuntimeError (propagated from provider).
    - Wrong embedding dimension -> ValueError (propagated from FAISSManager).
    - Empty FAISS index -> returns [] (not an error; callers should check).

Metadata filtering:
    The optional metadata_filters dict is passed through to
    FAISSManager.search(), which already implements document_id filtering
    as a Phase 4.4 hook.  Other filter keys are silently ignored by the
    current FAISSManager version.

Score semantics:
    Scores are cosine similarities in [-1, 1].  For L2-normalized vectors
    indexed with IndexFlatIP (the default), scores are in [0, 1].  A score
    of 1.0 means the query vector and chunk vector are identical.  Higher
    scores indicate greater semantic similarity.

    Scores are NOT re-scaled or clipped by RetrievalEngine.  They are
    returned exactly as reported by FAISSManager.search().

Ranking:
    Results are returned in descending score order, matching the order
    produced by FAISSManager.search().  The rank field of RetrievalResult
    is 1-indexed (rank=1 is the most similar result).

    RetrievalEngine preserves the FAISS ranking order and does not re-rank.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from adaptive_framework.rag.interfaces.i_embedding_provider import IEmbeddingProvider
from adaptive_framework.rag.retrieval.retrieval_metrics import RetrievalMetrics
from adaptive_framework.rag.retrieval.retrieval_result import RetrievalResult
from adaptive_framework.rag.vector_store.faiss_manager import FAISSManager

log = logging.getLogger(__name__)

# Default and ceiling values for top_k.
_DEFAULT_TOP_K: int = 5
_DEFAULT_MAX_TOP_K: int = 50


class RetrievalEngine:
    """Semantic retrieval engine: query -> BGE embedding -> FAISS -> results.

    Wraps an IEmbeddingProvider and a FAISSManager to provide a single
    retrieve() entry-point that returns RetrievalResult objects.

    Args:
        embedding_provider: An initialized IEmbeddingProvider instance.
            The caller is responsible for calling provider.initialize() before
            passing it to this constructor.
        faiss_manager: A populated FAISSManager instance.
            The manager must already contain indexed chunks; this class
            does not add or modify the FAISS index.
        default_top_k: Default number of results to return when top_k is not
            specified by the caller.  Must be >= 1.  Default: 5.
        max_top_k: Upper bound on top_k.  Any request for more results than
            this limit raises ValueError.  Default: 50.

    Raises:
        ValueError: If default_top_k < 1 or max_top_k < default_top_k.
        TypeError: If embedding_provider or faiss_manager are None.

    Example:
        >>> embedder = BGEEmbedder()
        >>> embedder.initialize()
        >>> manager = FAISSManager(dim=1024)
        >>> manager.add_embeddings_with_chunks(embeddings, chunks)
        >>> engine = RetrievalEngine(embedder, manager)
        >>> results = engine.retrieve("elevated troponin in cardiac patients", top_k=3)
        >>> results[0].rank
        1
        >>> results[0].score > 0.0
        True
    """

    def __init__(
        self,
        embedding_provider: IEmbeddingProvider,
        faiss_manager: FAISSManager,
        default_top_k: int = _DEFAULT_TOP_K,
        max_top_k: int = _DEFAULT_MAX_TOP_K,
    ) -> None:
        if embedding_provider is None:
            raise TypeError("RetrievalEngine: embedding_provider must not be None.")
        if faiss_manager is None:
            raise TypeError("RetrievalEngine: faiss_manager must not be None.")
        if default_top_k < 1:
            raise ValueError(
                f"RetrievalEngine: default_top_k must be >= 1, got {default_top_k}."
            )
        if max_top_k < default_top_k:
            raise ValueError(
                f"RetrievalEngine: max_top_k ({max_top_k}) must be >= "
                f"default_top_k ({default_top_k})."
            )

        self._provider = embedding_provider
        self._manager = faiss_manager
        self._default_top_k = default_top_k
        self._max_top_k = max_top_k
        self._metrics = RetrievalMetrics()

        log.info(
            "RetrievalEngine initialized: model='%s', default_top_k=%d, max_top_k=%d.",
            self._provider.get_model_name(),
            self._default_top_k,
            self._max_top_k,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: float = 0.0,
        metadata_filters: Optional[dict[str, Any]] = None,
    ) -> list[RetrievalResult]:
        """Retrieve the top-K most semantically similar chunks for a query.

        Pipeline:
            1. Validate query and top_k.
            2. Embed query with the BGE asymmetric prefix via embed_query().
            3. Search the FAISS index for the top-K nearest neighbours.
            4. Convert each SearchResult to a RetrievalResult.
            5. Apply min_score threshold (post-filter).
            6. Record metrics.
            7. Return results in descending score order.

        Args:
            query: Natural-language query string.  Must be non-empty and
                contain at least one non-whitespace character.
            top_k: Maximum number of results to return.  Must be >= 1 and
                <= max_top_k.  If None, uses default_top_k.
            min_score: Minimum cosine similarity score to include a result.
                Results with score < min_score are discarded after retrieval.
                Default: 0.0 (no filtering).
            metadata_filters: Optional dict passed through to
                FAISSManager.search() for document-level filtering.
                Supported key: 'document_id' (str or list[str]).

        Returns:
            List of RetrievalResult objects ordered by descending score.
            May be empty if the index is empty, all results are below
            min_score, or metadata filters exclude all candidates.

        Raises:
            ValueError: If query is empty or whitespace-only.
            ValueError: If top_k < 1 or top_k > max_top_k.
            RuntimeError: If the embedding provider has not been initialized.
            ValueError: If the query embedding has wrong dimension (propagated
                from FAISSManager.search()).
        """
        # ---- Step 1: Validate inputs -----------------------------------
        resolved_top_k = top_k if top_k is not None else self._default_top_k
        self._validate_query(query)
        self._validate_top_k(resolved_top_k)

        t0 = time.perf_counter()

        # ---- Step 2: Embed query ---------------------------------------
        # embed_query() applies the BGE instruction prefix internally.
        # It raises ValueError on empty/whitespace (already validated above).
        # It raises RuntimeError if not initialized.
        query_vector: list[float] = self._provider.embed_query(query)

        # ---- Step 3: FAISS search --------------------------------------
        # FAISSManager.search() normalizes the query vector internally,
        # clips k to index size, and returns [] for an empty index.
        raw_results = self._manager.search(
            query_vector=query_vector,
            top_k=resolved_top_k,
            metadata_filters=metadata_filters,
        )

        # ---- Step 4: Convert to RetrievalResult ------------------------
        results: list[RetrievalResult] = []
        for sr in raw_results:
            result = RetrievalResult(
                chunk_id=sr.chunk.chunk_id,
                document_id=sr.chunk.document_id,
                source_file=sr.chunk.source_file,
                page_numbers=sr.chunk.page_numbers,
                text=sr.chunk.text,
                section_heading=sr.chunk.section_heading,
                document_type=sr.chunk.document_type,
                chunk_index=sr.chunk.chunk_index,
                score=sr.score,
                rank=sr.faiss_rank,
            )
            results.append(result)

        # ---- Step 5: Apply min_score threshold -------------------------
        if min_score > 0.0:
            results = [r for r in results if r.score >= min_score]

        # ---- Step 6: Record metrics ------------------------------------
        elapsed = time.perf_counter() - t0
        self._metrics.record_query(n_results=len(results), elapsed_s=elapsed)

        log.debug(
            "RetrievalEngine.retrieve: query=%r, top_k=%d, results=%d, "
            "elapsed=%.3fs.",
            query[:80],
            resolved_top_k,
            len(results),
            elapsed,
        )

        # ---- Step 7: Return --------------------------------------------
        return results

    # ------------------------------------------------------------------
    # Information
    # ------------------------------------------------------------------

    @property
    def metrics(self) -> RetrievalMetrics:
        """Return the RetrievalMetrics instance for this engine."""
        return self._metrics

    @property
    def default_top_k(self) -> int:
        """Return the default top_k value."""
        return self._default_top_k

    @property
    def max_top_k(self) -> int:
        """Return the maximum allowed top_k value."""
        return self._max_top_k

    def get_index_stats(self) -> dict[str, Any]:
        """Return FAISS index statistics from the underlying FAISSManager.

        Returns:
            Dictionary with total_chunks, total_documents, embedding_dimension,
            index_type, model_name, and metrics sub-dictionary.
        """
        return self._manager.get_index_stats()

    def get_metrics(self) -> dict[str, Any]:
        """Return a snapshot of retrieval metrics as a plain dictionary.

        Returns:
            Dictionary with total_queries, total_results_returned,
            total_retrieval_time_s, empty_results_count, avg_retrieval_time_ms.
        """
        return self._metrics.to_dict()

    def __repr__(self) -> str:
        return (
            f"RetrievalEngine("
            f"model='{self._provider.get_model_name()}', "
            f"index_chunks={self._manager.total_chunks}, "
            f"default_top_k={self._default_top_k}, "
            f"max_top_k={self._max_top_k})"
        )

    # ------------------------------------------------------------------
    # Private validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_query(query: str) -> None:
        """Raise ValueError if query is empty, None, or whitespace-only.

        Args:
            query: The raw query string to validate.

        Raises:
            ValueError: If query fails validation.
        """
        if not query or not query.strip():
            raise ValueError(
                "RetrievalEngine.retrieve: query must be a non-empty, "
                "non-whitespace string."
            )

    def _validate_top_k(self, top_k: int) -> None:
        """Raise ValueError if top_k is outside the valid range.

        Args:
            top_k: The requested result count.

        Raises:
            ValueError: If top_k < 1 or top_k > self._max_top_k.
        """
        if top_k < 1:
            raise ValueError(
                f"RetrievalEngine.retrieve: top_k must be >= 1, got {top_k}."
            )
        if top_k > self._max_top_k:
            raise ValueError(
                f"RetrievalEngine.retrieve: top_k ({top_k}) exceeds "
                f"max_top_k ({self._max_top_k})."
            )
