"""BGEEmbedder — BAAI/bge-large-en-v1.5 embedding implementation (Phase 4.3).

Uses sentence-transformers to embed Chunk objects.  Implements the
asymmetric encoding rule required by BGE retrieval models:

    embed_query(query):
        PREPENDS the instruction prefix:
            "Represent this sentence for searching relevant passages: "

    embed_chunks(chunks):
        NO prefix is added.  Passages are encoded as-is.

Preserving this asymmetry is mandatory.  Violating it degrades retrieval
quality because the BGE model was trained with this asymmetry.

Model details:
    Name:      BAAI/bge-large-en-v1.5
    Dimension: 1024
    Max seq:   512 tokens (handled automatically by sentence-transformers)
    License:   MIT

Batch encoding:
    Chunks are split into sub-batches of size self._batch_size (default 64)
    and encoded sequentially.  sentence-transformers handles tokenization
    and padding internally.

Cache integration:
    If an EmbeddingCache is supplied, results are looked up before inference
    and stored after inference.  The cache key is SHA-256(chunk.text).
    Pass force_reembed=True to embed_chunks() to bypass the cache.

Determinism:
    Given identical input text and identical model weights, the output
    vectors are identical across runs (PyTorch deterministic mode is
    not required; sentence-transformers evaluation mode is deterministic
    by default on CPU).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from adaptive_framework.models.chunk import Chunk
from adaptive_framework.rag.embedding.embedding_cache import EmbeddingCache
from adaptive_framework.rag.embedding.embedding_provider import EmbeddingProvider
from adaptive_framework.rag.models.embedding import Embedding

# Module-level import so it can be patched in unit tests (mock.patch targets
# this attribute, not the inner import inside functions).
try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment,misc]

log = logging.getLogger(__name__)

# BGE asymmetric query instruction prefix.
_QUERY_PREFIX: str = "Represent this sentence for searching relevant passages: "

# BAAI/bge-large-en-v1.5 embedding dimension.
_BGE_LARGE_DIM: int = 1024


class BGEEmbedder(EmbeddingProvider):
    """Concrete embedder using BAAI/bge-large-en-v1.5 via sentence-transformers.

    Args:
        model_name: HuggingFace model identifier.
            Default: 'BAAI/bge-large-en-v1.5'.
        device: Compute device.  'cpu' (default) or 'cuda'.
        batch_size: Number of texts per forward pass.  Default: 64.
        cache: Optional EmbeddingCache.  If provided, cache hits skip
            model inference.
        normalize_embeddings: If True (default), L2-normalize vectors
            before returning.  Required for cosine similarity via
            inner-product FAISS indices.

    Example:
        >>> embedder = BGEEmbedder()
        >>> embedder.initialize()
        >>> embeddings = embedder.embed_chunks([chunk1, chunk2])
        >>> q_vec = embedder.embed_query("What is CRISPR?")
        >>> print(len(q_vec))
        1024
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: str = "cpu",
        batch_size: int = 64,
        cache: Optional[EmbeddingCache] = None,
        normalize_embeddings: bool = True,
    ) -> None:
        super().__init__(model_name=model_name, device=device, batch_size=batch_size)
        self._cache = cache
        self._normalize = normalize_embeddings
        self._model: object = None  # sentence_transformers.SentenceTransformer

    # ------------------------------------------------------------------
    # EmbeddingProvider._load_model
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the sentence-transformers model.

        Raises:
            RuntimeError: If sentence-transformers is not installed or the
                model cannot be loaded.
        """
        if SentenceTransformer is None:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is not installed.  "
                "Install with: uv add sentence-transformers"
            )
        self._model = SentenceTransformer(self._model_name, device=self._device)

    # ------------------------------------------------------------------
    # IEmbeddingProvider — information
    # ------------------------------------------------------------------

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension (1024 for bge-large-en-v1.5).

        Returns:
            1024 (hard-coded for this model; matches the model output).
        """
        return _BGE_LARGE_DIM

    # ------------------------------------------------------------------
    # IEmbeddingProvider — embedding
    # ------------------------------------------------------------------

    def embed_chunks(
        self,
        chunks: list[Chunk],
        force_reembed: bool = False,
    ) -> list[Embedding]:
        """Embed a list of Chunk objects WITHOUT any instruction prefix.

        Chunks are the document-side passages.  BGE requires that they
        are encoded without a prefix.

        Args:
            chunks: List of Chunk objects to embed.  Must be non-empty.
            force_reembed: If True, bypass the cache for all chunks and
                recompute embeddings even if cached entries exist.

        Returns:
            List of Embedding objects in the same order as input chunks.

        Raises:
            RuntimeError: If initialize() has not been called.
            ValueError: If chunks is empty.
        """
        self._require_initialized()
        if not chunks:
            raise ValueError("embed_chunks() requires at least one Chunk.")

        batch_start = time.perf_counter()
        results: list[Optional[Embedding]] = [None] * len(chunks)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []
        cache_hit_count = 0

        # --- Cache lookup pass ---
        for i, chunk in enumerate(chunks):
            if self._cache is not None and not force_reembed:
                key = self._cache.text_to_key(chunk.text)
                cached_vec = self._cache.get(key)
                if cached_vec is not None:
                    created_at = datetime.now(timezone.utc).isoformat()
                    results[i] = Embedding(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        vector=tuple(cached_vec),
                        model_name=self._model_name,
                        embedding_time_seconds=0.0,
                        created_at=created_at,
                    )
                    cache_hit_count += 1
                    continue
            uncached_indices.append(i)
            uncached_texts.append(chunk.text)

        # --- Model inference pass (only for cache misses) ---
        if uncached_texts:
            vectors = self._encode_texts(uncached_texts, add_prefix=False)
            infer_time_per_chunk = (time.perf_counter() - batch_start) / len(uncached_texts)

            for j, global_idx in enumerate(uncached_indices):
                chunk = chunks[global_idx]
                vec = vectors[j].tolist()
                created_at = datetime.now(timezone.utc).isoformat()
                emb = Embedding(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    vector=tuple(vec),
                    model_name=self._model_name,
                    embedding_time_seconds=infer_time_per_chunk,
                    created_at=created_at,
                )
                results[global_idx] = emb

                if self._cache is not None:
                    key = self._cache.text_to_key(chunk.text)
                    self._cache.put(key, vec)

        elapsed = time.perf_counter() - batch_start
        self._metrics.record_batch(
            batch_size=len(chunks),
            elapsed_s=elapsed,
            cache_hits=cache_hit_count,
            cache_misses=len(uncached_texts),
        )

        # Guaranteed: all slots are filled.
        return [r for r in results if r is not None]

    def embed_query(self, query: str) -> list[float]:
        """Embed a query string with the BGE instruction prefix.

        The prefix "Represent this sentence for searching relevant passages: "
        is prepended to the query before encoding.  This is required for
        correct asymmetric retrieval with BAAI/bge-large-en-v1.5.

        Args:
            query: Raw user query string.  Must be non-empty.

        Returns:
            Float list of length 1024.

        Raises:
            RuntimeError: If initialize() has not been called.
            ValueError: If query is empty.
        """
        self._require_initialized()
        if not query or not query.strip():
            raise ValueError("embed_query() requires a non-empty query string.")

        prefixed = _QUERY_PREFIX + query
        vectors = self._encode_texts([prefixed], add_prefix=False)
        return vectors[0].tolist()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_texts(self, texts: list[str], add_prefix: bool = False) -> np.ndarray:  # type: ignore[type-arg]
        """Run sentence-transformers encoding with optional normalization.

        Args:
            texts: List of text strings to encode.
            add_prefix: Not used; prefix is applied externally.

        Returns:
            NumPy array of shape (len(texts), embedding_dim), float32.
        """
        vectors: np.ndarray = self._model.encode(  # type: ignore[union-attr]
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors  # shape: (N, dim)
