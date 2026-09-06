"""BatchEmbeddingWorker — Ray remote function for parallel chunk embedding.

Uses the EXISTING Ray infrastructure from Phase 3.  Does NOT create new
actors, does NOT modify the scheduler, does NOT create a new distributed
runtime.

The worker is a Ray remote function (not an actor) that:
    1. Receives a batch of serialised Chunk objects.
    2. Instantiates a BGEEmbedder locally (one model per worker process).
    3. Embeds the batch and returns a list of Embedding objects.

The caller (typically the ingestion pipeline) schedules these remote
tasks using the same Ray cluster already started by Phase 3.

IMPORTANT:
    - The model is loaded fresh in each worker process.  On CPU, this
      is acceptable because sentence-transformers' load time is a one-time
      cost per process, not per batch.
    - On Windows development machines, Ray may not be available.  The
      module provides a plain Python fallback via embed_batch_local()
      that runs without Ray.
    - Do NOT import this module at the top level of the RAG package; it
      should be imported lazily so that Ray import errors do not break
      CPU-only operation.

Default batch size: 64 (configurable via the batch_size parameter).
"""

from __future__ import annotations

import logging
from typing import Optional

from adaptive_framework.models.chunk import Chunk
from adaptive_framework.rag.models.embedding import Embedding

log = logging.getLogger(__name__)

# Default batch size for embedding workers.
DEFAULT_BATCH_SIZE: int = 64


def embed_batch_local(
    chunks: list[Chunk],
    model_name: str = "BAAI/bge-large-en-v1.5",
    device: str = "cpu",
    batch_size: int = DEFAULT_BATCH_SIZE,
    cache_path: Optional[str] = None,
) -> list[Embedding]:
    """Embed a batch of Chunk objects without Ray (local, single-process).

    This function mirrors the Ray remote function API but runs synchronously
    in the calling process.  Use this when Ray is not available (e.g. on
    Windows dev machines) or for small batches in unit tests.

    Args:
        chunks: List of Chunk objects to embed.
        model_name: HuggingFace model identifier.
        device: Compute device ('cpu' or 'cuda').
        batch_size: Forward-pass sub-batch size.
        cache_path: Path to EmbeddingCache SQLite file, or None to skip.

    Returns:
        List of Embedding objects in the same order as the input chunks.
    """
    from adaptive_framework.rag.embedding.bge_embedder import BGEEmbedder
    from adaptive_framework.rag.embedding.embedding_cache import EmbeddingCache

    cache: Optional[EmbeddingCache] = None
    if cache_path is not None:
        cache = EmbeddingCache(cache_path=cache_path)

    embedder = BGEEmbedder(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        cache=cache,
    )
    embedder.initialize()
    embeddings = embedder.embed_chunks(chunks)
    embedder.shutdown()
    if cache is not None:
        cache.close()
    return embeddings


def _make_ray_remote_function() -> object:
    """Return a Ray remote version of embed_batch_local, or None if Ray unavailable."""
    try:
        import ray  # type: ignore[import-untyped]

        @ray.remote  # type: ignore[misc]
        def _remote_embed_batch(
            chunks: list[Chunk],
            model_name: str = "BAAI/bge-large-en-v1.5",
            device: str = "cpu",
            batch_size: int = DEFAULT_BATCH_SIZE,
            cache_path: Optional[str] = None,
        ) -> list[Embedding]:
            """Ray remote wrapper around embed_batch_local."""
            return embed_batch_local(
                chunks=chunks,
                model_name=model_name,
                device=device,
                batch_size=batch_size,
                cache_path=cache_path,
            )

        return _remote_embed_batch
    except ImportError:
        log.debug("Ray not available; embed_batch_ray will not be usable.")
        return None


# Lazy-initialised Ray remote function.
# None on Windows dev machines where Ray is not installed.
_ray_remote_fn: object = None


def get_ray_embed_fn() -> object:
    """Return the Ray remote embedding function, creating it on first call.

    Returns:
        Ray remote function, or None if Ray is not available.
    """
    global _ray_remote_fn
    if _ray_remote_fn is None:
        _ray_remote_fn = _make_ray_remote_function()
    return _ray_remote_fn


def embed_batch_ray(
    chunks: list[Chunk],
    model_name: str = "BAAI/bge-large-en-v1.5",
    device: str = "cpu",
    batch_size: int = DEFAULT_BATCH_SIZE,
    cache_path: Optional[str] = None,
) -> list[Embedding]:
    """Schedule embedding of a batch via Ray and return the results.

    Falls back to embed_batch_local() if Ray is not available.

    Args:
        chunks: Batch of Chunk objects to embed.
        model_name: HuggingFace model identifier.
        device: Compute device.
        batch_size: Forward-pass sub-batch size.
        cache_path: Path to EmbeddingCache SQLite file, or None.

    Returns:
        List of Embedding objects.
    """
    fn = get_ray_embed_fn()
    if fn is None:
        log.warning("Ray not available; falling back to local embedding.")
        return embed_batch_local(
            chunks=chunks,
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            cache_path=cache_path,
        )

    try:
        import ray  # type: ignore[import-untyped]

        future = fn.remote(  # type: ignore[union-attr]
            chunks=chunks,
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            cache_path=cache_path,
        )
        return ray.get(future)  # type: ignore[no-any-return]
    except Exception as exc:
        log.warning("Ray embed_batch failed (%s); falling back to local.", exc)
        return embed_batch_local(
            chunks=chunks,
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            cache_path=cache_path,
        )
