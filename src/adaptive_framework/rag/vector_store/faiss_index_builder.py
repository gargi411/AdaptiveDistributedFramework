"""FAISSIndexBuilder — factory for FAISS index types (Phase 4.3).

Supports three index types:

    flat   ->  faiss.IndexFlatIP        (exact inner-product search)
    ivf    ->  faiss.IndexIVFFlat       (approximate, partitioned)
    hnsw   ->  faiss.IndexHNSWFlat      (approximate, graph-based)

Default: IndexFlatIP (exact cosine search via L2-normalized vectors +
inner product).

Design principle: Keep it simple.  IndexFlatIP is the correct choice for
corpora up to a few hundred thousand chunks on CPU.  IndexIVFFlat and
IndexHNSWFlat are provided for when the corpus grows large enough to
benefit from approximate search.

FAISS index types:

IndexFlatIP
    Brute-force inner product.  Exact results.  O(N) per query.
    Fastest for small corpora (< 100K vectors).
    No training required.

IndexIVFFlat
    Inverted file index.  Partitions the space into n_clusters cells.
    Search visits nprobe cells.  Training is required on a representative
    sample of vectors.
    Good for 100K–10M vectors.

IndexHNSWFlat
    Hierarchical navigable small world graph.  No training.
    Very fast queries, high recall, moderate memory.
    Good for 10K–100M vectors.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# Known supported index type identifiers.
_VALID_INDEX_TYPES = {"flat", "ivf", "hnsw"}


class FAISSIndexBuilder:
    """Factory class for building FAISS indices.

    Usage:
        >>> builder = FAISSIndexBuilder(dim=1024, index_type="flat")
        >>> index = builder.build()
        >>> # For IVF, train before adding vectors:
        >>> builder2 = FAISSIndexBuilder(dim=1024, index_type="ivf", n_clusters=100)
        >>> index2 = builder2.build()

    Args:
        dim: Embedding dimension.  Must match the model output (1024 for BGE-large).
        index_type: One of 'flat', 'ivf', 'hnsw'.  Default: 'flat'.
        n_clusters: Number of Voronoi cells for IVF index.  Default: 100.
            Ignored for 'flat' and 'hnsw'.
        hnsw_m: Number of HNSW graph connections per element.  Default: 32.
            Ignored for 'flat' and 'ivf'.
    """

    def __init__(
        self,
        dim: int = 1024,
        index_type: str = "flat",
        n_clusters: int = 100,
        hnsw_m: int = 32,
    ) -> None:
        if index_type not in _VALID_INDEX_TYPES:
            raise ValueError(
                f"FAISSIndexBuilder: index_type must be one of {_VALID_INDEX_TYPES}, "
                f"got '{index_type}'."
            )
        if dim < 1:
            raise ValueError(f"FAISSIndexBuilder: dim must be >= 1, got {dim}.")
        self._dim = dim
        self._index_type = index_type
        self._n_clusters = n_clusters
        self._hnsw_m = hnsw_m

    def build(self) -> object:
        """Build and return a FAISS index of the configured type.

        For 'ivf', the returned index requires training before vectors
        can be added.  Call index.train(vectors) with a representative
        sample of at least n_clusters * 39 vectors.

        Returns:
            A FAISS index instance (faiss.Index subtype).

        Raises:
            RuntimeError: If faiss is not installed.
        """
        try:
            import faiss  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "faiss-cpu is not installed.  Install with: uv add faiss-cpu"
            ) from exc

        if self._index_type == "flat":
            index = faiss.IndexFlatIP(self._dim)
            log.debug("FAISSIndexBuilder: built IndexFlatIP(dim=%d).", self._dim)
        elif self._index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self._dim)
            index = faiss.IndexIVFFlat(quantizer, self._dim, self._n_clusters, faiss.METRIC_INNER_PRODUCT)
            log.debug(
                "FAISSIndexBuilder: built IndexIVFFlat(dim=%d, n_clusters=%d).",
                self._dim,
                self._n_clusters,
            )
        else:  # hnsw
            index = faiss.IndexHNSWFlat(self._dim, self._hnsw_m, faiss.METRIC_INNER_PRODUCT)
            log.debug(
                "FAISSIndexBuilder: built IndexHNSWFlat(dim=%d, M=%d).",
                self._dim,
                self._hnsw_m,
            )
        return index

    @property
    def index_type(self) -> str:
        """Return the configured index type string."""
        return self._index_type

    @property
    def dim(self) -> int:
        """Return the configured embedding dimension."""
        return self._dim
