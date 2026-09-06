"""EmbeddingProvider — abstract base implementing shared BGE infrastructure.

Provides the concrete scaffold that BGEEmbedder builds on:
    - Lazy model initialization guard.
    - Shared EmbeddingMetrics instance.
    - Common validation helpers.

Concrete subclasses only need to implement _load_model() and the
three abstract methods declared in IEmbeddingProvider.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any

from adaptive_framework.models.chunk import Chunk
from adaptive_framework.rag.embedding.embedding_metrics import EmbeddingMetrics
from adaptive_framework.rag.interfaces.i_embedding_provider import IEmbeddingProvider
from adaptive_framework.rag.models.embedding import Embedding

log = logging.getLogger(__name__)


class EmbeddingProvider(IEmbeddingProvider):
    """Concrete base class for embedding providers.

    Subclasses override _load_model() to load the specific model and
    implement embed_chunks / embed_query using that model.

    Args:
        model_name: HuggingFace or local model identifier.
        device: Compute device ('cpu' or 'cuda').
        batch_size: Number of chunks per forward pass.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 64,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._initialized = False
        self._metrics = EmbeddingMetrics()

    # ------------------------------------------------------------------
    # IEmbeddingProvider — lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Load the embedding model into memory.

        Safe to call multiple times; subsequent calls are no-ops.

        Raises:
            RuntimeError: If model loading fails.
        """
        if self._initialized:
            return
        log.info("Loading embedding model '%s' on device '%s'.", self._model_name, self._device)
        self._load_model()
        self._initialized = True
        log.info("Embedding model '%s' loaded (dim=%d).", self._model_name, self.get_embedding_dim())

    @abstractmethod
    def _load_model(self) -> None:
        """Load the underlying model.  Called exactly once by initialize()."""

    # ------------------------------------------------------------------
    # IEmbeddingProvider — information
    # ------------------------------------------------------------------

    def get_model_name(self) -> str:
        """Return the model identifier string."""
        return self._model_name

    def get_metrics(self) -> dict[str, Any]:
        """Return current embedding metrics snapshot."""
        return self._metrics.to_dict()

    def shutdown(self) -> None:
        """Release model weights and free resources."""
        self._initialized = False
        log.info("EmbeddingProvider '%s' shut down.", self._model_name)

    # ------------------------------------------------------------------
    # Abstract — must be implemented by concrete subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def embed_chunks(self, chunks: list[Chunk]) -> list[Embedding]:
        """Embed a list of Chunk objects without query prefix."""

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a query string with the asymmetric instruction prefix."""

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the output embedding dimension."""

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _require_initialized(self) -> None:
        """Raise RuntimeError if initialize() has not been called."""
        if not self._initialized:
            raise RuntimeError(
                f"EmbeddingProvider '{self._model_name}' is not initialized. "
                "Call initialize() before embed_chunks() or embed_query()."
            )
