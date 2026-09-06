"""CrossEncoderReranker -- sentence-transformers Cross-Encoder reranking (Phase 4.9).

Implements second-stage cross-attention reranking using ms-marco-MiniLM-L-6-v2,
scoring (query, passage) pairs independently in batched passes.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional, Sequence

import numpy as np

from adaptive_framework.rag.reranking.i_reranker import IReranker
from adaptive_framework.rag.reranking.rerank_result import RerankedRetrievalResult
from adaptive_framework.rag.retrieval.hybrid_result import HybridRetrievalResult
from adaptive_framework.rag.retrieval.retrieval_result import RetrievalResult

logger = logging.getLogger(__name__)

DEFAULT_CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_BATCH_SIZE: int = 16


def _resolve_device(device_pref: str) -> str:
    """Resolve compute device ('auto' -> 'cuda' if available else 'cpu')."""
    pref = device_pref.lower().strip()
    if pref == "auto":
        try:
            import torch  # type: ignore[import-untyped]
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    elif pref in ("cuda", "cpu"):
        return pref
    else:
        logger.warning("Unrecognized device '%s', defaulting to 'cpu'.", device_pref)
        return "cpu"


class CrossEncoderReranker(IReranker):
    """Passage reranker using SentenceTransformers CrossEncoder."""

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        device: str = "auto",
        batch_size: int = DEFAULT_BATCH_SIZE,
        model: Any = None,
    ) -> None:
        """Initialize CrossEncoderReranker.

        Args:
            model_name: HuggingFace model identifier.
            device: 'auto', 'cpu', or 'cuda'.
            batch_size: Batch size for cross-encoder inference.
            model: Optional pre-initialized CrossEncoder instance (useful for testing).
        """
        if not model_name or not model_name.strip():
            raise ValueError("model_name must not be empty.")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.model_name = model_name
        self.device = _resolve_device(device)
        self.batch_size = batch_size

        self._model = model
        self._is_initialized = model is not None

        self._query_count: int = 0
        self._total_pairs_scored: int = 0
        self._total_rerank_time_ms: float = 0.0
        self._last_rerank_time_ms: float = 0.0

    @property
    def last_rerank_time_ms(self) -> float:
        """Latency of the most recent rerank call in milliseconds."""
        return getattr(self, "_last_rerank_time_ms", 0.0)

    @property
    def is_initialized(self) -> bool:
        """Whether the model weights are loaded in memory."""
        return self._is_initialized and self._model is not None

    def initialize(self) -> None:
        """Load cross-encoder model weights into memory if not already initialized."""
        if self._is_initialized and self._model is not None:
            return

        t0 = time.perf_counter()
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]
            logger.info(
                "Loading cross-encoder model '%s' on device '%s'...",
                self.model_name,
                self.device,
            )
            self._model = CrossEncoder(self.model_name, device=self.device)
            self._is_initialized = True
            elapsed = (time.perf_counter() - t0) * 1000.0
            logger.info("Loaded cross-encoder model in %.2f ms.", elapsed)
        except Exception as exc:
            logger.error("Failed to load cross-encoder '%s': %s", self.model_name, exc)
            raise RuntimeError(
                f"Failed to initialize cross-encoder model '{self.model_name}': {exc}"
            ) from exc

    def warmup(self) -> float:
        """Explicitly load weights and run a dummy prediction pass to warm inference.

        Returns:
            Total cold-start loading time in milliseconds.
        """
        t0 = time.perf_counter()
        self.initialize()
        # Run one dummy prediction pair through the network
        if self._model is not None:
            self._model.predict([("warmup query", "warmup passage text")], batch_size=1)
        cold_start_ms = (time.perf_counter() - t0) * 1000.0
        logger.info("Cross-encoder model warmed up in %.2f ms.", cold_start_ms)
        return cold_start_ms

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> list[RerankedRetrievalResult]:
        """Rerank candidates using cross-encoder cross-attention.

        Args:
            query: Query string. Must not be empty.
            candidates: Sequence of RetrievalResult instances.
            top_k: Number of reranked results to return. If None, returns all candidates.
            batch_size: Optional batch size override for this call.

        Returns:
            List of RerankedRetrievalResult sorted descending by cross-encoder score.
        """
        if not query or not query.strip():
            raise ValueError("Query string must not be empty or whitespace-only.")
        if top_k is not None and top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        effective_batch_size = batch_size if batch_size is not None else self.batch_size
        if effective_batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {effective_batch_size}")

        if not candidates:
            return []

        # Ensure model is initialized
        if not self._is_initialized:
            self.initialize()

        t0 = time.perf_counter()
        # Form independent (query, chunk_text) pairs without cross-candidate concatenation
        pairs = [(query, c.text) for c in candidates]

        try:
            scores = self._model.predict(pairs, batch_size=effective_batch_size)
            if isinstance(scores, np.ndarray):
                raw_scores = scores.tolist()
            else:
                raw_scores = list(scores)
        except Exception as exc:
            logger.error("Cross-encoder inference failed: %s", exc)
            raise RuntimeError(f"Cross-encoder scoring failed: {exc}") from exc

        # Build candidate tuples with stage-1 metadata
        rescored: list[tuple[float, RetrievalResult]] = []
        for score, candidate in zip(raw_scores, candidates):
            rescored.append((float(score), candidate))

        # Sort descending by cross-encoder score
        rescored.sort(key=lambda item: item[0], reverse=True)

        results: list[RerankedRetrievalResult] = []
        limit = top_k if top_k is not None else len(rescored)

        for rank, (score, candidate) in enumerate(rescored[:limit], start=1):
            dense_rank = getattr(candidate, "dense_rank", None)
            sparse_rank = getattr(candidate, "sparse_rank", None)
            dense_score = getattr(candidate, "dense_score", None)
            sparse_score = getattr(candidate, "sparse_score", None)
            rrf_score = getattr(candidate, "rrf_score", None)

            # If candidate was standard dense result, preserve its dense attributes
            if dense_score is None and not isinstance(candidate, HybridRetrievalResult):
                dense_score = candidate.score
                dense_rank = candidate.rank

            results.append(
                RerankedRetrievalResult(
                    chunk_id=candidate.chunk_id,
                    document_id=candidate.document_id,
                    source_file=candidate.source_file,
                    page_numbers=candidate.page_numbers,
                    text=candidate.text,
                    section_heading=candidate.section_heading,
                    document_type=candidate.document_type,
                    chunk_index=candidate.chunk_index,
                    score=score,
                    rank=rank,
                    dense_rank=dense_rank,
                    sparse_rank=sparse_rank,
                    dense_score=dense_score,
                    sparse_score=sparse_score,
                    rrf_score=rrf_score,
                    reranker_score=score,
                    reranker_rank=rank,
                )
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._query_count += 1
        self._total_pairs_scored += len(pairs)
        self._total_rerank_time_ms += elapsed_ms
        self._last_rerank_time_ms = elapsed_ms

        return results

    def get_metrics(self) -> dict[str, Any]:
        """Return operational metrics."""
        avg_ms = (
            self._total_rerank_time_ms / self._query_count
            if self._query_count > 0
            else 0.0
        )
        return {
            "total_queries": self._query_count,
            "total_pairs_scored": self._total_pairs_scored,
            "total_rerank_time_ms": round(self._total_rerank_time_ms, 3),
            "avg_rerank_time_ms": round(avg_ms, 3),
            "last_rerank_time_ms": round(self.last_rerank_time_ms, 3),
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
        }
