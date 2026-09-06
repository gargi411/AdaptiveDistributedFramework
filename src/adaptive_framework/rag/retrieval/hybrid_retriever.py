"""HybridRetriever -- combined dense and sparse retrieval with RRF (Phase 4.8).

Coordinates dense vector search (via existing RetrievalEngine) and sparse lexical
search (via BM25Retriever) and merges the resulting rankings using Reciprocal Rank Fusion.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from adaptive_framework.rag.interfaces.i_hybrid_retriever import IHybridRetriever
from adaptive_framework.rag.interfaces.i_retrieval_engine import IRetrievalEngine
from adaptive_framework.rag.interfaces.i_sparse_retriever import ISparseRetriever
from adaptive_framework.rag.retrieval.hybrid_result import HybridRetrievalResult
from adaptive_framework.rag.retrieval.retrieval_result import RetrievalResult
from adaptive_framework.rag.retrieval.rrf import DEFAULT_RRF_K, reciprocal_rank_fusion

logger = logging.getLogger(__name__)

DEFAULT_DENSE_TOP_K: int = 20
DEFAULT_SPARSE_TOP_K: int = 20
DEFAULT_FINAL_TOP_K: int = 5


class HybridRetriever(IHybridRetriever):
    """Hybrid retriever combining dense FAISS and sparse BM25 via RRF.

    Implements IHybridRetriever (which extends IRetrievalEngine), allowing it to serve
    as a drop-in replacement across the RAG service and evaluation pipelines.
    """

    def __init__(
        self,
        dense_engine: IRetrievalEngine,
        sparse_retriever: ISparseRetriever,
        dense_top_k: int = DEFAULT_DENSE_TOP_K,
        sparse_top_k: int = DEFAULT_SPARSE_TOP_K,
        final_top_k: int = DEFAULT_FINAL_TOP_K,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        """Initialize HybridRetriever.

        Args:
            dense_engine: Dense retrieval engine (e.g. RetrievalEngine with FAISS).
            sparse_retriever: Sparse lexical retriever (e.g. BM25Retriever).
            dense_top_k: Number of candidate chunks to fetch from dense retrieval.
            sparse_top_k: Number of candidate chunks to fetch from sparse retrieval.
            final_top_k: Default top-K results to return after fusion.
            rrf_k: Smoothing constant for Reciprocal Rank Fusion.
        """
        if dense_engine is None:
            raise TypeError("dense_engine must not be None")
        if sparse_retriever is None:
            raise TypeError("sparse_retriever must not be None")
        if dense_top_k < 1:
            raise ValueError(f"dense_top_k must be >= 1, got {dense_top_k}")
        if sparse_top_k < 1:
            raise ValueError(f"sparse_top_k must be >= 1, got {sparse_top_k}")
        if final_top_k < 1:
            raise ValueError(f"final_top_k must be >= 1, got {final_top_k}")
        if rrf_k < 1:
            raise ValueError(f"rrf_k must be >= 1, got {rrf_k}")

        self.dense_engine = dense_engine
        self.sparse_retriever = sparse_retriever
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.final_top_k = final_top_k
        self.rrf_k = rrf_k

        self._query_count: int = 0
        self._total_retrieval_time_ms: float = 0.0

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: float = 0.0,
        metadata_filters: Optional[dict[str, Any]] = None,
    ) -> list[RetrievalResult]:
        """Standard IRetrievalEngine retrieve method returning RetrievalResult list."""
        hybrid_results = self.retrieve_hybrid(
            query=query,
            top_k=top_k,
            min_score=min_score,
            metadata_filters=metadata_filters,
        )
        # HybridRetrievalResult is an instance of RetrievalResult
        return list(hybrid_results)

    def retrieve_hybrid(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: float = 0.0,
        metadata_filters: Optional[dict[str, Any]] = None,
    ) -> list[HybridRetrievalResult]:
        """Retrieve top-K chunks with rich hybrid provenance metadata."""
        if not query or not query.strip():
            raise ValueError("Query string must not be empty or whitespace-only.")

        effective_top_k = top_k if top_k is not None else self.final_top_k
        if effective_top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {effective_top_k}")

        t0 = time.perf_counter()

        # 1. Fetch dense candidates
        try:
            dense_results = self.dense_engine.retrieve(
                query=query,
                top_k=self.dense_top_k,
                min_score=0.0,
                metadata_filters=metadata_filters,
            )
        except Exception as exc:
            logger.warning("Dense retrieval failed during hybrid search: %s", exc)
            dense_results = []

        # 2. Fetch sparse candidates
        try:
            sparse_results = self.sparse_retriever.retrieve(
                query=query,
                top_k=self.sparse_top_k,
                min_score=0.0,
            )
        except Exception as exc:
            logger.warning("Sparse retrieval failed during hybrid search: %s", exc)
            sparse_results = []

        # 3. Apply Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            rrf_k=self.rrf_k,
            top_k=effective_top_k,
        )

        # 4. Optional score filtering
        if min_score > 0.0:
            fused = [r for r in fused if r.score >= min_score]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._query_count += 1
        self._total_retrieval_time_ms += elapsed_ms

        return fused

    def get_metrics(self) -> dict[str, Any]:
        """Return operational metrics for the hybrid retriever."""
        avg_ms = (
            self._total_retrieval_time_ms / self._query_count
            if self._query_count > 0
            else 0.0
        )
        return {
            "total_queries": self._query_count,
            "total_retrieval_time_ms": round(self._total_retrieval_time_ms, 3),
            "avg_retrieval_time_ms": round(avg_ms, 3),
            "dense_top_k": self.dense_top_k,
            "sparse_top_k": self.sparse_top_k,
            "final_top_k": self.final_top_k,
            "rrf_k": self.rrf_k,
            "dense_metrics": self.dense_engine.get_metrics(),
            "sparse_metrics": self.sparse_retriever.get_metrics(),
        }
