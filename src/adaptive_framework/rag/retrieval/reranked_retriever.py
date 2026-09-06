"""RerankedRetriever -- two-stage retrieval with cross-encoder reranking (Phase 4.9).

Composes a stage-1 retriever (e.g. HybridRetriever or RetrievalEngine) with an
IReranker (e.g. CrossEncoderReranker), implementing IRetrievalEngine for direct
drop-in compatibility across RAGService and RAGEvaluator.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from adaptive_framework.rag.interfaces.i_retrieval_engine import IRetrievalEngine
from adaptive_framework.rag.reranking.i_reranker import IReranker
from adaptive_framework.rag.reranking.rerank_result import RerankedRetrievalResult
from adaptive_framework.rag.retrieval.retrieval_result import RetrievalResult

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATE_TOP_K: int = 20
DEFAULT_FINAL_TOP_K: int = 5


class RerankedRetriever(IRetrievalEngine):
    """Two-stage retrieval engine: Candidate Retrieval -> Cross-Encoder Reranking."""

    def __init__(
        self,
        base_retriever: IRetrievalEngine,
        reranker: IReranker,
        candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
        final_top_k: int = DEFAULT_FINAL_TOP_K,
    ) -> None:
        """Initialize RerankedRetriever.

        Args:
            base_retriever: Stage-1 retrieval engine (e.g. HybridRetriever).
            reranker: Stage-2 passage reranker (e.g. CrossEncoderReranker).
            candidate_top_k: Number of candidate chunks to fetch from stage-1.
            final_top_k: Number of reranked chunks to return.
        """
        if base_retriever is None:
            raise TypeError("base_retriever must not be None")
        if reranker is None:
            raise TypeError("reranker must not be None")
        if candidate_top_k < 1:
            raise ValueError(f"candidate_top_k must be >= 1, got {candidate_top_k}")
        if final_top_k < 1:
            raise ValueError(f"final_top_k must be >= 1, got {final_top_k}")

        self.base_retriever = base_retriever
        self.reranker = reranker
        self.candidate_top_k = candidate_top_k
        self.final_top_k = final_top_k

        self._query_count: int = 0
        self._total_pipeline_time_ms: float = 0.0
        self._last_stage1_latency_ms: float = 0.0
        self._last_rerank_latency_ms: float = 0.0

    @property
    def last_stage1_latency_ms(self) -> float:
        """Latency of stage-1 base retrieval for the most recent query in ms."""
        return getattr(self, "_last_stage1_latency_ms", 0.0)

    @property
    def last_rerank_latency_ms(self) -> float:
        """Latency of stage-2 cross-encoder reranking for the most recent query in ms."""
        return getattr(self, "_last_rerank_latency_ms", 0.0)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: float = 0.0,
        metadata_filters: Optional[dict[str, Any]] = None,
        candidate_top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> list[RetrievalResult]:
        """Execute two-stage retrieval and return standard RetrievalResult list."""
        results = self.retrieve_reranked(
            query=query,
            top_k=top_k,
            min_score=min_score,
            metadata_filters=metadata_filters,
            candidate_top_k=candidate_top_k,
            batch_size=batch_size,
        )
        return list(results)

    def retrieve_reranked(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: float = 0.0,
        metadata_filters: Optional[dict[str, Any]] = None,
        candidate_top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> list[RerankedRetrievalResult]:
        """Execute two-stage retrieval returning detailed RerankedRetrievalResult objects."""
        if not query or not query.strip():
            raise ValueError("Query string must not be empty or whitespace-only.")

        effective_top_k = top_k if top_k is not None else self.final_top_k
        if effective_top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {effective_top_k}")

        effective_candidate_k = (
            candidate_top_k if candidate_top_k is not None else self.candidate_top_k
        )
        if effective_candidate_k < 1:
            raise ValueError(f"candidate_top_k must be >= 1, got {effective_candidate_k}")

        t0 = time.perf_counter()

        # 1. Retrieve candidates from stage-1 base retriever
        t_base_0 = time.perf_counter()
        try:
            candidates = self.base_retriever.retrieve(
                query=query,
                top_k=effective_candidate_k,
                min_score=0.0,
                metadata_filters=metadata_filters,
            )
        except Exception as exc:
            logger.warning("Stage-1 retrieval failed during reranked search: %s", exc)
            candidates = []

        stage1_elapsed_ms = (time.perf_counter() - t_base_0) * 1000.0
        self._last_stage1_latency_ms = stage1_elapsed_ms

        if not candidates:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._query_count += 1
            self._total_pipeline_time_ms += elapsed_ms
            self._last_rerank_latency_ms = 0.0
            return []

        # 2. Rerank candidates with cross-encoder
        t_rerank_0 = time.perf_counter()
        try:
            rerank_kwargs: dict[str, Any] = {"top_k": effective_top_k}
            if batch_size is not None:
                rerank_kwargs["batch_size"] = batch_size
            reranked = self.reranker.rerank(
                query=query,
                candidates=candidates,
                **rerank_kwargs,
            )
            rerank_elapsed_ms = (time.perf_counter() - t_rerank_0) * 1000.0
            self._last_rerank_latency_ms = rerank_elapsed_ms
        except Exception as exc:
            rerank_elapsed_ms = (time.perf_counter() - t_rerank_0) * 1000.0
            self._last_rerank_latency_ms = rerank_elapsed_ms
            logger.error("Reranking failed: %s. Falling back to base candidate order.", exc)
            # Graceful fallback: convert candidates to reranked results preserving order
            reranked = []
            for rank, c in enumerate(candidates[:effective_top_k], start=1):
                reranked.append(
                    RerankedRetrievalResult(
                        chunk_id=c.chunk_id,
                        document_id=c.document_id,
                        source_file=c.source_file,
                        page_numbers=c.page_numbers,
                        text=c.text,
                        section_heading=c.section_heading,
                        document_type=c.document_type,
                        chunk_index=c.chunk_index,
                        score=c.score,
                        rank=rank,
                        dense_rank=getattr(c, "dense_rank", c.rank),
                        sparse_rank=getattr(c, "sparse_rank", None),
                        dense_score=getattr(c, "dense_score", c.score),
                        sparse_score=getattr(c, "sparse_score", None),
                        rrf_score=getattr(c, "rrf_score", None),
                        reranker_score=c.score,
                        reranker_rank=rank,
                    )
                )

        if min_score > 0.0:
            reranked = [r for r in reranked if r.score >= min_score]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._query_count += 1
        self._total_pipeline_time_ms += elapsed_ms

        return reranked

    def get_metrics(self) -> dict[str, Any]:
        """Return operational metrics across all retrieval and reranking stages."""
        avg_ms = (
            self._total_pipeline_time_ms / self._query_count
            if self._query_count > 0
            else 0.0
        )
        return {
            "total_queries": self._query_count,
            "total_pipeline_time_ms": round(self._total_pipeline_time_ms, 3),
            "avg_pipeline_time_ms": round(avg_ms, 3),
            "last_stage1_latency_ms": round(self.last_stage1_latency_ms, 3),
            "last_rerank_latency_ms": round(self.last_rerank_latency_ms, 3),
            "candidate_top_k": self.candidate_top_k,
            "final_top_k": self.final_top_k,
            "base_metrics": self.base_retriever.get_metrics(),
            "reranker_metrics": self.reranker.get_metrics(),
        }
