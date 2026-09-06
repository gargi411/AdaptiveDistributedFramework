"""Retrieval engine factory for strategy selection (Phase 4.8).

Instantiates the appropriate IRetrievalEngine implementation according to
the configured retrieval strategy ('dense' or 'hybrid').
"""

from __future__ import annotations

from typing import Optional

from adaptive_framework.rag.interfaces.i_retrieval_engine import IRetrievalEngine
from adaptive_framework.rag.interfaces.i_sparse_retriever import ISparseRetriever
from adaptive_framework.rag.reranking.cross_encoder_reranker import CrossEncoderReranker
from adaptive_framework.rag.reranking.i_reranker import IReranker
from adaptive_framework.rag.retrieval.hybrid_retriever import (
    DEFAULT_DENSE_TOP_K,
    DEFAULT_FINAL_TOP_K,
    DEFAULT_SPARSE_TOP_K,
    HybridRetriever,
)
from adaptive_framework.rag.retrieval.reranked_retriever import (
    DEFAULT_CANDIDATE_TOP_K,
    RerankedRetriever,
)
from adaptive_framework.rag.retrieval.rrf import DEFAULT_RRF_K


def create_retrieval_engine(
    strategy: str,
    dense_engine: IRetrievalEngine,
    sparse_retriever: Optional[ISparseRetriever] = None,
    reranker: Optional[IReranker] = None,
    dense_top_k: int = DEFAULT_DENSE_TOP_K,
    sparse_top_k: int = DEFAULT_SPARSE_TOP_K,
    candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
    final_top_k: int = DEFAULT_FINAL_TOP_K,
    rrf_k: int = DEFAULT_RRF_K,
) -> IRetrievalEngine:
    """Create an IRetrievalEngine according to the specified strategy.

    Args:
        strategy: 'dense', 'hybrid', or 'hybrid_reranked'.
        dense_engine: Configured dense retrieval engine.
        sparse_retriever: Optional sparse retriever (required if strategy is 'hybrid' or 'hybrid_reranked').
        reranker: Optional passage reranker (created if strategy is 'hybrid_reranked' and None).
        dense_top_k: Number of candidates fetched from dense retrieval in hybrid mode.
        sparse_top_k: Number of candidates fetched from sparse retrieval in hybrid mode.
        candidate_top_k: Number of candidate chunks passed to second-stage reranker.
        final_top_k: Number of fused/reranked results returned.
        rrf_k: RRF smoothing constant.

    Returns:
        An instance implementing IRetrievalEngine.

    Raises:
        ValueError: If strategy is invalid or if required dependencies are missing.
    """
    strat = strategy.lower().strip()
    if strat == "dense":
        return dense_engine
    elif strat == "hybrid":
        if sparse_retriever is None:
            raise ValueError(
                "sparse_retriever must be provided when retrieval strategy is 'hybrid'."
            )
        return HybridRetriever(
            dense_engine=dense_engine,
            sparse_retriever=sparse_retriever,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            final_top_k=final_top_k,
            rrf_k=rrf_k,
        )
    elif strat == "hybrid_reranked":
        if sparse_retriever is None:
            raise ValueError(
                "sparse_retriever must be provided when retrieval strategy is 'hybrid_reranked'."
            )
        active_reranker = reranker if reranker is not None else CrossEncoderReranker()
        hybrid_engine = HybridRetriever(
            dense_engine=dense_engine,
            sparse_retriever=sparse_retriever,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            final_top_k=candidate_top_k,
            rrf_k=rrf_k,
        )
        return RerankedRetriever(
            base_retriever=hybrid_engine,
            reranker=active_reranker,
            candidate_top_k=candidate_top_k,
            final_top_k=final_top_k,
        )
    else:
        raise ValueError(
            f"Unsupported retrieval strategy '{strategy}'. Supported strategies: 'dense', 'hybrid', 'hybrid_reranked'."
        )

