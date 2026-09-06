"""Reciprocal Rank Fusion (RRF) for Hybrid Retrieval (Phase 4.8).

Combines ranked lists from heterogeneous retrieval systems (e.g. dense vector search
and sparse lexical search) without requiring score calibration or normalization.

Standard RRF formula for an item d appearing at rank r_m in system m:
    RRF(d) = sum_{m} 1 / (k + r_m(d))
where k is a smoothing constant (typically 60) and r_m(d) is 1-based rank.
"""

from __future__ import annotations

from typing import Sequence

from adaptive_framework.rag.retrieval.hybrid_result import HybridRetrievalResult
from adaptive_framework.rag.retrieval.retrieval_result import RetrievalResult

DEFAULT_RRF_K: int = 60


def reciprocal_rank_fusion(
    dense_results: Sequence[RetrievalResult],
    sparse_results: Sequence[RetrievalResult],
    rrf_k: int = DEFAULT_RRF_K,
    top_k: int = 5,
) -> list[HybridRetrievalResult]:
    """Fuse dense and sparse rankings using Reciprocal Rank Fusion (RRF).

    Args:
        dense_results: Sequence of RetrievalResult instances from dense search.
        sparse_results: Sequence of RetrievalResult instances from sparse search.
        rrf_k: Smoothing constant to attenuate the influence of top ranks. Must be >= 1.
        top_k: Maximum number of fused results to return. Must be >= 1.

    Returns:
        List of HybridRetrievalResult instances sorted in descending RRF score order,
        with updated 1-based ranks and preserved dense/sparse provenance.

    Raises:
        ValueError: If rrf_k < 1 or top_k < 1.
    """
    if rrf_k < 1:
        raise ValueError(f"rrf_k must be >= 1, got {rrf_k}")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    # Track scores and metadata by chunk_id
    fused_scores: dict[str, float] = {}
    dense_ranks: dict[str, int] = {}
    sparse_ranks: dict[str, int] = {}
    dense_scores: dict[str, float] = {}
    sparse_scores: dict[str, float] = {}
    chunk_meta: dict[str, RetrievalResult] = {}

    # Process dense results
    for i, r in enumerate(dense_results):
        rank = r.rank if r.rank >= 1 else (i + 1)
        chunk_id = r.chunk_id
        contribution = 1.0 / (rrf_k + rank)

        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + contribution
        dense_ranks[chunk_id] = rank
        dense_scores[chunk_id] = r.score
        if chunk_id not in chunk_meta:
            chunk_meta[chunk_id] = r

    # Process sparse results
    for j, r in enumerate(sparse_results):
        rank = r.rank if r.rank >= 1 else (j + 1)
        chunk_id = r.chunk_id
        contribution = 1.0 / (rrf_k + rank)

        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + contribution
        sparse_ranks[chunk_id] = rank
        sparse_scores[chunk_id] = r.score
        if chunk_id not in chunk_meta:
            chunk_meta[chunk_id] = r

    # Sort descending by fused RRF score
    sorted_items = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    results: list[HybridRetrievalResult] = []
    for final_rank, (chunk_id, rrf_score) in enumerate(sorted_items[:top_k], start=1):
        meta = chunk_meta[chunk_id]
        results.append(
            HybridRetrievalResult(
                chunk_id=meta.chunk_id,
                document_id=meta.document_id,
                source_file=meta.source_file,
                page_numbers=meta.page_numbers,
                text=meta.text,
                section_heading=meta.section_heading,
                document_type=meta.document_type,
                chunk_index=meta.chunk_index,
                score=rrf_score,
                rank=final_rank,
                dense_rank=dense_ranks.get(chunk_id),
                sparse_rank=sparse_ranks.get(chunk_id),
                dense_score=dense_scores.get(chunk_id),
                sparse_score=sparse_scores.get(chunk_id),
                rrf_score=rrf_score,
            )
        )

    return results
