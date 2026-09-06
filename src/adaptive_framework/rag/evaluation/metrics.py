"""Mathematical metrics for evaluating RAG retrieval performance."""

from __future__ import annotations

import math
from typing import Sequence, Set


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str] | Sequence[str],
    k: int,
) -> float:
    """Calculate Precision@K.

    Definition:
        Precision@K = (number of relevant retrieved items in top K) / K

    Convention:
        If fewer than K results are retrieved, the divisor remains K
        in accordance with standard Information Retrieval definitions.

    Args:
        retrieved_ids: Ranked sequence of retrieved identifiers.
        relevant_ids: Set or sequence of ground-truth relevant identifiers.
        k: Cutoff rank (must be >= 1).

    Returns:
        Precision score in [0.0, 1.0].

    Raises:
        ValueError: If k < 1.
    """
    if k < 1:
        raise ValueError(f"k must be an integer >= 1, got {k}.")

    rel_set = set(relevant_ids)
    if not rel_set or not retrieved_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    hits = sum(1 for item in top_k if item in rel_set)
    return hits / float(k)


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str] | Sequence[str],
    k: int,
) -> float:
    """Calculate Recall@K.

    Definition:
        Recall@K = (number of relevant retrieved items in top K) / (total relevant items)

    Args:
        retrieved_ids: Ranked sequence of retrieved identifiers.
        relevant_ids: Set or sequence of ground-truth relevant identifiers.
        k: Cutoff rank (must be >= 1).

    Returns:
        Recall score in [0.0, 1.0], or 0.0 if relevant_ids is empty.

    Raises:
        ValueError: If k < 1.
    """
    if k < 1:
        raise ValueError(f"k must be an integer >= 1, got {k}.")

    rel_set = set(relevant_ids)
    if not rel_set:
        return 0.0

    top_k = retrieved_ids[:k]
    hits = sum(1 for item in top_k if item in rel_set)
    return hits / float(len(rel_set))


def reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str] | Sequence[str],
    k: int | None = None,
) -> float:
    """Calculate Reciprocal Rank (RR) for a single query.

    Definition:
        RR = 1 / rank_of_first_relevant_result
        RR = 0.0 if no relevant result is found within the evaluated ranking.

    Args:
        retrieved_ids: Ranked sequence of retrieved identifiers.
        relevant_ids: Set or sequence of ground-truth relevant identifiers.
        k: Optional cutoff rank. If specified, only the top K ranks are considered.

    Returns:
        Reciprocal rank score in [0.0, 1.0].

    Raises:
        ValueError: If k is not None and k < 1.
    """
    if k is not None and k < 1:
        raise ValueError(f"k must be an integer >= 1, got {k}.")

    rel_set = set(relevant_ids)
    if not rel_set or not retrieved_ids:
        return 0.0

    candidates = retrieved_ids[:k] if k is not None else retrieved_ids
    for rank, item in enumerate(candidates, start=1):
        if item in rel_set:
            return 1.0 / float(rank)

    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str] | Sequence[str],
    k: int,
) -> float:
    """Calculate normalized Discounted Cumulative Gain at rank K (nDCG@K).

    Uses standard binary relevance (rel in {0, 1}):
        DCG@K = sum_{i=1}^{min(len, K)} rel_i / log2(i + 1)
        IDCG@K = sum_{i=1}^{min(|relevant|, K)} 1.0 / log2(i + 1)
        nDCG@K = DCG@K / IDCG@K

    Args:
        retrieved_ids: Ranked sequence of retrieved identifiers.
        relevant_ids: Set or sequence of ground-truth relevant identifiers.
        k: Cutoff rank (must be >= 1).

    Returns:
        nDCG score in [0.0, 1.0], or 0.0 if no relevant items exist.

    Raises:
        ValueError: If k < 1.
    """
    if k < 1:
        raise ValueError(f"k must be an integer >= 1, got {k}.")

    rel_set = set(relevant_ids)
    if not rel_set or not retrieved_ids:
        return 0.0

    top_k = retrieved_ids[:k]

    # Calculate Discounted Cumulative Gain (DCG)
    dcg = 0.0
    for i, item in enumerate(top_k, start=1):
        if item in rel_set:
            dcg += 1.0 / math.log2(i + 1)

    # Calculate Ideal Discounted Cumulative Gain (IDCG)
    ideal_hits = min(len(rel_set), k)
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0.0 else 0.0
