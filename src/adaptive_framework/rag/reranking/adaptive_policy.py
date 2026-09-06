"""Adaptive candidate-depth policy for second-stage reranking (Phase 4.10).

Dynamically determines the candidate depth (e.g. 5, 10, 15, or 20) for cross-encoder
reranking based solely on pre-reranking stage-1 retrieval signals (score distributions,
score margins, and dense-sparse rank agreement).

CRITICAL RULE:
This module has zero knowledge of ground truth, test case IDs, or relevant document labels.
All decisions are computed strictly from runtime RetrievalResult properties.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from adaptive_framework.rag.retrieval.hybrid_result import HybridRetrievalResult
from adaptive_framework.rag.retrieval.retrieval_result import RetrievalResult

logger = logging.getLogger(__name__)


class AdaptiveCandidateDepthPolicy:
    """Computes dynamic candidate depth from stage-1 retrieval confidence signals."""

    def __init__(
        self,
        min_depth: int = 5,
        default_depth: int = 10,
        max_depth: int = 20,
        high_confidence_gap_threshold: float = 0.005,
        dense_sparse_rank_threshold: int = 5,
    ) -> None:
        """Initialize the adaptive candidate depth policy.

        Args:
            min_depth: Minimum candidate depth for high-confidence queries (>= 1).
            default_depth: Nominal candidate depth for moderate-confidence queries.
            max_depth: Maximum candidate depth for ambiguous/low-confidence queries.
            high_confidence_gap_threshold: Minimum RRF score delta between candidate 1
                and candidate 5 to indicate strong stage-1 separation.
            dense_sparse_rank_threshold: Rank threshold within which both dense and
                sparse engines must agree to classify a result as consensual.
        """
        if min_depth < 1:
            raise ValueError(f"min_depth must be >= 1, got {min_depth}")
        if default_depth < min_depth:
            raise ValueError(f"default_depth ({default_depth}) must be >= min_depth ({min_depth})")
        if max_depth < default_depth:
            raise ValueError(f"max_depth ({max_depth}) must be >= default_depth ({default_depth})")

        self.min_depth = min_depth
        self.default_depth = default_depth
        self.max_depth = max_depth
        self.high_confidence_gap_threshold = high_confidence_gap_threshold
        self.dense_sparse_rank_threshold = dense_sparse_rank_threshold

    def compute_signals(self, candidates: Sequence[RetrievalResult]) -> dict[str, Any]:
        """Extract pre-reranking confidence signals from stage-1 candidates.

        Args:
            candidates: Sequence of retrieved candidates from stage-1 (e.g. Hybrid).

        Returns:
            Dictionary of computed signal values.
        """
        if not candidates:
            return {
                "candidate_count": 0,
                "score_gap_1_5": 0.0,
                "top_score": 0.0,
                "dense_sparse_agreement": False,
                "has_consensus_top1": False,
            }

        top_score = candidates[0].score
        score_gap_1_5 = 0.0
        if len(candidates) >= 5:
            score_gap_1_5 = candidates[0].score - candidates[4].score
        elif len(candidates) > 1:
            score_gap_1_5 = candidates[0].score - candidates[-1].score

        # Check if the leading candidate is confirmed by both dense and sparse retrieval
        top_cand = candidates[0]
        dense_rank = getattr(top_cand, "dense_rank", None)
        sparse_rank = getattr(top_cand, "sparse_rank", None)

        has_consensus_top1 = (
            dense_rank is not None
            and sparse_rank is not None
            and dense_rank <= self.dense_sparse_rank_threshold
            and sparse_rank <= self.dense_sparse_rank_threshold
        )

        # Count how many of top-3 candidates have dual-engine consensus
        consensus_count = 0
        for c in candidates[:3]:
            dr = getattr(c, "dense_rank", None)
            sr = getattr(c, "sparse_rank", None)
            if dr is not None and sr is not None:
                if dr <= self.dense_sparse_rank_threshold and sr <= self.dense_sparse_rank_threshold:
                    consensus_count += 1

        dense_sparse_agreement = consensus_count >= 1

        return {
            "candidate_count": len(candidates),
            "score_gap_1_5": round(score_gap_1_5, 6),
            "top_score": round(top_score, 6),
            "dense_sparse_agreement": dense_sparse_agreement,
            "has_consensus_top1": has_consensus_top1,
            "consensus_top3_count": consensus_count,
        }

    def determine_depth(self, candidates: Sequence[RetrievalResult]) -> int:
        """Decide the optimal candidate depth based on pre-reranking signals.

        Decision boundaries:
        1. If fewer candidates than max_depth are available, clamp to available count.
        2. High Confidence (both dense & sparse agree on top candidate AND score gap > threshold):
           -> Use min_depth (e.g. 5 or 10)
        3. Moderate Confidence (dense & sparse agree on at least one top-3 candidate):
           -> Use default_depth (e.g. 10 or 15)
        4. Low Confidence / Ambiguous (no dual-engine agreement or near-zero score gap):
           -> Use max_depth (e.g. 20)

        Args:
            candidates: Sequence of stage-1 candidates.

        Returns:
            Selected integer candidate depth.
        """
        if not candidates:
            return self.min_depth

        available = len(candidates)
        signals = self.compute_signals(candidates)

        if signals["has_consensus_top1"] and signals["score_gap_1_5"] >= self.high_confidence_gap_threshold:
            chosen = self.min_depth
        elif signals["dense_sparse_agreement"]:
            chosen = self.default_depth
        else:
            chosen = self.max_depth

        return min(chosen, available)
