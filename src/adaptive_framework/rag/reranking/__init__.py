"""Reranking package -- Phase 4.9.

Exports interfaces, data models, and implementations for second-stage passage reranking.
"""

from adaptive_framework.rag.reranking.adaptive_policy import AdaptiveCandidateDepthPolicy
from adaptive_framework.rag.reranking.cross_encoder_reranker import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CROSS_ENCODER_MODEL,
    CrossEncoderReranker,
)
from adaptive_framework.rag.reranking.i_reranker import IReranker
from adaptive_framework.rag.reranking.rerank_result import RerankedRetrievalResult

__all__ = [
    "AdaptiveCandidateDepthPolicy",
    "CrossEncoderReranker",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CROSS_ENCODER_MODEL",
    "IReranker",
    "RerankedRetrievalResult",
]
