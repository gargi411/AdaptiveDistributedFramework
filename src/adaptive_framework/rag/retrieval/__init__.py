"""Retrieval package — Phase 4.4 Milestone 1.

Implements the top-K semantic retrieval layer on top of the Phase 4.3
BGE embedding engine and FAISS vector store.

Pipeline:
    natural-language query
        -> IEmbeddingProvider.embed_query()   (applies BGE asymmetric prefix)
        -> 1024-dim query vector
        -> FAISSManager.search()              (exact cosine via IndexFlatIP)
        -> list[SearchResult]                 (raw FAISS results)
        -> list[RetrievalResult]              (flat, user-facing result objects)

Exports:
    RetrievalEngine:   Top-K retrieval over FAISS with a BGE embedder.
    RetrievalResult:   Flat, immutable result object for one retrieved chunk.
    RetrievalMetrics:  Per-session counters and latency tracker.
    IRetrievalEngine:  Abstract retrieval interface.

Not implemented in Phase 4.4 Milestone 1:
    BM25 retrieval
    Hybrid search
    Query planner
    Context builder
    Prompt builder
    LLM integration
    GPU acceleration
"""

from adaptive_framework.rag.retrieval.bm25_retriever import (
    DEFAULT_B,
    DEFAULT_K1,
    BM25Retriever,
    tokenize_clinical_text,
)
from adaptive_framework.rag.retrieval.hybrid_result import HybridRetrievalResult
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
from adaptive_framework.rag.retrieval.retrieval_engine import RetrievalEngine
from adaptive_framework.rag.retrieval.retrieval_factory import create_retrieval_engine
from adaptive_framework.rag.retrieval.retrieval_metrics import RetrievalMetrics
from adaptive_framework.rag.retrieval.retrieval_result import RetrievalResult
from adaptive_framework.rag.retrieval.rrf import DEFAULT_RRF_K, reciprocal_rank_fusion

__all__ = [
    "BM25Retriever",
    "DEFAULT_B",
    "DEFAULT_CANDIDATE_TOP_K",
    "DEFAULT_DENSE_TOP_K",
    "DEFAULT_FINAL_TOP_K",
    "DEFAULT_K1",
    "DEFAULT_RRF_K",
    "DEFAULT_SPARSE_TOP_K",
    "HybridRetrievalResult",
    "HybridRetriever",
    "RerankedRetriever",
    "RetrievalEngine",
    "RetrievalMetrics",
    "RetrievalResult",
    "create_retrieval_engine",
    "reciprocal_rank_fusion",
    "tokenize_clinical_text",
]
