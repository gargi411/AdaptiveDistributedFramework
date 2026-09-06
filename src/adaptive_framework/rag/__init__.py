"""RAG package — Phase 4.4: Retrieval Engine.

Architecture v2.0 section 2.5:
    Document ingestion -> Chunking -> Embedding -> Vector Store -> Query

Phase 4.2 implements the Chunking stage.
Phase 4.3 implements the Embedding and Vector Store stages.
Phase 4.4 Milestone 1 implements the Retrieval Engine.
Context building, prompt building, and LLM are Phase 4.4+ later milestones.

Public exports (Phase 4.2):
    SemanticChunker: Converts UnifiedDocument -> list[Chunk].
    Chunk: Immutable, provenance-aware chunk data model.

Public exports (Phase 4.3):
    Embedding: Immutable dense-vector representation of a Chunk.
    BGEEmbedder: BAAI/bge-large-en-v1.5 sentence-transformers embedder.
    EmbeddingCache: SHA-256-keyed SQLite persistent embedding cache.
    EmbeddingMetrics: Throughput and cache counters for the embedding layer.
    FAISSManager: In-memory FAISS index with metadata maps.
    FAISSIndexBuilder: Factory for flat/ivf/hnsw FAISS index types.
    FAISSPersistence: Save and load FAISSManager state to disk.
    FAISSMetrics: Indexing and search statistics.

Public exports (Phase 4.4 Milestone 1):
    RetrievalEngine: Semantic retrieval — query -> BGE embedding -> FAISS -> results.
    RetrievalResult: Flat, immutable result object for one retrieved chunk.
    RetrievalMetrics: Per-session retrieval counters and latency tracker.

Interfaces:
    IEmbeddingProvider: Abstract embedding contract (Phase 4.3).
    IRetrievalEngine: Abstract retrieval contract (Phase 4.4).
"""

from adaptive_framework.models.chunk import Chunk
from adaptive_framework.rag.chunker import SemanticChunker
from adaptive_framework.rag.models.embedding import Embedding
from adaptive_framework.rag.embedding.bge_embedder import BGEEmbedder
from adaptive_framework.rag.embedding.embedding_cache import EmbeddingCache
from adaptive_framework.rag.embedding.embedding_metrics import EmbeddingMetrics
from adaptive_framework.rag.vector_store.faiss_manager import FAISSManager, SearchResult
from adaptive_framework.rag.vector_store.faiss_index_builder import FAISSIndexBuilder
from adaptive_framework.rag.vector_store.faiss_persistence import FAISSPersistence
from adaptive_framework.rag.vector_store.faiss_metrics import FAISSMetrics
from adaptive_framework.rag.retrieval.retrieval_engine import RetrievalEngine
from adaptive_framework.rag.retrieval.retrieval_result import RetrievalResult
from adaptive_framework.rag.retrieval.retrieval_metrics import RetrievalMetrics
from adaptive_framework.rag.retrieval.bm25_retriever import BM25Retriever
from adaptive_framework.rag.retrieval.hybrid_result import HybridRetrievalResult
from adaptive_framework.rag.retrieval.hybrid_retriever import HybridRetriever
from adaptive_framework.rag.retrieval.rrf import reciprocal_rank_fusion
from adaptive_framework.rag.reranking.adaptive_policy import AdaptiveCandidateDepthPolicy
from adaptive_framework.rag.reranking.cross_encoder_reranker import CrossEncoderReranker
from adaptive_framework.rag.reranking.i_reranker import IReranker
from adaptive_framework.rag.reranking.rerank_result import RerankedRetrievalResult
from adaptive_framework.rag.retrieval.reranked_retriever import RerankedRetriever
from adaptive_framework.rag.retrieval.retrieval_factory import create_retrieval_engine
from adaptive_framework.rag.interfaces.i_retrieval_engine import IRetrievalEngine
from adaptive_framework.rag.interfaces.i_sparse_retriever import ISparseRetriever
from adaptive_framework.rag.interfaces.i_hybrid_retriever import IHybridRetriever
from adaptive_framework.rag.interfaces.i_llm_provider import ILLMProvider
from adaptive_framework.rag.generation.context_builder import (
    EvidenceBlock,
    BuiltContext,
    ContextBuilder,
)
from adaptive_framework.rag.generation.prompt_builder import (
    BuiltPrompt,
    PromptBuilder,
)
from adaptive_framework.rag.generation.llm_response import LLMResponse
from adaptive_framework.rag.generation.fake_llm_provider import FakeLLMProvider
from adaptive_framework.rag.generation.answer import (
    SourceReference,
    RAGAnswer,
)
from adaptive_framework.rag.generation.rag_service import RAGService
from adaptive_framework.rag.generation.clinical_rag_orchestrator import ClinicalRAGOrchestrator
from adaptive_framework.rag.models.clinical_models import (
    CLINICAL_DISCLAIMER,
    ClinicalDocumentUpload,
    ClinicalQueryRequest,
    DoctorClinicalResponse,
    DoctorEvidenceItem,
    DocumentIngestionResult,
)

__all__ = [
    # Phase 4.2
    "SemanticChunker",
    "Chunk",
    # Phase 4.3
    "BGEEmbedder",
    "Embedding",
    "EmbeddingCache",
    "EmbeddingMetrics",
    "FAISSManager",
    "SearchResult",
    "FAISSIndexBuilder",
    "FAISSPersistence",
    "FAISSMetrics",
    # Phase 4.4
    "RetrievalEngine",
    "RetrievalResult",
    "RetrievalMetrics",
    "IRetrievalEngine",
    # Phase 4.5
    "EvidenceBlock",
    "BuiltContext",
    "ContextBuilder",
    "BuiltPrompt",
    "PromptBuilder",
    "LLMResponse",
    "FakeLLMProvider",
    "SourceReference",
    "RAGAnswer",
    "RAGService",
    "ILLMProvider",
    # Phase 4.8
    "BM25Retriever",
    "HybridRetrievalResult",
    "HybridRetriever",
    "reciprocal_rank_fusion",
    "ISparseRetriever",
    "IHybridRetriever",
    "create_retrieval_engine",
    # Phase 4.9
    "IReranker",
    "RerankedRetrievalResult",
    "CrossEncoderReranker",
    "RerankedRetriever",
    # Phase 4.10 — adaptive reranking optimization
    "AdaptiveCandidateDepthPolicy",
    # Phase 4.11 — doctor-facing clinical integration
    "ClinicalRAGOrchestrator",
    "ClinicalQueryRequest",
    "ClinicalDocumentUpload",
    "DoctorClinicalResponse",
    "DoctorEvidenceItem",
    "DocumentIngestionResult",
    "CLINICAL_DISCLAIMER",
]