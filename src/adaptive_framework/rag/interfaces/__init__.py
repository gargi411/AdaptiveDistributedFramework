"""RAG interfaces package — Phase 4.4.

Exports abstract base classes for the embedding and retrieval layers.
"""

from adaptive_framework.rag.interfaces.i_embedding_provider import IEmbeddingProvider
from adaptive_framework.rag.interfaces.i_hybrid_retriever import IHybridRetriever
from adaptive_framework.rag.interfaces.i_llm_provider import ILLMProvider
from adaptive_framework.rag.interfaces.i_retrieval_engine import IRetrievalEngine
from adaptive_framework.rag.interfaces.i_sparse_retriever import ISparseRetriever

__all__ = [
    "IEmbeddingProvider",
    "IHybridRetriever",
    "ILLMProvider",
    "IRetrievalEngine",
    "ISparseRetriever",
]

