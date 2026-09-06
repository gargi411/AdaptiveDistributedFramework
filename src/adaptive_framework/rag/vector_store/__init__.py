"""Vector store package — Phase 4.3.

Exports:
    FAISSManager:       Manages in-memory FAISS index with metadata maps.
    FAISSIndexBuilder:  Factory for flat/ivf/hnsw FAISS index types.
    FAISSPersistence:   Save and load FAISSManager state to disk.
    FAISSMetrics:       Indexing and search statistics.
    SearchResult:       Single search result (Chunk + score).
"""

from adaptive_framework.rag.vector_store.faiss_index_builder import FAISSIndexBuilder
from adaptive_framework.rag.vector_store.faiss_manager import FAISSManager, SearchResult
from adaptive_framework.rag.vector_store.faiss_metrics import FAISSMetrics
from adaptive_framework.rag.vector_store.faiss_persistence import FAISSPersistence

__all__ = [
    "FAISSManager",
    "FAISSIndexBuilder",
    "FAISSPersistence",
    "FAISSMetrics",
    "SearchResult",
]
