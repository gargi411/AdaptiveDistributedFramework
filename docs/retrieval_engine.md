Phase 4.4 Milestone 1 - Retrieval Engine
Adaptive Distributed Framework v2.0
=========================================

This document describes the retrieval layer added in Phase 4.4 Milestone 1.
It is the third stage of the Phase 4 Knowledge Layer pipeline:

    UnifiedDocument
        -> SemanticChunker (Phase 4.2)
        -> list[Chunk]
        -> BGEEmbedder              (Phase 4.3)
        -> list[Embedding]
        -> FAISSManager             (Phase 4.3)
        -> FAISS index on disk
        -> RetrievalEngine          (Phase 4.4 Milestone 1)
        -> list[RetrievalResult]

No answer generation, LLM, or context building is included in this milestone.


1. Purpose
----------

The retrieval engine takes a natural-language query and returns the K most
semantically similar chunks from the FAISS index.  It does not generate
answers.  Its output (list[RetrievalResult]) is the input to the context
builder (Phase 4.4 Milestone 2, not yet implemented).


2. Retrieval Flow
-----------------

Step 1: Validate inputs.
    - query must be non-empty, non-whitespace.
    - top_k must be >= 1 and <= max_top_k (default: 50).

Step 2: Embed the query.
    Method: IEmbeddingProvider.embed_query(query)
    The BGEEmbedder implementation prepends:
        "Represent this sentence for searching relevant passages: "
    This asymmetric encoding is a hard requirement of BAAI/bge-large-en-v1.5.
    The prefix is applied inside embed_query(); the RetrievalEngine does NOT
    modify the query string.

Step 3: Search the FAISS index.
    Method: FAISSManager.search(query_vector, top_k, metadata_filters)
    The FAISSManager normalizes the query vector to unit length internally.
    Scores are cosine similarities in [-1, 1].
    For L2-normalized IndexFlatIP, effective range is [0, 1].
    Results are returned in descending score order.

Step 4: Convert to RetrievalResult.
    Each FAISSManager.SearchResult is converted to a flat RetrievalResult
    that carries all Chunk metadata plus score and rank.

Step 5: Apply min_score threshold (optional post-filter).
    Results with score < min_score are discarded.
    Default: 0.0 (no filtering).

Step 6: Record metrics.
    RetrievalMetrics is updated after every successful retrieve() call.

Step 7: Return list[RetrievalResult] in descending score order.


3. Relationship Between BGE Embeddings and FAISS
-------------------------------------------------

Embedding vectors are produced by BAAI/bge-large-en-v1.5:
    - Dimension: 1024
    - L2-normalized to unit length by BGEEmbedder (normalize_embeddings=True)
    - Stored in FAISSManager with additional L2-normalization for safety

FAISS index type: IndexFlatIP (default)
    - Uses inner product (dot product) as the distance metric.
    - When both query and stored vectors are L2-normalized:
        inner_product(u, v) == cosine_similarity(u, v)
    - Score of 1.0 means the query vector and chunk vector are identical.
    - Score near 0.0 means the vectors are nearly orthogonal.

Query encoding uses the BGE asymmetric prefix:
    "Represent this sentence for searching relevant passages: " + query

Chunk encoding uses no prefix (passages are encoded as-is).

This asymmetry is enforced in the IEmbeddingProvider interface and the
BGEEmbedder implementation.  The RetrievalEngine does NOT apply or modify
any prefix; that responsibility belongs entirely to the embedding provider.


4. RetrievalResult Structure
----------------------------

RetrievalResult is a frozen (immutable) dataclass.  All fields are set at
construction time and cannot be changed.

Fields:

    chunk_id        str              16-char hex digest from the Chunk.
    document_id     str              Parent document identifier.
    source_file     str              Absolute path to the source PDF.
    page_numbers    tuple[int, ...]  1-indexed page numbers the chunk spans.
    text            str              The chunk text content.
    section_heading str | None       Active section heading, or None.
    document_type   str | None       Document kind label, or None.
    chunk_index     int              0-indexed position within the document.
    score           float            Cosine similarity in [-1, 1].
    rank            int              1-indexed rank (1 = most similar).

Method:

    to_dict() -> dict[str, Any]
        Returns a plain JSON-serialisable dictionary.
        page_numbers is converted to a list.

Example:

    {
      "chunk_id": "a1b2c3d4e5f60718",
      "document_id": "doc-001",
      "source_file": "/data/doc-001.pdf",
      "page_numbers": [3],
      "text": "Elevated troponin is a marker for myocardial infarction.",
      "section_heading": "Lab Results",
      "document_type": "clinical_note",
      "chunk_index": 2,
      "score": 0.8731,
      "rank": 1
    }


5. Top-K Behavior
-----------------

top_k controls the maximum number of results returned.

    - Default: configured in rag.yaml -> rag.retrieval.top_k (default: 5)
    - Maximum: configured in rag.yaml -> rag.retrieval.max_top_k (default: 50)
    - Requests for top_k > max_top_k raise ValueError.
    - If the FAISS index has fewer than top_k entries, all entries are returned.
    - If min_score filtering removes results, the final count may be < top_k.
    - Results are always ordered by descending score.


6. Edge Cases and Error Handling
---------------------------------

Empty or whitespace-only query:
    Raises ValueError before any embedding call.

top_k < 1:
    Raises ValueError before any embedding call.

top_k > max_top_k:
    Raises ValueError before any embedding call.

Empty FAISS index:
    Returns [] (not an error).  FAISSManager.search() already handles this.

Embedding provider not initialized:
    embed_query() raises RuntimeError, which is propagated by retrieve().
    The caller must call provider.initialize() before creating the engine.

Wrong embedding dimension:
    FAISSManager.search() raises ValueError if the query vector dimension
    does not match the index dimension.  This is propagated by retrieve().

FAISS returns fewer results than top_k:
    This is normal when the index has fewer entries than top_k.
    retrieve() returns however many results are available.

FAISS invalid result (-1 faiss_id):
    FAISSManager filters these silently.  retrieve() never sees them.

All results filtered by min_score:
    Returns [].  The empty_results_count metric is incremented.

Unknown chunk_id in metadata map:
    FAISSManager silently skips such entries.  retrieve() never sees them.

None embedding_provider or faiss_manager:
    RetrievalEngine constructor raises TypeError.


7. Configuration (configs/rag.yaml)
------------------------------------

    rag:
      retrieval:
        top_k: 5           # Default top-K per query
        max_top_k: 50       # Hard upper bound; requests above this are rejected
        similarity_metric: "cosine"   # For documentation; index is always IndexFlatIP
        min_score_threshold: 0.0      # No post-filter by default


8. RetrievalMetrics
-------------------

RetrievalMetrics tracks per-session statistics for one RetrievalEngine instance.

Fields:
    total_queries             int     Number of retrieve() calls.
    total_results_returned    int     Total RetrievalResult objects returned.
    total_retrieval_time_s    float   Cumulative wall-clock seconds.
    empty_results_count       int     Queries that returned zero results.

Property:
    avg_retrieval_time_ms     float   Mean latency per query in milliseconds.

Method:
    to_dict() -> dict[str, Any]


9. Testing
----------

Unit tests:
    tests/unit/rag/test_retrieval_engine.py
        - RetrievalResult construction and to_dict()
        - RetrievalMetrics counters and latency
        - Constructor validation (None, invalid top_k)
        - Query validation (empty, whitespace)
        - top_k validation (0, negative, above max)
        - Successful retrieval (result structure, fields, ranking)
        - Empty FAISS index -> []
        - Fewer results than top_k
        - min_score threshold
        - metadata_filters passthrough
        - Metrics after retrieve()
        - get_index_stats()
        - Provider not initialized (RuntimeError propagation)
        - Wrong embedding dimension (ValueError propagation)
        - IRetrievalEngine interface compliance

Integration test:
    tests/integration/rag/test_retrieval_pipeline.py
        Full pipeline: synthetic clinical Chunks -> mocked BGEEmbedder
        -> FAISSManager -> RetrievalEngine.retrieve() -> RetrievalResult

Run unit tests:
    uv run pytest tests/unit/rag/test_retrieval_engine.py -v --no-cov

Run integration tests:
    uv run pytest tests/integration/rag/test_retrieval_pipeline.py -v --no-cov

Run all RAG unit tests (Phases 4.2, 4.3, 4.4):
    uv run pytest tests/unit/rag/ -v --no-cov


10. Not Implemented in Phase 4.4 Milestone 1
---------------------------------------------

The following components are intentionally excluded:
    BM25 retrieval
    Hybrid search (vector + BM25)
    Query planner
    Context builder
    Prompt builder
    LLM integration
    GPU acceleration
    Answer generator
    RAG evaluator
    Re-ranking
    Dense passage retrieval (DPR) fine-tuning
