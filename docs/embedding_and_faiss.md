Phase 4.3 Embedding Engine and FAISS Manager
Adaptive Distributed Framework v2.0
============================================

This document describes the embedding and vector store layer (Phase 4.3).
It is the second stage of the Phase 4 Knowledge Layer pipeline:

    UnifiedDocument
        -> SemanticChunker (Phase 4.2)
        -> list[Chunk]
        -> BGEEmbedder           (Phase 4.3)
        -> list[Embedding]
        -> FAISSManager          (Phase 4.3)
        -> FAISS index on disk

No GPU acceleration is implemented in Phase 4.3.
GPU acceleration will be integrated separately in a later phase.
The embedding device is configurable and defaults to CPU.


1. Embedding Model
------------------

Model:       BAAI/bge-large-en-v1.5
Library:     sentence-transformers
Dimension:   1024
Max tokens:  512 (handled automatically by sentence-transformers)
License:     MIT
Device:      CPU (default); CUDA configurable but not implemented in Phase 4.3

The model is loaded lazily: BGEEmbedder.initialize() must be called before
embed_chunks() or embed_query().  Calling initialize() twice is a no-op.


2. Asymmetric Query/Chunk Encoding
-----------------------------------

BAAI/bge-large-en-v1.5 uses an asymmetric retrieval architecture.  The
document passages (chunks) and user queries are encoded differently.

QUERY ENCODING:
    The instruction prefix is prepended before encoding:
        "Represent this sentence for searching relevant passages: " + query

    Method:  BGEEmbedder.embed_query(query: str) -> list[float]

CHUNK ENCODING:
    No prefix is added.  Passages are encoded as-is.

    Method:  BGEEmbedder.embed_chunks(chunks: list[Chunk]) -> list[Embedding]

This asymmetry is a hard requirement of the BGE model training protocol.
Violating it (e.g. adding the prefix to chunks, or omitting it from queries)
degrades retrieval quality because the model was trained with this distinction.

The asymmetry is enforced at the interface level (IEmbeddingProvider contract),
at the implementation level (BGEEmbedder), and verified in unit tests
(test_bge_embedder.py::TestAsymmetricEncoding).


3. Embedding Cache
------------------

Cache key:       SHA-256(chunk.text.encode("utf-8")).hexdigest()
Storage:         SQLite database
Default path:    outputs/rag/cache/embeddings.db
Cache hit:       Returns stored vector; model is NOT called
Cache miss:      Calls model; stores result in cache
Invalidation:    EmbeddingCache.invalidate(key) removes one entry
Force reembed:   BGEEmbedder.embed_chunks(chunks, force_reembed=True) bypasses cache

The SHA-256 key means that two chunks with identical text share one cache entry,
regardless of their chunk_id or document_id.  This is correct behavior: if the
text is the same, the embedding is identical.

The cache uses a SQLite database, which is:
    - Persistent across process restarts
    - Safe for concurrent readers
    - Tolerant of corrupt entries (corrupt rows are deleted and treated as misses)

Vectors are stored as packed float32 bytes (struct.pack("1024f", ...)).
The stored size is 1024 * 4 = 4096 bytes per entry.

If cache_enabled is False in rag.yaml, no cache is created and every chunk
is embedded by the model.


4. Ray Batching
---------------

Module:  src/adaptive_framework/rag/embedding/batch_embedding_worker.py

The embedding worker uses the EXISTING Ray infrastructure from Phase 3.
No new actor infrastructure is created.
No modification to the scheduler or coordinator.

API:
    embed_batch_local(chunks, model_name, device, batch_size, cache_path)
        Runs synchronously in the calling process.  Used on Windows (where
        Ray is not available) and in unit tests.

    embed_batch_ray(chunks, model_name, device, batch_size, cache_path)
        Schedules the embedding as a Ray remote task.  Falls back to
        embed_batch_local() if Ray is not available.

Default batch size: 64 (configurable via batch_size parameter and rag.yaml)

On Windows development machines, Ray is not available.  The code detects
this and falls back to local embedding automatically.


5. FAISS Index Types
---------------------

Three index types are supported via FAISSIndexBuilder:

flat   ->  faiss.IndexFlatIP (exact cosine search via inner product)
    Default.  No training required.
    O(N) query complexity.
    Correct for corpora up to approximately 100,000 chunks.
    Recommended for Phase 4.3.

ivf    ->  faiss.IndexIVFFlat (inverted file, approximate)
    Partitions space into n_clusters Voronoi cells.
    Requires training on a representative sample of vectors.
    Good for 100,000 to 10,000,000 vectors.
    Configure n_clusters in rag.yaml.

hnsw   ->  faiss.IndexHNSWFlat (hierarchical navigable small world)
    No training required.
    Very fast queries, high recall.
    Good for 10,000 to 100,000,000 vectors.

All index types use METRIC_INNER_PRODUCT.  Because all vectors are
L2-normalized before insertion, inner product equals cosine similarity.


6. Vector Normalization
-----------------------

All vectors are L2-normalized to unit length before insertion into FAISS.
Query vectors are also L2-normalized before search.

This means:
    inner_product(u, v) == cosine_similarity(u, v)
    when ||u|| == ||v|| == 1

Using IndexFlatIP with normalized vectors is the standard approach for
exact cosine similarity search in FAISS.

The normalization is performed inside FAISSManager.add_embeddings() and
FAISSManager.search().  Callers do not need to normalize.


7. Persistence
--------------

Three files are written by FAISSPersistence.save():

    outputs/rag/index/index.faiss
        Binary FAISS index (written by faiss.write_index).
        Contains all vector data.

    outputs/rag/index/metadata.pkl
        Python pickle file containing:
            faiss_id_to_chunk_id  dict[int, str]
            chunk_id_to_chunk     dict[str, Chunk]
            stored_vectors        dict[str, np.ndarray]
            dim, index_type, model_name

    outputs/rag/index/index_stats.json
        Human-readable JSON summary:
            total_chunks          Number of vectors in the index
            total_documents       Number of unique document_ids
            embedding_dimension   Vector dimension (1024)
            index_type            'flat', 'ivf', or 'hnsw'
            model_name            'BAAI/bge-large-en-v1.5'

Loading is performed by FAISSPersistence.load(manager), which replaces
the internal state of an existing FAISSManager with the persisted state.

Safe round-trip guarantee:
    A FAISSManager saved with persist() can be reloaded with load() and
    search() / delete_document() work correctly.


8. Metadata Mapping
--------------------

FAISSManager maintains two complementary dictionaries:

    _faiss_id_to_chunk_id : dict[int, str]
        Maps a FAISS integer row index to the chunk_id string.
        Used to look up the chunk after a FAISS search returns faiss_ids.

    _chunk_id_to_chunk : dict[str, Chunk]
        Maps a chunk_id string to the full Chunk object.
        Used to return the Chunk in SearchResult.

Both are rebuilt from scratch when delete_document() triggers an index rebuild.

A third dictionary supports rebuild:
    _stored_vectors : dict[str, np.ndarray]
        Maps chunk_id to the normalized vector (dim,) array.
        Used during rebuild_after_deletion to reconstruct the FAISS index
        from the surviving chunks.


9. Deletion and Rebuild
------------------------

FAISSManager.delete_document(document_id):
    1. Identifies all chunk_ids belonging to the document.
    2. Removes them from _chunk_id_to_chunk and _stored_vectors.
    3. Calls _rebuild_index(), which:
       a. Creates a new empty FAISS index.
       b. Adds all remaining vectors from _stored_vectors.
       c. Rebuilds _faiss_id_to_chunk_id with new consecutive integer keys.
    4. Records the deletion in FAISSMetrics.

Deletion is O(N) (rebuild cost).  For Phase 4.3, this is acceptable.
Incremental deletion (without full rebuild) is a future optimization.


10. Configuration
-----------------

rag.yaml:

    rag:
      embedding:
        model: "BAAI/bge-large-en-v1.5"
        batch_size: 64
        device: "cpu"
        cache_enabled: true
        cache_path: "outputs/rag/cache/embeddings.db"

      vector_store:
        backend: "faiss"
        index_type: "flat"
        index_path: "outputs/rag/index"
        n_clusters: 100

      embedder:
        model: "BAAI/bge-large-en-v1.5"
        device: "cpu"
        batch_size: 64
        embedding_dim: 1024

The 'embedder' section is retained for ConfigManager backward compatibility.
The 'embedding' section is the primary Phase 4.3 configuration.


11. Performance Targets
------------------------

The project target for Phase 4.3 is:
    Embedding throughput >= 50 chunks/s on CPU with batch_size=64
    FAISS search <= 20 ms for a 10,000-chunk IndexFlatIP index

Do not claim these targets are met unless they are measured.
GPU acceleration is not implemented in Phase 4.3.


12. Testing
-----------

Unit tests:
    tests/unit/rag/test_bge_embedder.py     BGEEmbedder (mocked model)
    tests/unit/rag/test_embedding_cache.py  EmbeddingCache (SQLite)
    tests/unit/rag/test_faiss_manager.py    FAISSManager + Builder + Persistence

Integration test:
    tests/integration/rag/test_embedding_faiss_pipeline.py
    Full pipeline: Chunks -> mocked embedder -> FAISSManager -> persist -> reload -> search

Run unit tests:
    uv run pytest tests/unit/rag/ -v

Run integration tests:
    uv run pytest tests/integration/rag/ -v


13. Not Implemented in Phase 4.3
----------------------------------

The following components are intentionally excluded from Phase 4.3:
    BM25 retrieval
    Hybrid search (vector + BM25)
    Query planner
    Context builder
    Prompt builder
    LLM integration
    GPU acceleration
    Answer generator
    RAG evaluator
