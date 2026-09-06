"""Unit tests for FAISSManager, FAISSIndexBuilder, FAISSPersistence (Phase 4.3).

All tests use 1024-dimensional dummy vectors.  No BGE model is loaded.

Tested requirements:
    - initialization (flat/ivf/hnsw index types)
    - add embeddings
    - vector normalization before insertion
    - search returns expected chunks with scores
    - metadata mapping (faiss_id_to_chunk_id, chunk_id_to_chunk)
    - persistence round-trip (save and reload)
    - load existing index restores search capability
    - delete_document removes correct chunks
    - rebuild after deletion maintains search correctness
    - empty index search returns []
    - duplicate chunk_id handling (second add is a no-op)
    - FAISSIndexBuilder factory
    - index_stats content
"""

from __future__ import annotations

import pickle
import pytest
import numpy as np
from pathlib import Path

from adaptive_framework.models.chunk import Chunk, _make_chunk_id
from adaptive_framework.rag.models.embedding import Embedding
from adaptive_framework.rag.vector_store.faiss_manager import FAISSManager, SearchResult
from adaptive_framework.rag.vector_store.faiss_index_builder import FAISSIndexBuilder
from adaptive_framework.rag.vector_store.faiss_persistence import FAISSPersistence


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

DIM = 1024


def _make_chunk(doc_id: str = "doc-001", idx: int = 0, text: str = "Sample.") -> Chunk:
    return Chunk(
        chunk_id=_make_chunk_id(doc_id, idx),
        document_id=doc_id,
        source_file="/data/doc.pdf",
        chunk_index=idx,
        page_numbers=(1,),
        text=text,
        section_heading=None,
        document_type=None,
        character_count=len(text),
        word_count=len(text.split()),
        start_char_in_page=0,
        end_char_in_page=len(text),
    )


def _make_embedding(chunk: Chunk, seed: int = 0) -> Embedding:
    """Create an Embedding with a deterministic unit vector."""
    rng = np.random.default_rng(seed)
    vec = rng.random(DIM).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return Embedding(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        vector=tuple(float(x) for x in vec),
        model_name="BAAI/bge-large-en-v1.5",
        embedding_time_seconds=0.001,
        created_at="2026-08-31T00:00:00+00:00",
    )


def _build_manager_with_data(
    n_chunks: int = 5, n_docs: int = 1
) -> tuple[FAISSManager, list[Chunk], list[Embedding]]:
    """Return a FAISSManager populated with n_chunks across n_docs documents."""
    manager = FAISSManager(dim=DIM)
    chunks: list[Chunk] = []
    embeddings: list[Embedding] = []
    for doc_i in range(n_docs):
        for chunk_i in range(n_chunks // n_docs + (1 if doc_i < n_chunks % n_docs else 0)):
            global_idx = doc_i * (n_chunks // n_docs) + chunk_i
            chunk = _make_chunk(doc_id=f"doc-{doc_i:03d}", idx=chunk_i, text=f"Doc{doc_i} Chunk{chunk_i}")
            emb = _make_embedding(chunk, seed=global_idx)
            chunks.append(chunk)
            embeddings.append(emb)
    manager.add_embeddings_with_chunks(embeddings, chunks)
    return manager, chunks, embeddings


# ---------------------------------------------------------------------------
# FAISSIndexBuilder
# ---------------------------------------------------------------------------

class TestFAISSIndexBuilder:
    def test_flat_index(self):
        """IndexFlatIP must be created for index_type='flat'."""
        import faiss
        builder = FAISSIndexBuilder(dim=DIM, index_type="flat")
        index = builder.build()
        assert isinstance(index, faiss.IndexFlatIP)

    def test_ivf_index(self):
        """IndexIVFFlat must be created for index_type='ivf'."""
        import faiss
        builder = FAISSIndexBuilder(dim=DIM, index_type="ivf", n_clusters=10)
        index = builder.build()
        assert isinstance(index, faiss.IndexIVFFlat)

    def test_hnsw_index(self):
        """IndexHNSWFlat must be created for index_type='hnsw'."""
        import faiss
        builder = FAISSIndexBuilder(dim=DIM, index_type="hnsw")
        index = builder.build()
        assert isinstance(index, faiss.IndexHNSWFlat)

    def test_invalid_index_type_raises(self):
        """Unknown index_type must raise ValueError."""
        with pytest.raises(ValueError, match="index_type must be one of"):
            FAISSIndexBuilder(dim=DIM, index_type="unknown")

    def test_invalid_dim_raises(self):
        """dim < 1 must raise ValueError."""
        with pytest.raises(ValueError, match="dim must be"):
            FAISSIndexBuilder(dim=0)

    def test_builder_reports_index_type(self):
        """FAISSIndexBuilder.index_type must reflect the configured value."""
        b = FAISSIndexBuilder(dim=DIM, index_type="hnsw")
        assert b.index_type == "hnsw"
        assert b.dim == DIM


# ---------------------------------------------------------------------------
# FAISSManager — initialization
# ---------------------------------------------------------------------------

class TestFAISSManagerInit:
    def test_default_init(self):
        """FAISSManager must initialize with zero chunks."""
        manager = FAISSManager(dim=DIM)
        assert manager.total_chunks == 0

    def test_repr(self):
        """FAISSManager.__repr__ must be a non-empty string."""
        manager = FAISSManager(dim=DIM)
        r = repr(manager)
        assert "FAISSManager" in r

    def test_empty_search(self):
        """search() on empty index must return empty list."""
        manager = FAISSManager(dim=DIM)
        rng = np.random.default_rng(0)
        query = rng.random(DIM).tolist()
        results = manager.search(query, top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# FAISSManager — add_embeddings
# ---------------------------------------------------------------------------

class TestAddEmbeddings:
    def test_add_single_embedding(self):
        """Adding one embedding must result in total_chunks == 1."""
        manager = FAISSManager(dim=DIM)
        chunk = _make_chunk()
        emb = _make_embedding(chunk, seed=0)
        manager.add_embeddings_with_chunks([emb], [chunk])
        assert manager.total_chunks == 1

    def test_add_multiple_embeddings(self):
        """Adding N embeddings must result in total_chunks == N."""
        manager, chunks, embeddings = _build_manager_with_data(n_chunks=7)
        assert manager.total_chunks == 7

    def test_add_empty_list_is_noop(self):
        """Adding an empty list must not change the index."""
        manager = FAISSManager(dim=DIM)
        manager.add_embeddings([])
        assert manager.total_chunks == 0

    def test_duplicate_chunk_id_skipped(self):
        """Adding the same chunk_id twice must not increase the index count."""
        manager = FAISSManager(dim=DIM)
        chunk = _make_chunk()
        emb = _make_embedding(chunk, seed=0)
        manager.add_embeddings_with_chunks([emb], [chunk])
        manager.add_embeddings_with_chunks([emb], [chunk])  # duplicate
        assert manager.total_chunks == 1

    def test_wrong_dim_raises(self):
        """Adding an embedding with wrong dimension must raise ValueError."""
        manager = FAISSManager(dim=DIM)
        chunk = _make_chunk()
        bad_emb = Embedding(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            vector=tuple([0.0] * 128),  # wrong dim
            model_name="BAAI/bge-large-en-v1.5",
            embedding_time_seconds=0.0,
            created_at="2026-08-31T00:00:00+00:00",
        )
        with pytest.raises(ValueError, match="dim mismatch"):
            manager.add_embeddings([bad_emb])

    def test_mismatched_lengths_raises(self):
        """add_embeddings_with_chunks with mismatched lengths must raise ValueError."""
        manager = FAISSManager(dim=DIM)
        chunk = _make_chunk()
        emb = _make_embedding(chunk)
        with pytest.raises(ValueError, match="equal length"):
            manager.add_embeddings_with_chunks([emb], [chunk, chunk])


# ---------------------------------------------------------------------------
# FAISSManager — vector normalization
# ---------------------------------------------------------------------------

class TestVectorNormalization:
    def test_vectors_are_unit_length_in_index(self):
        """After insertion, all stored vectors must be unit-length (L2 norm == 1)."""
        import faiss
        manager = FAISSManager(dim=DIM)
        chunk = _make_chunk()
        # Use a non-unit vector to test that normalization is applied.
        raw_vec = np.ones(DIM, dtype=np.float32) * 2.0  # norm = sqrt(DIM * 4)
        emb = Embedding(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            vector=tuple(float(x) for x in raw_vec),
            model_name="BAAI/bge-large-en-v1.5",
            embedding_time_seconds=0.0,
            created_at="2026-08-31T00:00:00+00:00",
        )
        manager.add_embeddings_with_chunks([emb], [chunk])

        # Reconstruct the stored vector from the FAISS index.
        stored = faiss.rev_swig_ptr(manager._index.get_xb(), DIM)  # type: ignore[attr-defined]
        # Simpler approach: just verify the search score for the same vector is close to 1.0.
        query_norm = raw_vec / np.linalg.norm(raw_vec)
        results = manager.search(query_norm.tolist(), top_k=1)
        assert len(results) == 1
        # Score must be close to 1.0 (cosine similarity of identical unit vectors).
        assert results[0].score == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# FAISSManager — search
# ---------------------------------------------------------------------------

class TestSearch:
    def test_search_returns_correct_chunk(self):
        """Searching with a chunk's own vector must return that chunk as top-1."""
        manager = FAISSManager(dim=DIM)
        chunk = _make_chunk(text="The CRISPR-Cas9 system enables precise genome editing.")
        emb = _make_embedding(chunk, seed=42)
        manager.add_embeddings_with_chunks([emb], [chunk])

        # Use the same vector as query (normalized).
        query = np.array(emb.vector, dtype=np.float32)
        query /= np.linalg.norm(query)
        results = manager.search(query.tolist(), top_k=1)

        assert len(results) == 1
        assert results[0].chunk.chunk_id == chunk.chunk_id

    def test_search_returns_at_most_top_k(self):
        """search() must return at most top_k results."""
        manager, chunks, embeddings = _build_manager_with_data(n_chunks=10)
        query = list(embeddings[0].vector)
        results = manager.search(query, top_k=3)
        assert len(results) <= 3

    def test_search_scores_are_valid(self):
        """All returned scores must be in [-1, 1] (cosine similarity range)."""
        manager, chunks, embeddings = _build_manager_with_data(n_chunks=5)
        query = list(embeddings[0].vector)
        results = manager.search(query, top_k=5)
        for r in results:
            assert -1.0 <= r.score <= 1.0 + 1e-5

    def test_search_wrong_dim_raises(self):
        """Searching with wrong-dimension query must raise ValueError."""
        manager, _, _ = _build_manager_with_data(n_chunks=3)
        with pytest.raises(ValueError, match="query_vector dim mismatch"):
            manager.search([0.1] * 128, top_k=1)

    def test_search_metadata_document_filter(self):
        """metadata_filters['document_id'] must filter results."""
        manager = FAISSManager(dim=DIM)
        chunk_a = _make_chunk(doc_id="doc-A", idx=0)
        chunk_b = _make_chunk(doc_id="doc-B", idx=0)
        emb_a = _make_embedding(chunk_a, seed=1)
        emb_b = _make_embedding(chunk_b, seed=2)
        manager.add_embeddings_with_chunks([emb_a, emb_b], [chunk_a, chunk_b])

        query = list(emb_a.vector)
        results = manager.search(query, top_k=5, metadata_filters={"document_id": "doc-A"})
        assert all(r.document_id == "doc-A" for r in results)

    def test_search_ranks_are_1_indexed(self):
        """Search result ranks must start at 1."""
        manager, chunks, embeddings = _build_manager_with_data(n_chunks=3)
        query = list(embeddings[0].vector)
        results = manager.search(query, top_k=3)
        assert results[0].faiss_rank == 1


# ---------------------------------------------------------------------------
# FAISSManager — metadata mapping
# ---------------------------------------------------------------------------

class TestMetadataMapping:
    def test_faiss_id_to_chunk_id_populated(self):
        """After adding, faiss_id_to_chunk_id must have N entries."""
        manager, chunks, _ = _build_manager_with_data(n_chunks=4)
        assert len(manager._faiss_id_to_chunk_id) == 4

    def test_chunk_id_to_chunk_populated(self):
        """After adding, chunk_id_to_chunk must have N entries."""
        manager, chunks, _ = _build_manager_with_data(n_chunks=4)
        assert len(manager._chunk_id_to_chunk) == 4

    def test_chunk_id_to_chunk_correct(self):
        """chunk_id_to_chunk must map to the correct Chunk objects."""
        manager, chunks, _ = _build_manager_with_data(n_chunks=3)
        for chunk in chunks:
            stored = manager._chunk_id_to_chunk.get(chunk.chunk_id)
            assert stored is not None
            assert stored.chunk_id == chunk.chunk_id

    def test_index_stats_content(self):
        """get_index_stats() must return required fields with correct values."""
        manager, chunks, _ = _build_manager_with_data(n_chunks=5, n_docs=2)
        stats = manager.get_index_stats()
        assert "total_chunks" in stats
        assert "total_documents" in stats
        assert "embedding_dimension" in stats
        assert "index_type" in stats
        assert "model_name" in stats
        assert stats["total_chunks"] == 5
        assert stats["embedding_dimension"] == DIM


# ---------------------------------------------------------------------------
# FAISSManager — delete_document
# ---------------------------------------------------------------------------

class TestDeleteDocument:
    def test_delete_removes_chunks(self):
        """delete_document must remove all chunks for that document."""
        manager, chunks, _ = _build_manager_with_data(n_chunks=6, n_docs=2)
        initial_total = manager.total_chunks
        n_removed = manager.delete_document("doc-000")
        assert n_removed > 0
        assert manager.total_chunks == initial_total - n_removed

    def test_delete_unknown_doc_returns_zero(self):
        """delete_document for an unknown doc must return 0."""
        manager, _, _ = _build_manager_with_data(n_chunks=3)
        n_removed = manager.delete_document("doc-nonexistent")
        assert n_removed == 0

    def test_delete_makes_chunks_unsearchable(self):
        """After deletion, the removed document's chunks must not appear in search."""
        manager = FAISSManager(dim=DIM)
        chunk_a = _make_chunk(doc_id="doc-A", idx=0)
        chunk_b = _make_chunk(doc_id="doc-B", idx=0)
        emb_a = _make_embedding(chunk_a, seed=10)
        emb_b = _make_embedding(chunk_b, seed=20)
        manager.add_embeddings_with_chunks([emb_a, emb_b], [chunk_a, chunk_b])

        manager.delete_document("doc-A")

        # Search should not return doc-A chunks.
        query = list(emb_a.vector)
        results = manager.search(query, top_k=5)
        doc_ids = [r.document_id for r in results]
        assert "doc-A" not in doc_ids

    def test_rebuild_after_deletion_preserves_remaining(self):
        """After rebuilding, remaining chunks must still be searchable."""
        manager = FAISSManager(dim=DIM)
        chunk_keep = _make_chunk(doc_id="doc-keep", idx=0, text="Keep this.")
        chunk_del = _make_chunk(doc_id="doc-del", idx=0, text="Delete this.")
        emb_keep = _make_embedding(chunk_keep, seed=1)
        emb_del = _make_embedding(chunk_del, seed=2)
        manager.add_embeddings_with_chunks([emb_keep, emb_del], [chunk_keep, chunk_del])

        manager.delete_document("doc-del")

        query = list(emb_keep.vector)
        results = manager.search(query, top_k=5)
        ids = [r.chunk.chunk_id for r in results]
        assert chunk_keep.chunk_id in ids

    def test_empty_index_after_all_deleted(self):
        """After deleting all documents, search must return empty list."""
        manager = FAISSManager(dim=DIM)
        chunk = _make_chunk(doc_id="doc-only")
        emb = _make_embedding(chunk)
        manager.add_embeddings_with_chunks([emb], [chunk])
        manager.delete_document("doc-only")
        query = list(emb.vector)
        assert manager.search(query, top_k=5) == []


# ---------------------------------------------------------------------------
# FAISSPersistence — round trip
# ---------------------------------------------------------------------------

class TestFAISSPersistence:
    def test_persist_creates_files(self, tmp_path: Path):
        """save() must create index.faiss, metadata.pkl, index_stats.json."""
        persistence = FAISSPersistence(index_dir=tmp_path)
        manager, _, _ = _build_manager_with_data(n_chunks=3)
        persistence.save(manager)

        assert persistence.index_path.exists()
        assert persistence.metadata_path.exists()
        assert persistence.stats_path.exists()

    def test_round_trip_preserves_chunk_count(self, tmp_path: Path):
        """After save+load, total_chunks must match the original."""
        persistence = FAISSPersistence(index_dir=tmp_path)
        manager1, chunks, embeddings = _build_manager_with_data(n_chunks=5)
        persistence.save(manager1)

        manager2 = FAISSManager(dim=DIM)
        persistence.load(manager2)
        assert manager2.total_chunks == manager1.total_chunks

    def test_round_trip_metadata_mapping(self, tmp_path: Path):
        """After save+load, chunk_id_to_chunk must contain all original chunks."""
        persistence = FAISSPersistence(index_dir=tmp_path)
        manager1, chunks, embeddings = _build_manager_with_data(n_chunks=4)
        persistence.save(manager1)

        manager2 = FAISSManager(dim=DIM)
        persistence.load(manager2)
        for chunk in chunks:
            assert chunk.chunk_id in manager2._chunk_id_to_chunk

    def test_round_trip_search_returns_correct_chunk(self, tmp_path: Path):
        """After save+load, search must still return the expected chunk."""
        persistence = FAISSPersistence(index_dir=tmp_path)
        chunk = _make_chunk(text="Persistent chunk for FAISS round trip.")
        emb = _make_embedding(chunk, seed=99)
        manager1 = FAISSManager(dim=DIM)
        manager1.add_embeddings_with_chunks([emb], [chunk])
        persistence.save(manager1)

        manager2 = FAISSManager(dim=DIM)
        persistence.load(manager2)
        query = np.array(emb.vector, dtype=np.float32)
        query /= np.linalg.norm(query)
        results = manager2.search(query.tolist(), top_k=1)
        assert len(results) == 1
        assert results[0].chunk.chunk_id == chunk.chunk_id

    def test_load_missing_index_raises(self, tmp_path: Path):
        """load() on a nonexistent index must raise FileNotFoundError."""
        persistence = FAISSPersistence(index_dir=tmp_path / "nonexistent")
        manager = FAISSManager(dim=DIM)
        with pytest.raises(FileNotFoundError):
            persistence.load(manager)

    def test_read_stats_returns_dict(self, tmp_path: Path):
        """read_stats() must return a non-empty dict with required keys."""
        persistence = FAISSPersistence(index_dir=tmp_path)
        manager, _, _ = _build_manager_with_data(n_chunks=3)
        persistence.save(manager)
        stats = persistence.read_stats()
        assert "total_chunks" in stats
        assert "embedding_dimension" in stats
        assert stats["total_chunks"] == 3

    def test_read_stats_missing_returns_empty(self, tmp_path: Path):
        """read_stats() with no stats file must return empty dict."""
        persistence = FAISSPersistence(index_dir=tmp_path / "no_stats")
        assert persistence.read_stats() == {}

    def test_exists_false_before_save(self, tmp_path: Path):
        """exists() must return False before save() is called."""
        persistence = FAISSPersistence(index_dir=tmp_path / "empty")
        assert not persistence.exists()

    def test_exists_true_after_save(self, tmp_path: Path):
        """exists() must return True after save() is called."""
        persistence = FAISSPersistence(index_dir=tmp_path)
        manager, _, _ = _build_manager_with_data(n_chunks=2)
        persistence.save(manager)
        assert persistence.exists()
