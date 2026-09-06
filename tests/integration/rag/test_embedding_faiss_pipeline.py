"""Integration test: Embedding Engine + FAISS Manager pipeline (Phase 4.3).

This test exercises the complete ingestion path:

    Synthetic Chunk objects
        -> Mock BGEEmbedder (returns fixed 1024-dim vectors)
        -> list[Embedding]
        -> FAISSManager (add + register)
        -> FAISSPersistence (save)
        -> FAISSManager reload (load)
        -> search
        -> verify results

Verified:
    - embedding dimension == 1024
    - index contains expected number of chunks
    - metadata mapping survives reload
    - search returns the expected chunk
    - scores are valid floats in [-1, 1]

The real BAAI/bge-large-en-v1.5 model is NOT used.  The BGEEmbedder is
tested via a mock that returns deterministic 1024-dimensional unit vectors.
"""

from __future__ import annotations

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from adaptive_framework.models.chunk import Chunk, _make_chunk_id
from adaptive_framework.rag.embedding.bge_embedder import BGEEmbedder, _BGE_LARGE_DIM
from adaptive_framework.rag.vector_store.faiss_manager import FAISSManager
from adaptive_framework.rag.vector_store.faiss_persistence import FAISSPersistence


DIM = _BGE_LARGE_DIM  # 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(doc_id: str, idx: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=_make_chunk_id(doc_id, idx),
        document_id=doc_id,
        source_file=f"/data/{doc_id}.pdf",
        chunk_index=idx,
        page_numbers=(1,),
        text=text,
        section_heading=None,
        document_type="research_paper",
        character_count=len(text),
        word_count=len(text.split()),
        start_char_in_page=0,
        end_char_in_page=len(text),
    )


def _make_mock_model(dim: int = DIM) -> MagicMock:
    """Return a mock SentenceTransformer that returns deterministic unit vectors."""
    mock = MagicMock()

    def fake_encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True):
        n = len(texts)
        rng = np.random.default_rng(sum(len(t) for t in texts))
        vecs = rng.random((n, dim)).astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs /= np.where(norms == 0, 1, norms)
        return vecs

    mock.encode.side_effect = fake_encode
    return mock


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

BIOMEDICAL_CHUNKS = [
    ("doc-001", 0, "CRISPR-Cas9 enables precise editing of the human genome."),
    ("doc-001", 1, "Off-target effects remain a challenge in CRISPR applications."),
    ("doc-001", 2, "The guide RNA directs Cas9 to the target DNA sequence."),
    ("doc-002", 0, "Alzheimer disease is characterized by amyloid plaques."),
    ("doc-002", 1, "Tau protein aggregation leads to neurofibrillary tangles."),
    ("doc-003", 0, "mRNA vaccines produce spike protein to train the immune system."),
]


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

class TestEmbeddingFAISSPipeline:
    """Full pipeline: Chunks -> Embeddings -> FAISSManager -> persist -> reload -> search."""

    @pytest.fixture()
    def pipeline(self, tmp_path: Path):
        """Set up the full embedding + FAISS pipeline with mock model."""
        chunks = [_make_chunk(doc_id, idx, text) for doc_id, idx, text in BIOMEDICAL_CHUNKS]

        with patch(
            "adaptive_framework.rag.embedding.bge_embedder.SentenceTransformer",
            return_value=_make_mock_model(),
        ):
            embedder = BGEEmbedder(model_name="BAAI/bge-large-en-v1.5", device="cpu")
            embedder.initialize()
            embeddings = embedder.embed_chunks(chunks)

        manager = FAISSManager(dim=DIM, model_name="BAAI/bge-large-en-v1.5")
        manager.add_embeddings_with_chunks(embeddings, chunks)

        persistence = FAISSPersistence(index_dir=tmp_path / "index")
        persistence.save(manager)

        return {
            "chunks": chunks,
            "embeddings": embeddings,
            "manager": manager,
            "persistence": persistence,
            "tmp_path": tmp_path,
        }

    def test_embedding_dimension_is_1024(self, pipeline):
        """Every embedding must have exactly 1024 dimensions."""
        for emb in pipeline["embeddings"]:
            assert len(emb.vector) == DIM, f"Wrong dim for chunk {emb.chunk_id}: {len(emb.vector)}"

    def test_index_contains_all_chunks(self, pipeline):
        """Index must contain one entry per input chunk."""
        manager = pipeline["manager"]
        assert manager.total_chunks == len(BIOMEDICAL_CHUNKS)

    def test_metadata_mapping_complete(self, pipeline):
        """All chunk_ids must be present in chunk_id_to_chunk."""
        manager = pipeline["manager"]
        chunks = pipeline["chunks"]
        for chunk in chunks:
            assert chunk.chunk_id in manager._chunk_id_to_chunk

    def test_metadata_survives_reload(self, pipeline):
        """After save+load, all chunk_ids must still be in chunk_id_to_chunk."""
        persistence = pipeline["persistence"]
        chunks = pipeline["chunks"]

        manager2 = FAISSManager(dim=DIM)
        persistence.load(manager2)

        for chunk in chunks:
            assert chunk.chunk_id in manager2._chunk_id_to_chunk, (
                f"chunk_id {chunk.chunk_id} lost after reload"
            )

    def test_search_returns_expected_chunk(self, pipeline):
        """Searching with a known embedding must return the correct chunk."""
        manager = pipeline["manager"]
        embeddings = pipeline["embeddings"]
        chunks = pipeline["chunks"]

        # Use the first embedding as the query.
        query_emb = embeddings[0]
        query_vec = np.array(query_emb.vector, dtype=np.float32)
        query_vec /= np.linalg.norm(query_vec)
        results = manager.search(query_vec.tolist(), top_k=3)

        assert len(results) > 0
        top_chunk_id = results[0].chunk.chunk_id
        assert top_chunk_id == chunks[0].chunk_id, (
            f"Expected {chunks[0].chunk_id} as top-1, got {top_chunk_id}"
        )

    def test_search_after_reload_returns_expected_chunk(self, pipeline):
        """After save+load, search must still return the expected chunk."""
        persistence = pipeline["persistence"]
        embeddings = pipeline["embeddings"]
        chunks = pipeline["chunks"]

        manager2 = FAISSManager(dim=DIM)
        persistence.load(manager2)

        query_vec = np.array(embeddings[0].vector, dtype=np.float32)
        query_vec /= np.linalg.norm(query_vec)
        results = manager2.search(query_vec.tolist(), top_k=3)

        assert len(results) > 0
        assert results[0].chunk.chunk_id == chunks[0].chunk_id

    def test_search_scores_are_valid(self, pipeline):
        """All search scores must be numeric and in a valid cosine range."""
        manager = pipeline["manager"]
        query = pipeline["embeddings"][0].vector
        query_vec = np.array(query, dtype=np.float32)
        query_vec /= np.linalg.norm(query_vec)
        results = manager.search(query_vec.tolist(), top_k=6)
        for r in results:
            assert isinstance(r.score, float)
            assert -1.0 - 1e-5 <= r.score <= 1.0 + 1e-5

    def test_index_stats_json_content(self, pipeline):
        """index_stats.json must have all required fields with correct values."""
        persistence = pipeline["persistence"]
        stats = persistence.read_stats()

        assert stats["total_chunks"] == len(BIOMEDICAL_CHUNKS)
        assert stats["total_documents"] > 0
        assert stats["embedding_dimension"] == DIM
        assert stats["index_type"] == "flat"
        assert stats["model_name"] == "BAAI/bge-large-en-v1.5"

    def test_multi_document_index(self, pipeline):
        """Index must span multiple documents."""
        manager = pipeline["manager"]
        stats = manager.get_index_stats()
        # BIOMEDICAL_CHUNKS has 3 documents.
        assert stats["total_documents"] == 3

    def test_delete_document_and_search(self, pipeline):
        """After deleting a document, its chunks must not appear in search."""
        manager = pipeline["manager"]
        embeddings = pipeline["embeddings"]

        # Delete doc-001 (chunks at index 0, 1, 2 in BIOMEDICAL_CHUNKS).
        n_removed = manager.delete_document("doc-001")
        assert n_removed == 3

        # Search should not return doc-001 chunks.
        query_vec = np.array(embeddings[0].vector, dtype=np.float32)
        query_vec /= np.linalg.norm(query_vec)
        results = manager.search(query_vec.tolist(), top_k=6)
        doc_ids = {r.document_id for r in results}
        assert "doc-001" not in doc_ids
