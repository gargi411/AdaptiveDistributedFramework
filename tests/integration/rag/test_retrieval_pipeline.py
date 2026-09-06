"""Integration test: Retrieval Engine pipeline (Phase 4.4 Milestone 1).

This test exercises the complete retrieval path from synthetic clinical
chunks through to RetrievalResult objects:

    Synthetic Chunk objects (clinical topics)
        -> Mock BGEEmbedder (deterministic 1024-dim vectors)
        -> list[Embedding]
        -> FAISSManager (add + register)
        -> RetrievalEngine.retrieve(natural-language query)
        -> list[RetrievalResult]
        -> verify rank, score, metadata

Synthetic clinical chunk topics used:
    - Medication: metformin for type-2 diabetes
    - Lab results: elevated troponin in myocardial infarction
    - Diagnosis: Alzheimer disease amyloid plaques
    - Unrelated: atmospheric pressure in meteorology

The test uses DETERMINISTIC query vectors aligned to specific chunk vectors
to ensure reproducible top-1 results that do not depend on actual model
semantic similarity.  This is consistent with the Phase 4.3 integration
test approach.

The real BAAI/bge-large-en-v1.5 model is NOT used.  The BGEEmbedder is
mocked to return deterministic 1024-dimensional unit vectors.

Note: Because we use random but deterministic mock embeddings (not real
semantic embeddings), the 'semantic relevance' verified here is based on
which chunk has the highest cosine similarity to the query vector —
not on real clinical meaning.  Real semantic accuracy is a model-level
property, not a retrieval-engine property.
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from adaptive_framework.models.chunk import Chunk, _make_chunk_id
from adaptive_framework.rag.embedding.bge_embedder import BGEEmbedder, _BGE_LARGE_DIM
from adaptive_framework.rag.models.embedding import Embedding
from adaptive_framework.rag.vector_store.faiss_manager import FAISSManager
from adaptive_framework.rag.vector_store.faiss_persistence import FAISSPersistence
from adaptive_framework.rag.retrieval.retrieval_engine import RetrievalEngine
from adaptive_framework.rag.retrieval.retrieval_result import RetrievalResult


DIM = _BGE_LARGE_DIM  # 1024

# ---------------------------------------------------------------------------
# Synthetic clinical data
# ---------------------------------------------------------------------------

CLINICAL_CHUNKS_DATA = [
    (
        "doc-medication",
        0,
        "Metformin is the first-line pharmacological treatment for type-2 diabetes mellitus.",
        "Pharmacology",
        "clinical_guideline",
    ),
    (
        "doc-lab",
        0,
        "Elevated troponin levels are a hallmark biomarker for acute myocardial infarction.",
        "Laboratory Results",
        "clinical_note",
    ),
    (
        "doc-diagnosis",
        0,
        "Alzheimer disease is characterized by amyloid beta plaques and neurofibrillary tangles.",
        "Diagnosis",
        "research_paper",
    ),
    (
        "doc-unrelated",
        0,
        "Atmospheric pressure decreases with altitude above sea level.",
        None,
        "other",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(
    doc_id: str,
    idx: int,
    text: str,
    section: str | None,
    doc_type: str,
) -> Chunk:
    return Chunk(
        chunk_id=_make_chunk_id(doc_id, idx),
        document_id=doc_id,
        source_file=f"/data/{doc_id}.pdf",
        chunk_index=idx,
        page_numbers=(1,),
        text=text,
        section_heading=section,
        document_type=doc_type,
        character_count=len(text),
        word_count=len(text.split()),
        start_char_in_page=0,
        end_char_in_page=len(text),
    )


def _unit_vector(seed: int) -> np.ndarray:
    """Return a deterministic L2-unit vector of shape (DIM,) float32."""
    rng = np.random.default_rng(seed)
    v = rng.random(DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


def _make_mock_model_with_fixed_vectors(
    chunk_texts: list[str],
    chunk_vectors: list[np.ndarray],
) -> MagicMock:
    """Return a mock SentenceTransformer whose encode() returns pre-assigned vectors.

    chunk_texts[i] -> chunk_vectors[i]
    Any other text (queries) -> a random vector seeded by total char count.
    """
    text_to_vec: dict[str, np.ndarray] = {}
    for text, vec in zip(chunk_texts, chunk_vectors):
        text_to_vec[text] = vec

    mock = MagicMock()

    def fake_encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ):
        result = []
        for t in texts:
            if t in text_to_vec:
                result.append(text_to_vec[t].copy())
            else:
                # Query text (possibly prefixed) or unknown -> seed by char count
                rng = np.random.default_rng(len(t))
                v = rng.random(DIM).astype(np.float32)
                if normalize_embeddings:
                    v /= np.linalg.norm(v)
                result.append(v)
        arr = np.stack(result, axis=0)
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr /= np.where(norms == 0, 1, norms)
        return arr

    mock.encode.side_effect = fake_encode
    return mock


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

class TestRetrievalPipeline:
    """Full pipeline: clinical Chunks -> mocked BGEEmbedder -> FAISSManager -> RetrievalEngine."""

    @pytest.fixture()
    def pipeline(self):
        """Build the full retrieval pipeline with deterministic mock embeddings."""
        # Build Chunk objects.
        chunks = [
            _make_chunk(doc_id, idx, text, section, doc_type)
            for doc_id, idx, text, section, doc_type in CLINICAL_CHUNKS_DATA
        ]

        # Assign one unique unit vector per chunk (seed = chunk index).
        chunk_vectors = [_unit_vector(seed=i) for i in range(len(chunks))]
        chunk_texts = [c.text for c in chunks]

        # Embed chunks via mocked BGEEmbedder.
        mock_model = _make_mock_model_with_fixed_vectors(chunk_texts, chunk_vectors)
        with patch(
            "adaptive_framework.rag.embedding.bge_embedder.SentenceTransformer",
            return_value=mock_model,
        ):
            embedder = BGEEmbedder(model_name="BAAI/bge-large-en-v1.5", device="cpu")
            embedder.initialize()
            embeddings = embedder.embed_chunks(chunks)

        # Build FAISSManager and index all chunks.
        manager = FAISSManager(dim=DIM, model_name="BAAI/bge-large-en-v1.5")
        manager.add_embeddings_with_chunks(embeddings, chunks)

        # Build RetrievalEngine with the same mocked embedder.
        engine = RetrievalEngine(
            embedding_provider=embedder,
            faiss_manager=manager,
            default_top_k=4,
            max_top_k=20,
        )

        return {
            "chunks": chunks,
            "embeddings": embeddings,
            "manager": manager,
            "engine": engine,
            "chunk_vectors": chunk_vectors,
        }

    # ---- correctness -------------------------------------------------------

    def test_retrieve_returns_retrieval_results(self, pipeline):
        """retrieve() returns a list of RetrievalResult objects."""
        engine = pipeline["engine"]
        results = engine.retrieve("cardiac biomarker", top_k=3)
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, RetrievalResult)

    def test_all_chunks_indexed(self, pipeline):
        """FAISSManager contains all 4 synthetic chunks."""
        manager = pipeline["manager"]
        assert manager.total_chunks == len(CLINICAL_CHUNKS_DATA)

    def test_top_k_respected(self, pipeline):
        """retrieve() returns at most top_k results."""
        engine = pipeline["engine"]
        results = engine.retrieve("diabetes treatment", top_k=2)
        assert len(results) <= 2

    def test_result_count_at_most_all_chunks(self, pipeline):
        """Result count cannot exceed the total number of indexed chunks."""
        engine = pipeline["engine"]
        results = engine.retrieve("any query", top_k=20)
        assert len(results) <= len(CLINICAL_CHUNKS_DATA)

    def test_ranks_are_sequential_from_one(self, pipeline):
        """Ranks start at 1 and increase by 1."""
        engine = pipeline["engine"]
        results = engine.retrieve("clinical findings", top_k=4)
        for i, r in enumerate(results):
            assert r.rank == i + 1

    def test_scores_descending(self, pipeline):
        """Results are ordered by descending score."""
        engine = pipeline["engine"]
        results = engine.retrieve("troponin elevation", top_k=4)
        scores = [r.score for r in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1] - 1e-6

    def test_scores_are_valid_floats(self, pipeline):
        """All scores are floats in [-1, 1]."""
        engine = pipeline["engine"]
        results = engine.retrieve("metformin diabetes", top_k=4)
        for r in results:
            assert isinstance(r.score, float)
            assert -1.0 - 1e-5 <= r.score <= 1.0 + 1e-5

    def test_chunk_metadata_preserved(self, pipeline):
        """Chunk metadata (doc_id, text, section, etc.) survives the pipeline."""
        engine = pipeline["engine"]
        chunks = pipeline["chunks"]
        results = engine.retrieve("clinical", top_k=4)
        result_chunk_ids = {r.chunk_id for r in results}
        expected_chunk_ids = {c.chunk_id for c in chunks}
        # All returned chunk_ids must come from the indexed set.
        assert result_chunk_ids.issubset(expected_chunk_ids)

    def test_document_id_in_result_matches_chunk(self, pipeline):
        """result.document_id matches the indexed Chunk's document_id."""
        engine = pipeline["engine"]
        chunks = pipeline["chunks"]
        chunk_id_to_doc = {c.chunk_id: c.document_id for c in chunks}
        results = engine.retrieve("query", top_k=4)
        for r in results:
            assert r.document_id == chunk_id_to_doc[r.chunk_id]

    def test_source_file_in_result(self, pipeline):
        """result.source_file is a non-empty string."""
        engine = pipeline["engine"]
        results = engine.retrieve("query", top_k=4)
        for r in results:
            assert isinstance(r.source_file, str)
            assert len(r.source_file) > 0

    def test_page_numbers_are_tuples(self, pipeline):
        """result.page_numbers is a tuple of ints."""
        engine = pipeline["engine"]
        results = engine.retrieve("query", top_k=4)
        for r in results:
            assert isinstance(r.page_numbers, tuple)
            assert all(isinstance(p, int) for p in r.page_numbers)

    def test_to_dict_is_serialisable(self, pipeline):
        """RetrievalResult.to_dict() produces a JSON-serialisable dict."""
        import json
        engine = pipeline["engine"]
        results = engine.retrieve("query", top_k=2)
        for r in results:
            d = r.to_dict()
            json_str = json.dumps(d)
            assert isinstance(json_str, str)

    # ---- query-vector-aligned top-1 result ---------------------------------

    def test_exact_vector_query_is_top1(self, pipeline):
        """Querying with a vector identical to a stored chunk vector gives rank=1 for that chunk."""
        from unittest.mock import patch

        # Seed 0 corresponds to chunk 0 (doc-medication).
        chunk_0 = pipeline["chunks"][0]
        chunk_vec = pipeline["chunk_vectors"][0]
        engine = pipeline["engine"]

        # Patch embed_query on the real BGEEmbedder instance so it returns
        # the exact chunk-0 vector, guaranteeing rank=1 for chunk-0.
        with patch.object(
            engine._provider, "embed_query", return_value=chunk_vec.tolist()
        ):
            results = engine.retrieve("metformin diabetes treatment", top_k=4)

        assert len(results) >= 1
        assert results[0].chunk_id == chunk_0.chunk_id
        assert results[0].rank == 1

    # ---- metadata filter ---------------------------------------------------

    def test_metadata_filter_restricts_to_document(self, pipeline):
        """metadata_filters={'document_id': ...} restricts results to that document."""
        engine = pipeline["engine"]
        results = engine.retrieve(
            "query",
            top_k=4,
            metadata_filters={"document_id": "doc-lab"},
        )
        for r in results:
            assert r.document_id == "doc-lab"

    def test_unknown_document_id_filter_returns_empty(self, pipeline):
        """Filtering by a non-existent document_id returns []."""
        engine = pipeline["engine"]
        results = engine.retrieve(
            "query",
            top_k=4,
            metadata_filters={"document_id": "doc-nonexistent"},
        )
        assert results == []

    # ---- metrics -----------------------------------------------------------

    def test_metrics_updated_after_retrieval(self, pipeline):
        """Retrieval metrics are updated after retrieve() is called."""
        engine = pipeline["engine"]
        engine.retrieve("query 1", top_k=2)
        engine.retrieve("query 2", top_k=2)
        assert engine.metrics.total_queries == 2
        assert engine.metrics.avg_retrieval_time_ms >= 0.0

    def test_get_metrics_dict(self, pipeline):
        """get_metrics() returns a dictionary."""
        engine = pipeline["engine"]
        engine.retrieve("query")
        m = engine.get_metrics()
        assert isinstance(m, dict)
        assert m["total_queries"] == 1

    # ---- index stats -------------------------------------------------------

    def test_get_index_stats(self, pipeline):
        """get_index_stats() returns expected stats."""
        engine = pipeline["engine"]
        stats = engine.get_index_stats()
        assert stats["total_chunks"] == len(CLINICAL_CHUNKS_DATA)
        assert stats["embedding_dimension"] == DIM
        assert stats["index_type"] == "flat"

    # ---- persist and reload ------------------------------------------------

    def test_retrieval_after_persist_and_reload(self, pipeline, tmp_path):
        """After persisting and reloading the FAISS index, retrieval still works."""
        from unittest.mock import patch

        manager = pipeline["manager"]
        engine = pipeline["engine"]
        chunk_0 = pipeline["chunks"][0]
        chunk_vec = pipeline["chunk_vectors"][0]

        # Persist the index.
        persistence = FAISSPersistence(index_dir=tmp_path / "index")
        persistence.save(manager)

        # Load into a new manager.
        manager2 = FAISSManager(dim=DIM)
        persistence.load(manager2)

        # Build a new engine reusing the same provider (real BGEEmbedder).
        engine2 = RetrievalEngine(
            embedding_provider=engine._provider,
            faiss_manager=manager2,
        )

        # Patch embed_query to return the exact chunk-0 vector.
        with patch.object(
            engine2._provider, "embed_query", return_value=chunk_vec.tolist()
        ):
            results = engine2.retrieve("metformin type 2 diabetes", top_k=2)

        assert len(results) >= 1
        assert results[0].chunk_id == chunk_0.chunk_id

    # ---- empty index -------------------------------------------------------

    def test_retrieve_on_empty_index_returns_empty(self):
        """An engine wrapping an empty index returns [] for any query."""
        from unittest.mock import MagicMock
        provider = MagicMock()
        provider.get_model_name.return_value = "BAAI/bge-large-en-v1.5"
        provider.embed_query.return_value = _unit_vector(seed=0).tolist()
        manager = FAISSManager(dim=DIM)
        engine = RetrievalEngine(provider, manager)
        results = engine.retrieve("cardiac disease", top_k=3)
        assert results == []
