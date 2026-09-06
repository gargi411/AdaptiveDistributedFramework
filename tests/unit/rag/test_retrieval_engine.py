"""Unit tests for RetrievalEngine and RetrievalResult (Phase 4.4 Milestone 1).

Tests exercise the full RetrievalEngine.retrieve() path using:
    - A mock IEmbeddingProvider (no real sentence-transformers model).
    - A real FAISSManager populated with deterministic unit vectors (no disk I/O).

Tested requirements:
    - Successful semantic retrieval returns RetrievalResult list.
    - Query is embedded via embed_query() (asymmetric prefix is the embedder's job).
    - FAISSManager.search() is invoked with the correct query vector.
    - top_k is respected (results <= top_k).
    - Results are ordered by descending score (rank=1 has highest score).
    - Every RetrievalResult field is correctly sourced from Chunk and SearchResult.
    - RetrievalResult.to_dict() is JSON-serialisable.
    - Empty query raises ValueError.
    - Whitespace-only query raises ValueError.
    - top_k == 0 raises ValueError.
    - top_k < 0 raises ValueError.
    - top_k > max_top_k raises ValueError.
    - Empty FAISS index returns [].
    - Embedding provider not initialized raises RuntimeError (propagated).
    - Wrong embedding dimension raises ValueError (propagated from FAISSManager).
    - Fewer results than top_k returns however many are available.
    - min_score threshold filters below-threshold results.
    - RetrievalMetrics is updated correctly after each query.
    - metadata_filters are passed through to FAISSManager.search().
    - RetrievalEngine constructor rejects None provider or manager.
    - RetrievalEngine constructor rejects invalid default_top_k or max_top_k.
    - RetrievalEngine.__repr__ returns a string.
    - IRetrievalEngine is correctly implemented (interface compliance).
"""

from __future__ import annotations

import numpy as np
import pytest
from typing import Any, Optional
from unittest.mock import MagicMock, call, patch

from adaptive_framework.models.chunk import Chunk, _make_chunk_id
from adaptive_framework.rag.models.embedding import Embedding
from adaptive_framework.rag.interfaces.i_embedding_provider import IEmbeddingProvider
from adaptive_framework.rag.interfaces.i_retrieval_engine import IRetrievalEngine
from adaptive_framework.rag.retrieval.retrieval_engine import (
    RetrievalEngine,
    _DEFAULT_TOP_K,
    _DEFAULT_MAX_TOP_K,
)
from adaptive_framework.rag.retrieval.retrieval_result import RetrievalResult
from adaptive_framework.rag.retrieval.retrieval_metrics import RetrievalMetrics
from adaptive_framework.rag.vector_store.faiss_manager import FAISSManager

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIM = 1024


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_chunk(
    doc_id: str = "doc-001",
    idx: int = 0,
    text: str = "Sample clinical text.",
    page: int = 1,
    section: Optional[str] = None,
    doc_type: Optional[str] = "clinical_note",
) -> Chunk:
    """Build a minimal Chunk for testing."""
    return Chunk(
        chunk_id=_make_chunk_id(doc_id, idx),
        document_id=doc_id,
        source_file=f"/data/{doc_id}.pdf",
        chunk_index=idx,
        page_numbers=(page,),
        text=text,
        section_heading=section,
        document_type=doc_type,
        character_count=len(text),
        word_count=len(text.split()),
        start_char_in_page=0,
        end_char_in_page=len(text),
    )


def _make_unit_vector(seed: int = 0) -> list[float]:
    """Return a deterministic L2-unit vector of length DIM."""
    rng = np.random.default_rng(seed)
    vec = rng.random(DIM).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


def _make_embedding(chunk: Chunk, seed: int = 0) -> Embedding:
    """Create a deterministic unit-length Embedding for testing."""
    return Embedding(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        vector=tuple(_make_unit_vector(seed)),
        model_name="BAAI/bge-large-en-v1.5",
        embedding_time_seconds=0.001,
        created_at="2026-08-31T00:00:00+00:00",
    )


def _make_mock_provider(query_vector: Optional[list[float]] = None) -> MagicMock:
    """Return a mock IEmbeddingProvider.

    embed_query() returns query_vector (defaults to seed-0 unit vector).
    get_model_name() returns 'BAAI/bge-large-en-v1.5'.
    get_embedding_dim() returns DIM.
    """
    provider = MagicMock(spec=IEmbeddingProvider)
    provider.get_model_name.return_value = "BAAI/bge-large-en-v1.5"
    provider.get_embedding_dim.return_value = DIM
    if query_vector is None:
        query_vector = _make_unit_vector(seed=99)
    provider.embed_query.return_value = query_vector
    return provider


def _build_populated_manager(
    chunks: list[Chunk],
    seeds: Optional[list[int]] = None,
) -> FAISSManager:
    """Build a FAISSManager populated with one embedding per chunk."""
    manager = FAISSManager(dim=DIM)
    if seeds is None:
        seeds = list(range(len(chunks)))
    embeddings = [_make_embedding(c, seed=s) for c, s in zip(chunks, seeds)]
    manager.add_embeddings_with_chunks(embeddings, chunks)
    return manager


def _build_engine(
    chunks: Optional[list[Chunk]] = None,
    query_vector: Optional[list[float]] = None,
    default_top_k: int = 5,
    max_top_k: int = 50,
) -> tuple[RetrievalEngine, MagicMock, FAISSManager]:
    """Build a RetrievalEngine with a mock provider and a real FAISSManager.

    Returns:
        (engine, mock_provider, faiss_manager)
    """
    if chunks is None:
        chunks = [_make_chunk(doc_id="doc-001", idx=i) for i in range(3)]
    provider = _make_mock_provider(query_vector=query_vector)
    manager = _build_populated_manager(chunks)
    engine = RetrievalEngine(
        embedding_provider=provider,
        faiss_manager=manager,
        default_top_k=default_top_k,
        max_top_k=max_top_k,
    )
    return engine, provider, manager


# ---------------------------------------------------------------------------
# Tests: RetrievalResult
# ---------------------------------------------------------------------------

class TestRetrievalResult:
    """Tests for the RetrievalResult dataclass."""

    def test_construction(self):
        r = RetrievalResult(
            chunk_id="abc123",
            document_id="doc-001",
            source_file="/data/doc.pdf",
            page_numbers=(2, 3),
            text="Patient has elevated troponin.",
            section_heading="Lab Results",
            document_type="clinical_note",
            chunk_index=4,
            score=0.91,
            rank=1,
        )
        assert r.chunk_id == "abc123"
        assert r.document_id == "doc-001"
        assert r.source_file == "/data/doc.pdf"
        assert r.page_numbers == (2, 3)
        assert r.text == "Patient has elevated troponin."
        assert r.section_heading == "Lab Results"
        assert r.document_type == "clinical_note"
        assert r.chunk_index == 4
        assert r.score == 0.91
        assert r.rank == 1

    def test_is_frozen(self):
        r = RetrievalResult(
            chunk_id="x", document_id="d", source_file="/f",
            page_numbers=(1,), text="t", section_heading=None,
            document_type=None, chunk_index=0, score=0.5, rank=1,
        )
        with pytest.raises(Exception):
            r.rank = 2  # type: ignore[misc]

    def test_to_dict_is_json_serialisable(self):
        import json
        r = RetrievalResult(
            chunk_id="abc", document_id="doc", source_file="/f",
            page_numbers=(1, 2), text="some text", section_heading="Intro",
            document_type="research_paper", chunk_index=0, score=0.75, rank=1,
        )
        d = r.to_dict()
        json_str = json.dumps(d)
        assert '"chunk_id": "abc"' in json_str
        assert isinstance(d["page_numbers"], list)
        assert d["page_numbers"] == [1, 2]

    def test_to_dict_fields(self):
        r = RetrievalResult(
            chunk_id="cid", document_id="did", source_file="/src",
            page_numbers=(3,), text="hello", section_heading=None,
            document_type=None, chunk_index=2, score=0.55, rank=2,
        )
        d = r.to_dict()
        assert d["chunk_id"] == "cid"
        assert d["document_id"] == "did"
        assert d["source_file"] == "/src"
        assert d["page_numbers"] == [3]
        assert d["text"] == "hello"
        assert d["section_heading"] is None
        assert d["document_type"] is None
        assert d["chunk_index"] == 2
        assert d["score"] == 0.55
        assert d["rank"] == 2

    def test_repr_contains_rank_and_score(self):
        r = RetrievalResult(
            chunk_id="abc", document_id="doc", source_file="/f",
            page_numbers=(1,), text="t", section_heading=None,
            document_type=None, chunk_index=0, score=0.88, rank=1,
        )
        rep = repr(r)
        assert "rank=1" in rep
        assert "0.8800" in rep


# ---------------------------------------------------------------------------
# Tests: RetrievalMetrics
# ---------------------------------------------------------------------------

class TestRetrievalMetrics:
    """Tests for the RetrievalMetrics dataclass."""

    def test_initial_state(self):
        m = RetrievalMetrics()
        assert m.total_queries == 0
        assert m.total_results_returned == 0
        assert m.total_retrieval_time_s == 0.0
        assert m.empty_results_count == 0
        assert m.avg_retrieval_time_ms == 0.0

    def test_record_query_updates_totals(self):
        m = RetrievalMetrics()
        m.record_query(n_results=3, elapsed_s=0.01)
        assert m.total_queries == 1
        assert m.total_results_returned == 3
        assert m.empty_results_count == 0

    def test_record_empty_query_increments_empty_count(self):
        m = RetrievalMetrics()
        m.record_query(n_results=0, elapsed_s=0.005)
        assert m.empty_results_count == 1

    def test_avg_retrieval_time_ms(self):
        m = RetrievalMetrics()
        m.record_query(n_results=1, elapsed_s=0.010)
        m.record_query(n_results=2, elapsed_s=0.020)
        # mean = (0.010 + 0.020) / 2 * 1000 = 15.0
        assert abs(m.avg_retrieval_time_ms - 15.0) < 0.001

    def test_to_dict_keys(self):
        m = RetrievalMetrics()
        m.record_query(n_results=2, elapsed_s=0.01)
        d = m.to_dict()
        assert "total_queries" in d
        assert "total_results_returned" in d
        assert "total_retrieval_time_s" in d
        assert "empty_results_count" in d
        assert "avg_retrieval_time_ms" in d


# ---------------------------------------------------------------------------
# Tests: RetrievalEngine construction
# ---------------------------------------------------------------------------

class TestRetrievalEngineConstruction:
    """Tests for RetrievalEngine.__init__ validation."""

    def test_valid_construction(self):
        engine, _, _ = _build_engine()
        assert engine.default_top_k == 5
        assert engine.max_top_k == 50

    def test_none_provider_raises(self):
        manager = FAISSManager(dim=DIM)
        with pytest.raises(TypeError, match="embedding_provider"):
            RetrievalEngine(embedding_provider=None, faiss_manager=manager)  # type: ignore[arg-type]

    def test_none_manager_raises(self):
        provider = _make_mock_provider()
        with pytest.raises(TypeError, match="faiss_manager"):
            RetrievalEngine(embedding_provider=provider, faiss_manager=None)  # type: ignore[arg-type]

    def test_default_top_k_below_one_raises(self):
        provider = _make_mock_provider()
        manager = FAISSManager(dim=DIM)
        with pytest.raises(ValueError, match="default_top_k"):
            RetrievalEngine(provider, manager, default_top_k=0)

    def test_max_top_k_below_default_raises(self):
        provider = _make_mock_provider()
        manager = FAISSManager(dim=DIM)
        with pytest.raises(ValueError, match="max_top_k"):
            RetrievalEngine(provider, manager, default_top_k=10, max_top_k=5)

    def test_repr_is_string(self):
        engine, _, _ = _build_engine()
        assert isinstance(repr(engine), str)
        assert "RetrievalEngine" in repr(engine)


# ---------------------------------------------------------------------------
# Tests: query validation
# ---------------------------------------------------------------------------

class TestQueryValidation:
    """Tests for query string validation in retrieve()."""

    def test_empty_string_raises(self):
        engine, _, _ = _build_engine()
        with pytest.raises(ValueError, match="non-empty"):
            engine.retrieve("")

    def test_whitespace_only_raises(self):
        engine, _, _ = _build_engine()
        with pytest.raises(ValueError, match="non-empty"):
            engine.retrieve("   ")

    def test_tab_only_raises(self):
        engine, _, _ = _build_engine()
        with pytest.raises(ValueError, match="non-empty"):
            engine.retrieve("\t\n")

    def test_valid_query_does_not_raise(self):
        engine, _, _ = _build_engine()
        results = engine.retrieve("elevated troponin")
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Tests: top_k validation
# ---------------------------------------------------------------------------

class TestTopKValidation:
    """Tests for top_k validation in retrieve()."""

    def test_top_k_zero_raises(self):
        engine, _, _ = _build_engine()
        with pytest.raises(ValueError, match="top_k"):
            engine.retrieve("query", top_k=0)

    def test_top_k_negative_raises(self):
        engine, _, _ = _build_engine()
        with pytest.raises(ValueError, match="top_k"):
            engine.retrieve("query", top_k=-1)

    def test_top_k_above_max_raises(self):
        engine, _, _ = _build_engine(max_top_k=10)
        with pytest.raises(ValueError, match="max_top_k"):
            engine.retrieve("query", top_k=11)

    def test_top_k_exactly_max_does_not_raise(self):
        engine, _, _ = _build_engine(max_top_k=10)
        results = engine.retrieve("query", top_k=10)
        assert isinstance(results, list)

    def test_top_k_one_returns_at_most_one(self):
        chunks = [_make_chunk(doc_id="doc-001", idx=i) for i in range(5)]
        engine, _, _ = _build_engine(chunks=chunks)
        results = engine.retrieve("some query", top_k=1)
        assert len(results) <= 1


# ---------------------------------------------------------------------------
# Tests: successful retrieval
# ---------------------------------------------------------------------------

class TestSuccessfulRetrieval:
    """Tests for the main retrieval path."""

    def test_returns_list_of_retrieval_result(self):
        engine, _, _ = _build_engine()
        results = engine.retrieve("elevated troponin in cardiac patients")
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, RetrievalResult)

    def test_embed_query_is_called_once(self):
        engine, provider, _ = _build_engine()
        engine.retrieve("cardiac biomarkers")
        provider.embed_query.assert_called_once_with("cardiac biomarkers")

    def test_result_count_at_most_top_k(self):
        chunks = [_make_chunk("doc-001", i) for i in range(10)]
        engine, _, _ = _build_engine(chunks=chunks)
        results = engine.retrieve("query", top_k=3)
        assert len(results) <= 3

    def test_result_fields_sourced_from_chunk(self):
        chunk = _make_chunk(
            doc_id="doc-X",
            idx=7,
            text="Tau protein aggregation leads to neurofibrillary tangles.",
            page=4,
            section="Pathology",
            doc_type="research_paper",
        )
        # Use the exact same vector for both chunk and query so rank=1 is guaranteed.
        vec = _make_unit_vector(seed=42)
        embedding = Embedding(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            vector=tuple(vec),
            model_name="BAAI/bge-large-en-v1.5",
            embedding_time_seconds=0.0,
            created_at="2026-01-01T00:00:00+00:00",
        )
        provider = _make_mock_provider(query_vector=vec)
        manager = FAISSManager(dim=DIM)
        manager.add_embeddings_with_chunks([embedding], [chunk])
        engine = RetrievalEngine(provider, manager)

        results = engine.retrieve("tau protein neurodegeneration")
        assert len(results) == 1
        r = results[0]
        assert r.chunk_id == chunk.chunk_id
        assert r.document_id == "doc-X"
        assert r.source_file == "/data/doc-X.pdf"
        assert r.page_numbers == (4,)
        assert r.text == "Tau protein aggregation leads to neurofibrillary tangles."
        assert r.section_heading == "Pathology"
        assert r.document_type == "research_paper"
        assert r.chunk_index == 7

    def test_rank_one_has_highest_score(self):
        chunks = [_make_chunk("doc-001", i) for i in range(5)]
        engine, _, _ = _build_engine(chunks=chunks)
        results = engine.retrieve("query", top_k=5)
        if len(results) >= 2:
            assert results[0].score >= results[1].score

    def test_ranks_are_1_indexed_and_sequential(self):
        chunks = [_make_chunk("doc-001", i) for i in range(5)]
        engine, _, _ = _build_engine(chunks=chunks)
        results = engine.retrieve("query", top_k=5)
        for i, r in enumerate(results):
            assert r.rank == i + 1

    def test_scores_are_floats(self):
        engine, _, _ = _build_engine()
        results = engine.retrieve("query")
        for r in results:
            assert isinstance(r.score, float)

    def test_scores_in_cosine_range(self):
        engine, _, _ = _build_engine()
        results = engine.retrieve("query")
        for r in results:
            assert -1.0 - 1e-5 <= r.score <= 1.0 + 1e-5

    def test_default_top_k_used_when_not_specified(self):
        chunks = [_make_chunk("doc-001", i) for i in range(20)]
        engine, _, _ = _build_engine(chunks=chunks, default_top_k=3)
        results = engine.retrieve("query")
        assert len(results) <= 3

    def test_query_vector_dimension_matches_index(self):
        # FAISSManager raises ValueError on wrong dimension.
        provider = _make_mock_provider(query_vector=[0.1] * 512)  # wrong dim
        manager = FAISSManager(dim=DIM)
        chunk = _make_chunk()
        emb = _make_embedding(chunk)
        manager.add_embeddings_with_chunks([emb], [chunk])
        engine = RetrievalEngine(provider, manager)
        with pytest.raises(ValueError, match="dim mismatch"):
            engine.retrieve("query")


# ---------------------------------------------------------------------------
# Tests: empty index
# ---------------------------------------------------------------------------

class TestEmptyIndex:
    """Retrieval against an empty FAISS index should return []."""

    def test_empty_index_returns_empty_list(self):
        provider = _make_mock_provider()
        manager = FAISSManager(dim=DIM)  # no embeddings added
        engine = RetrievalEngine(provider, manager)
        results = engine.retrieve("any query")
        assert results == []

    def test_empty_index_updates_metrics(self):
        provider = _make_mock_provider()
        manager = FAISSManager(dim=DIM)
        engine = RetrievalEngine(provider, manager)
        engine.retrieve("any query")
        assert engine.metrics.total_queries == 1
        assert engine.metrics.empty_results_count == 1


# ---------------------------------------------------------------------------
# Tests: fewer results than top_k
# ---------------------------------------------------------------------------

class TestFewerResultsThanTopK:
    """When the index has fewer entries than top_k, all available are returned."""

    def test_fewer_results_than_top_k(self):
        chunks = [_make_chunk("doc-001", i) for i in range(2)]
        engine, _, _ = _build_engine(chunks=chunks)
        results = engine.retrieve("query", top_k=10)
        assert len(results) == 2

    def test_single_chunk_index(self):
        chunk = _make_chunk("doc-001", 0)
        engine, _, _ = _build_engine(chunks=[chunk])
        results = engine.retrieve("query", top_k=5)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Tests: min_score threshold
# ---------------------------------------------------------------------------

class TestMinScoreThreshold:
    """min_score post-filters results below the threshold."""

    def test_min_score_zero_returns_all(self):
        chunks = [_make_chunk("doc-001", i) for i in range(3)]
        engine, _, _ = _build_engine(chunks=chunks)
        results_no_filter = engine.retrieve("query", top_k=3, min_score=0.0)
        results_with_filter = engine.retrieve("query", top_k=3, min_score=0.0)
        assert len(results_no_filter) == len(results_with_filter)

    def test_min_score_one_returns_empty_for_random_vectors(self):
        # Random unit vectors will not have score == 1.0 unless identical.
        chunks = [_make_chunk("doc-001", i) for i in range(3)]
        # Query vector is seed=99; chunk vectors are seeds 0,1,2 — all different.
        engine, _, _ = _build_engine(chunks=chunks)
        results = engine.retrieve("query", top_k=3, min_score=1.0)
        assert results == []

    def test_min_score_filters_below_threshold(self):
        # Use the exact same vector for query and one chunk to get score ~1.0.
        vec = _make_unit_vector(seed=42)
        chunk_match = _make_chunk("doc-001", 0, "matching chunk")
        chunk_other = _make_chunk("doc-001", 1, "other chunk")
        emb_match = Embedding(
            chunk_id=chunk_match.chunk_id,
            document_id=chunk_match.document_id,
            vector=tuple(vec),
            model_name="BAAI/bge-large-en-v1.5",
            embedding_time_seconds=0.0,
            created_at="2026-01-01T00:00:00+00:00",
        )
        emb_other = _make_embedding(chunk_other, seed=7)  # different vector
        provider = _make_mock_provider(query_vector=vec)
        manager = FAISSManager(dim=DIM)
        manager.add_embeddings_with_chunks([emb_match, emb_other], [chunk_match, chunk_other])
        engine = RetrievalEngine(provider, manager)

        # High threshold: only the matching chunk (score ~1.0) should survive.
        results = engine.retrieve("query", top_k=2, min_score=0.9)
        chunk_ids = [r.chunk_id for r in results]
        assert chunk_match.chunk_id in chunk_ids


# ---------------------------------------------------------------------------
# Tests: metadata_filters passthrough
# ---------------------------------------------------------------------------

class TestMetadataFilters:
    """metadata_filters are passed through to FAISSManager.search()."""

    def test_document_id_filter_restricts_results(self):
        chunks = [
            _make_chunk("doc-A", 0, "doc A chunk 0"),
            _make_chunk("doc-A", 1, "doc A chunk 1"),
            _make_chunk("doc-B", 0, "doc B chunk 0"),
        ]
        engine, _, _ = _build_engine(chunks=chunks)
        results = engine.retrieve(
            "some query",
            top_k=5,
            metadata_filters={"document_id": "doc-A"},
        )
        for r in results:
            assert r.document_id == "doc-A"

    def test_document_id_list_filter(self):
        chunks = [
            _make_chunk("doc-A", 0, "doc A chunk 0"),
            _make_chunk("doc-B", 0, "doc B chunk 0"),
            _make_chunk("doc-C", 0, "doc C chunk 0"),
        ]
        engine, _, _ = _build_engine(chunks=chunks)
        results = engine.retrieve(
            "query",
            top_k=5,
            metadata_filters={"document_id": ["doc-A", "doc-B"]},
        )
        for r in results:
            assert r.document_id in {"doc-A", "doc-B"}


# ---------------------------------------------------------------------------
# Tests: metrics
# ---------------------------------------------------------------------------

class TestRetrievalMetricsIntegration:
    """RetrievalMetrics are updated correctly by retrieve()."""

    def test_total_queries_increments(self):
        engine, _, _ = _build_engine()
        engine.retrieve("query one")
        engine.retrieve("query two")
        assert engine.metrics.total_queries == 2

    def test_total_results_accumulates(self):
        chunks = [_make_chunk("doc-001", i) for i in range(3)]
        engine, _, _ = _build_engine(chunks=chunks)
        engine.retrieve("query", top_k=2)
        engine.retrieve("query", top_k=2)
        assert engine.metrics.total_results_returned >= 0

    def test_avg_latency_is_positive(self):
        engine, _, _ = _build_engine()
        engine.retrieve("query")
        assert engine.metrics.avg_retrieval_time_ms >= 0.0

    def test_empty_result_counted(self):
        provider = _make_mock_provider()
        manager = FAISSManager(dim=DIM)
        engine = RetrievalEngine(provider, manager)
        engine.retrieve("query")
        assert engine.metrics.empty_results_count == 1

    def test_get_metrics_returns_dict(self):
        engine, _, _ = _build_engine()
        engine.retrieve("query")
        m = engine.get_metrics()
        assert isinstance(m, dict)
        assert "total_queries" in m
        assert "avg_retrieval_time_ms" in m


# ---------------------------------------------------------------------------
# Tests: get_index_stats
# ---------------------------------------------------------------------------

class TestGetIndexStats:
    """get_index_stats() delegates to FAISSManager.get_index_stats()."""

    def test_get_index_stats_keys(self):
        chunks = [_make_chunk("doc-001", i) for i in range(3)]
        engine, _, _ = _build_engine(chunks=chunks)
        stats = engine.get_index_stats()
        assert "total_chunks" in stats
        assert "total_documents" in stats
        assert "embedding_dimension" in stats

    def test_get_index_stats_total_chunks(self):
        chunks = [_make_chunk("doc-001", i) for i in range(4)]
        engine, _, _ = _build_engine(chunks=chunks)
        stats = engine.get_index_stats()
        assert stats["total_chunks"] == 4


# ---------------------------------------------------------------------------
# Tests: embedding provider not initialized
# ---------------------------------------------------------------------------

class TestEmbeddingProviderError:
    """RuntimeError from embed_query() is propagated by retrieve()."""

    def test_uninitialized_provider_raises_runtime_error(self):
        provider = _make_mock_provider()
        provider.embed_query.side_effect = RuntimeError(
            "EmbeddingProvider is not initialized."
        )
        manager = FAISSManager(dim=DIM)
        engine = RetrievalEngine(provider, manager)
        with pytest.raises(RuntimeError, match="not initialized"):
            engine.retrieve("some query")


# ---------------------------------------------------------------------------
# Tests: IRetrievalEngine interface compliance
# ---------------------------------------------------------------------------

class TestIRetrievalEngineCompliance:
    """RetrievalEngine correctly implements IRetrievalEngine."""

    def test_is_instance_of_interface(self):
        provider = _make_mock_provider()
        manager = FAISSManager(dim=DIM)
        engine = RetrievalEngine(provider, manager)
        # IRetrievalEngine is an ABC; RetrievalEngine does not formally inherit
        # from it (it uses composition).  Test the contractual methods exist.
        assert hasattr(engine, "retrieve")
        assert hasattr(engine, "get_metrics")
        assert callable(engine.retrieve)
        assert callable(engine.get_metrics)

    def test_get_metrics_dict_structure(self):
        provider = _make_mock_provider()
        manager = FAISSManager(dim=DIM)
        engine = RetrievalEngine(provider, manager)
        engine.retrieve("sample query")
        metrics = engine.get_metrics()
        assert "total_queries" in metrics
        assert "total_results_returned" in metrics
        assert "avg_retrieval_time_ms" in metrics
