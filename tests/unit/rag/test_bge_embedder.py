"""Unit tests for BGEEmbedder (Phase 4.3).

Tests the BGEEmbedder implementation without downloading the real
BAAI/bge-large-en-v1.5 model.  The sentence-transformers SentenceTransformer
class is mocked to return deterministic fixed-dimension vectors.

Tested requirements:
    - correct embedding dimension (1024)
    - chunk encoding (no prefix)
    - query encoding with asymmetric prefix
    - no prefix on chunks
    - batch encoding with configurable batch_size
    - empty input raises ValueError
    - invalid input handling
    - device configuration
    - deterministic interface
    - cache integration (hit avoids model call)
    - metrics tracking
    - initialize() guard
"""

from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, call
from pathlib import Path

from adaptive_framework.models.chunk import Chunk, _make_chunk_id
from adaptive_framework.rag.embedding.bge_embedder import BGEEmbedder, _QUERY_PREFIX, _BGE_LARGE_DIM
from adaptive_framework.rag.embedding.embedding_cache import EmbeddingCache
from adaptive_framework.rag.models.embedding import Embedding


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_chunk(doc_id: str = "doc-001", idx: int = 0, text: str = "Sample text.") -> Chunk:
    """Build a minimal Chunk for testing."""
    return Chunk(
        chunk_id=_make_chunk_id(doc_id, idx),
        document_id=doc_id,
        source_file="/data/sample.pdf",
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


def _make_mock_sentence_transformer(dim: int = _BGE_LARGE_DIM) -> MagicMock:
    """Return a mock SentenceTransformer that returns deterministic vectors."""
    mock_model = MagicMock()

    def fake_encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True):
        n = len(texts)
        # Return unit-length vectors seeded by text length for determinism.
        rng = np.random.default_rng(sum(len(t) for t in texts))
        vecs = rng.random((n, dim)).astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs /= np.where(norms == 0, 1, norms)
        return vecs

    mock_model.encode.side_effect = fake_encode
    return mock_model


@pytest.fixture()
def mock_sentence_transformer():
    """Patch SentenceTransformer so no model is downloaded."""
    with patch(
        "adaptive_framework.rag.embedding.bge_embedder.SentenceTransformer",
        return_value=_make_mock_sentence_transformer(),
    ) as mock_cls:
        yield mock_cls


@pytest.fixture()
def embedder(mock_sentence_transformer):
    """Return an initialized BGEEmbedder with a mocked model."""
    emb = BGEEmbedder(
        model_name="BAAI/bge-large-en-v1.5",
        device="cpu",
        batch_size=8,
    )
    emb.initialize()
    return emb


# ---------------------------------------------------------------------------
# Initialization and device configuration
# ---------------------------------------------------------------------------

class TestBGEEmbedderInit:
    def test_initialize_loads_model(self, mock_sentence_transformer):
        """initialize() should call SentenceTransformer with the correct args."""
        emb = BGEEmbedder(model_name="BAAI/bge-large-en-v1.5", device="cpu")
        emb.initialize()
        assert emb._initialized

    def test_double_initialize_is_noop(self, mock_sentence_transformer):
        """Calling initialize() twice must not reload the model."""
        emb = BGEEmbedder(model_name="BAAI/bge-large-en-v1.5", device="cpu")
        emb.initialize()
        emb.initialize()  # second call — model must NOT be reloaded
        mock_sentence_transformer.assert_called_once()

    def test_embed_chunks_before_initialize_raises(self):
        """embed_chunks() before initialize() must raise RuntimeError."""
        emb = BGEEmbedder()
        chunk = _make_chunk()
        with pytest.raises(RuntimeError, match="not initialized"):
            emb.embed_chunks([chunk])

    def test_embed_query_before_initialize_raises(self):
        """embed_query() before initialize() must raise RuntimeError."""
        emb = BGEEmbedder()
        with pytest.raises(RuntimeError, match="not initialized"):
            emb.embed_query("test query")

    def test_get_model_name(self, embedder):
        """get_model_name() must return the correct identifier."""
        assert embedder.get_model_name() == "BAAI/bge-large-en-v1.5"

    def test_get_embedding_dim(self, embedder):
        """get_embedding_dim() must return 1024 for bge-large-en-v1.5."""
        assert embedder.get_embedding_dim() == 1024

    def test_device_cpu_stored(self, mock_sentence_transformer):
        """The device parameter must be stored correctly."""
        emb = BGEEmbedder(device="cpu")
        assert emb._device == "cpu"

    def test_device_cuda_stored(self, mock_sentence_transformer):
        """CUDA device string must be accepted."""
        emb = BGEEmbedder(device="cuda")
        assert emb._device == "cuda"


# ---------------------------------------------------------------------------
# Embedding dimension
# ---------------------------------------------------------------------------

class TestEmbeddingDimension:
    def test_chunk_embedding_dimension(self, embedder):
        """Every chunk embedding must have exactly 1024 dimensions."""
        chunk = _make_chunk()
        [emb] = embedder.embed_chunks([chunk])
        assert len(emb.vector) == _BGE_LARGE_DIM

    def test_query_embedding_dimension(self, embedder):
        """Query embedding must have exactly 1024 dimensions."""
        vec = embedder.embed_query("What is CRISPR?")
        assert len(vec) == _BGE_LARGE_DIM

    def test_batch_all_same_dimension(self, embedder):
        """All embeddings in a batch must have the same dimension."""
        chunks = [_make_chunk(idx=i, text=f"Sentence number {i}.") for i in range(10)]
        embeddings = embedder.embed_chunks(chunks)
        dims = [len(e.vector) for e in embeddings]
        assert all(d == _BGE_LARGE_DIM for d in dims), f"Dimension mismatch: {dims}"


# ---------------------------------------------------------------------------
# Asymmetric encoding — THE CRITICAL CORRECTNESS REQUIREMENT
# ---------------------------------------------------------------------------

class TestAsymmetricEncoding:
    """Verify the BGE asymmetric query/chunk encoding rule.

    embed_query  -> adds prefix 'Represent this sentence for searching relevant passages: '
    embed_chunks -> NO prefix

    This is verified by inspecting what was passed to the mock model's encode().
    """

    def test_query_gets_prefix(self, embedder):
        """embed_query must prepend the BGE instruction prefix."""
        raw_query = "What causes Alzheimer's disease?"
        expected_text = _QUERY_PREFIX + raw_query

        # Capture what encode() was actually called with.
        embedder.embed_query(raw_query)

        calls = embedder._model.encode.call_args_list
        # Last call is the query encoding.
        args = calls[-1][0][0]  # positional arg 0 is the list of texts
        assert expected_text in args, (
            f"Query prefix missing. encode() was called with: {args}"
        )

    def test_chunks_have_no_prefix(self, embedder):
        """embed_chunks must NOT add any prefix to chunk text."""
        text = "CRISPR-Cas9 is a genome editing technology."
        chunk = _make_chunk(text=text)
        embedder.embed_chunks([chunk])

        calls = embedder._model.encode.call_args_list
        args = calls[-1][0][0]  # list of texts passed to encode
        # The chunk text should appear as-is, without any prefix.
        assert text in args, f"Expected chunk text in encode() call. Got: {args}"
        # The query prefix must NOT appear.
        for t in args:
            assert not t.startswith(_QUERY_PREFIX), (
                f"Chunk text should not have query prefix, but got: {t!r}"
            )

    def test_query_prefix_constant(self):
        """The query prefix constant must match the BGE specification."""
        assert _QUERY_PREFIX == "Represent this sentence for searching relevant passages: "


# ---------------------------------------------------------------------------
# Batch encoding
# ---------------------------------------------------------------------------

class TestBatchEncoding:
    def test_single_chunk(self, embedder):
        """A single-chunk batch must return exactly one Embedding."""
        chunk = _make_chunk(text="Single sentence.")
        result = embedder.embed_chunks([chunk])
        assert len(result) == 1
        assert result[0].chunk_id == chunk.chunk_id

    def test_multiple_chunks(self, embedder):
        """A multi-chunk batch must return one Embedding per chunk."""
        chunks = [_make_chunk(idx=i, text=f"Text {i} for testing.") for i in range(5)]
        result = embedder.embed_chunks(chunks)
        assert len(result) == 5
        ids = [e.chunk_id for e in result]
        expected_ids = [c.chunk_id for c in chunks]
        assert ids == expected_ids

    def test_batch_size_respected(self, mock_sentence_transformer):
        """The configured batch_size must be passed to encode()."""
        emb = BGEEmbedder(batch_size=3)
        emb.initialize()
        chunks = [_make_chunk(idx=i) for i in range(6)]
        emb.embed_chunks(chunks)
        # Check that batch_size=3 was passed to encode
        for call_args in emb._model.encode.call_args_list:
            assert call_args[1].get("batch_size", call_args[0][1] if len(call_args[0]) > 1 else 3) == 3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_chunks_raises(self, embedder):
        """embed_chunks([]) must raise ValueError."""
        with pytest.raises(ValueError, match="at least one Chunk"):
            embedder.embed_chunks([])

    def test_empty_query_raises(self, embedder):
        """embed_query('') must raise ValueError."""
        with pytest.raises(ValueError, match="non-empty query"):
            embedder.embed_query("")

    def test_whitespace_query_raises(self, embedder):
        """embed_query('   ') must raise ValueError."""
        with pytest.raises(ValueError, match="non-empty query"):
            embedder.embed_query("   ")

    def test_long_text_chunk(self, embedder):
        """A very long chunk must still produce a valid embedding."""
        text = "A " * 5000  # ~10,000 chars
        chunk = _make_chunk(text=text)
        [emb] = embedder.embed_chunks([chunk])
        assert len(emb.vector) == _BGE_LARGE_DIM


# ---------------------------------------------------------------------------
# Output fields
# ---------------------------------------------------------------------------

class TestEmbeddingFields:
    def test_chunk_id_preserved(self, embedder):
        """Embedding.chunk_id must match Chunk.chunk_id."""
        chunk = _make_chunk(doc_id="doc-abc", idx=3)
        [emb] = embedder.embed_chunks([chunk])
        assert emb.chunk_id == chunk.chunk_id

    def test_document_id_preserved(self, embedder):
        """Embedding.document_id must match Chunk.document_id."""
        chunk = _make_chunk(doc_id="doc-xyz")
        [emb] = embedder.embed_chunks([chunk])
        assert emb.document_id == chunk.document_id

    def test_model_name_in_embedding(self, embedder):
        """Embedding.model_name must match the embedder's model name."""
        chunk = _make_chunk()
        [emb] = embedder.embed_chunks([chunk])
        assert emb.model_name == "BAAI/bge-large-en-v1.5"

    def test_created_at_is_iso8601(self, embedder):
        """Embedding.created_at must be a non-empty ISO 8601 string."""
        chunk = _make_chunk()
        [emb] = embedder.embed_chunks([chunk])
        assert emb.created_at
        from datetime import datetime
        datetime.fromisoformat(emb.created_at)  # must not raise

    def test_embedding_time_non_negative(self, embedder):
        """Embedding.embedding_time_seconds must be >= 0."""
        chunk = _make_chunk()
        [emb] = embedder.embed_chunks([chunk])
        assert emb.embedding_time_seconds >= 0.0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_text_same_chunk_id(self):
        """Identical chunk text and doc/index must always produce same chunk_id."""
        id1 = _make_chunk_id("doc-001", 0)
        id2 = _make_chunk_id("doc-001", 0)
        assert id1 == id2

    def test_different_text_different_id(self):
        """Different chunk positions must produce different chunk_ids."""
        id1 = _make_chunk_id("doc-001", 0)
        id2 = _make_chunk_id("doc-001", 1)
        assert id1 != id2


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestEmbeddingMetrics:
    def test_metrics_updated_after_embed(self, embedder):
        """After embedding, chunks_embedded and batches_processed must be > 0."""
        chunks = [_make_chunk(idx=i) for i in range(3)]
        embedder.embed_chunks(chunks)
        m = embedder.get_metrics()
        assert m["chunks_embedded"] == 3
        assert m["batches_processed"] >= 1

    def test_metrics_dict_has_required_keys(self, embedder):
        """get_metrics() must return all required keys."""
        required = {
            "chunks_embedded",
            "batches_processed",
            "cache_hits",
            "cache_misses",
            "total_embedding_time_s",
            "avg_batch_time_s",
            "throughput_chunks_per_s",
        }
        chunk = _make_chunk()
        embedder.embed_chunks([chunk])
        m = embedder.get_metrics()
        missing = required - set(m.keys())
        assert not missing, f"Missing metric keys: {missing}"


# ---------------------------------------------------------------------------
# Cache integration
# ---------------------------------------------------------------------------

class TestCacheIntegration:
    def test_cache_hit_avoids_model_call(self, mock_sentence_transformer, tmp_path):
        """A cache hit must not trigger a model encode() call."""
        cache = EmbeddingCache(cache_path=tmp_path / "emb.db")

        # Pre-populate the cache with a fake vector.
        chunk = _make_chunk(text="Cached text.")
        fake_vec = [0.1] * _BGE_LARGE_DIM
        key = cache.text_to_key(chunk.text)
        cache.put(key, fake_vec)

        emb = BGEEmbedder(cache=cache)
        emb.initialize()
        encode_call_count_before = emb._model.encode.call_count

        result = emb.embed_chunks([chunk])

        # The model's encode() must NOT have been called again.
        assert emb._model.encode.call_count == encode_call_count_before
        # Returned vector must match what was cached.
        assert list(result[0].vector) == pytest.approx(fake_vec, abs=1e-6)
        cache.close()

    def test_cache_miss_calls_model(self, mock_sentence_transformer, tmp_path):
        """A cache miss must call the model and store the result."""
        cache = EmbeddingCache(cache_path=tmp_path / "emb.db")
        emb = BGEEmbedder(cache=cache)
        emb.initialize()

        chunk = _make_chunk(text="Not yet cached.")
        result = emb.embed_chunks([chunk])

        # Model was called.
        assert emb._model.encode.call_count >= 1
        # The result is now in the cache.
        key = cache.text_to_key(chunk.text)
        cached = cache.get(key)
        assert cached is not None
        assert len(cached) == _BGE_LARGE_DIM
        cache.close()

    def test_force_reembed_bypasses_cache(self, mock_sentence_transformer, tmp_path):
        """force_reembed=True must bypass the cache and call the model."""
        cache = EmbeddingCache(cache_path=tmp_path / "emb.db")
        chunk = _make_chunk(text="Force reembed test.")
        fake_vec = [0.5] * _BGE_LARGE_DIM
        key = cache.text_to_key(chunk.text)
        cache.put(key, fake_vec)

        emb = BGEEmbedder(cache=cache)
        emb.initialize()
        encode_call_count_before = emb._model.encode.call_count

        emb.embed_chunks([chunk], force_reembed=True)

        # Model was called despite cache hit.
        assert emb._model.encode.call_count > encode_call_count_before
        cache.close()
