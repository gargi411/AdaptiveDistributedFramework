"""Unit tests for EmbeddingCache (Phase 4.3).

Tests the SHA-256-keyed SQLite persistent embedding cache.

Tested requirements:
    - cache miss returns None
    - cache hit returns the stored vector
    - persistence across cache instances (same file)
    - SHA-256 key derivation correctness
    - invalidation removes a single entry
    - corrupt cache handling (wrong byte size -> cache miss)
    - clear() empties the cache
    - len() returns correct count
    - put() raises on wrong-dimension vector
"""

from __future__ import annotations

import hashlib
import struct
import pytest
from pathlib import Path

from adaptive_framework.rag.embedding.embedding_cache import (
    EmbeddingCache,
    _sha256_key,
    _EMBEDDING_DIM,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def cache(tmp_path: Path) -> EmbeddingCache:
    """Return a fresh EmbeddingCache backed by a temporary file."""
    c = EmbeddingCache(cache_path=tmp_path / "test_emb.db")
    yield c
    c.close()


def _make_vector(dim: int = _EMBEDDING_DIM, seed: float = 0.1) -> list[float]:
    """Return a deterministic float vector of the given dimension."""
    return [seed + i * 0.0001 for i in range(dim)]


def _make_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Key derivation
# ---------------------------------------------------------------------------

class TestSHA256Key:
    def test_deterministic(self):
        """Same text always produces same SHA-256 key."""
        text = "Hello, biomedical world!"
        assert _sha256_key(text) == _sha256_key(text)

    def test_different_texts_different_keys(self):
        """Different texts must produce different keys (collision is vanishingly rare)."""
        assert _sha256_key("text A") != _sha256_key("text B")

    def test_key_is_hex_string(self):
        """SHA-256 digest must be a 64-character lowercase hex string."""
        key = _sha256_key("some text")
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_text_to_key_matches_sha256(self, cache):
        """EmbeddingCache.text_to_key() must produce the same result as _sha256_key()."""
        text = "Neural network embedding test."
        assert cache.text_to_key(text) == _sha256_key(text)


# ---------------------------------------------------------------------------
# Miss
# ---------------------------------------------------------------------------

class TestCacheMiss:
    def test_get_on_empty_cache_returns_none(self, cache):
        """A cache miss returns None."""
        result = cache.get(_make_key("nonexistent text"))
        assert result is None

    def test_get_unknown_key_returns_none(self, cache):
        """An unknown key returns None without error."""
        cache.put(_make_key("known text"), _make_vector())
        result = cache.get(_make_key("unknown text"))
        assert result is None


# ---------------------------------------------------------------------------
# Hit
# ---------------------------------------------------------------------------

class TestCacheHit:
    def test_put_then_get_returns_vector(self, cache):
        """After put(), get() with the same key returns the stored vector."""
        key = _make_key("retrieve me")
        vec = _make_vector()
        cache.put(key, vec)
        result = cache.get(key)
        assert result is not None
        assert len(result) == _EMBEDDING_DIM
        assert result == pytest.approx(vec, abs=1e-5)

    def test_overwrite_existing_key(self, cache):
        """put() with an existing key replaces the old value."""
        key = _make_key("overwrite")
        old_vec = _make_vector(seed=0.1)
        new_vec = _make_vector(seed=0.9)
        cache.put(key, old_vec)
        cache.put(key, new_vec)
        result = cache.get(key)
        assert result == pytest.approx(new_vec, abs=1e-5)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_survives_reopen(self, tmp_path):
        """Cached vectors survive closing and reopening the database."""
        db_path = tmp_path / "persist_test.db"
        text = "This text should survive a restart."
        vec = _make_vector(seed=0.42)
        key = _make_key(text)

        # Write in one instance.
        c1 = EmbeddingCache(cache_path=db_path)
        c1.put(key, vec)
        c1.close()

        # Read in a new instance pointing to the same file.
        c2 = EmbeddingCache(cache_path=db_path)
        result = c2.get(key)
        c2.close()

        assert result is not None
        assert result == pytest.approx(vec, abs=1e-5)

    def test_multiple_entries_persist(self, tmp_path):
        """Multiple entries all persist across cache instances."""
        db_path = tmp_path / "multi.db"
        texts = [f"Sentence {i} for biomedical embedding." for i in range(10)]
        vectors = {t: _make_vector(seed=0.1 * i) for i, t in enumerate(texts)}

        c1 = EmbeddingCache(cache_path=db_path)
        for text, vec in vectors.items():
            c1.put(_make_key(text), vec)
        c1.close()

        c2 = EmbeddingCache(cache_path=db_path)
        for text, expected in vectors.items():
            result = c2.get(_make_key(text))
            assert result is not None, f"Missing cached vector for: {text}"
            assert result == pytest.approx(expected, abs=1e-5)
        c2.close()


# ---------------------------------------------------------------------------
# Invalidation
# ---------------------------------------------------------------------------

class TestInvalidation:
    def test_invalidate_existing_key(self, cache):
        """invalidate() must remove the entry and return True."""
        key = _make_key("to be deleted")
        cache.put(key, _make_vector())
        removed = cache.invalidate(key)
        assert removed is True
        assert cache.get(key) is None

    def test_invalidate_nonexistent_key(self, cache):
        """invalidate() on a nonexistent key must return False."""
        removed = cache.invalidate(_make_key("does not exist"))
        assert removed is False

    def test_invalidate_only_removes_target(self, cache):
        """invalidate() must not affect other entries."""
        key_a = _make_key("keep me")
        key_b = _make_key("delete me")
        vec_a = _make_vector(seed=0.1)
        vec_b = _make_vector(seed=0.9)
        cache.put(key_a, vec_a)
        cache.put(key_b, vec_b)
        cache.invalidate(key_b)
        result_a = cache.get(key_a)
        assert result_a is not None
        assert result_a == pytest.approx(vec_a, abs=1e-5)
        assert cache.get(key_b) is None


# ---------------------------------------------------------------------------
# Corrupt cache handling
# ---------------------------------------------------------------------------

class TestCorruptCacheHandling:
    def test_corrupt_entry_treated_as_miss(self, tmp_path):
        """A corrupt (wrong-size) entry must be treated as a cache miss and deleted."""
        db_path = tmp_path / "corrupt.db"
        cache = EmbeddingCache(cache_path=db_path)
        key = _make_key("corrupt text")

        # Manually insert a corrupt entry (wrong byte count).
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT INTO embeddings (cache_key, vector_bytes, created_at) VALUES (?, ?, ?)",
            (key, b"\x00\x01\x02", "2026-01-01T00:00:00+00:00"),
        )
        conn.commit()
        conn.close()

        # Should not raise; should return None (miss).
        result = cache.get(key)
        assert result is None

        # The corrupt entry should have been deleted.
        assert cache.get(key) is None
        cache.close()


# ---------------------------------------------------------------------------
# len() and clear()
# ---------------------------------------------------------------------------

class TestLenAndClear:
    def test_len_empty(self, cache):
        """Empty cache must report len 0."""
        assert len(cache) == 0

    def test_len_after_put(self, cache):
        """len() must reflect the number of stored entries."""
        for i in range(5):
            cache.put(_make_key(f"text {i}"), _make_vector(seed=float(i)))
        assert len(cache) == 5

    def test_clear_empties_cache(self, cache):
        """clear() must remove all entries and return the deleted count."""
        for i in range(3):
            cache.put(_make_key(f"entry {i}"), _make_vector())
        n_deleted = cache.clear()
        assert n_deleted == 3
        assert len(cache) == 0

    def test_clear_on_empty_cache(self, cache):
        """clear() on an empty cache must return 0."""
        assert cache.clear() == 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_put_wrong_dimension_raises(self, cache):
        """put() with wrong-dimension vector must raise ValueError."""
        key = _make_key("wrong dim")
        with pytest.raises(ValueError, match="expected"):
            cache.put(key, [0.1, 0.2, 0.3])  # dim=3 != 1024

    def test_put_correct_dimension_ok(self, cache):
        """put() with correct-dimension vector must not raise."""
        key = _make_key("correct dim")
        cache.put(key, _make_vector())  # dim=1024 — should not raise
