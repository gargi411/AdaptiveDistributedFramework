"""EmbeddingCache — persistent SHA-256-keyed embedding cache (Phase 4.3).

Every unique piece of chunk text maps to exactly one cached vector via a
SHA-256 digest of the UTF-8-encoded text.  The cache is backed by an
SQLite database so it survives process restarts.

Cache key:   SHA-256(chunk.text.encode("utf-8")).hexdigest()
Cache value: vector stored as raw bytes (struct-packed float32 array).

Persistence path (default):
    outputs/rag/cache/embeddings.db

Concurrency:
    Each process uses its own SQLite connection.  Multiple concurrent
    processes are safe because SQLite handles write serialisation.  For
    high-throughput parallel embedding (e.g. via Ray), each worker opens
    its own connection to the same shared file.

Safe handling of corrupt/missing cache:
    - Missing database: created automatically on first access.
    - Corrupt row (bad byte count): treated as a cache miss; the row is
      deleted and the embedding is recomputed.
    - Locked database: sqlite3 retries are handled by the timeout parameter.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import struct
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Number of float32 values per vector (BAAI/bge-large-en-v1.5).
_EMBEDDING_DIM = 1024

# Number of bytes for one 1024-dimensional float32 vector.
_VECTOR_BYTES = _EMBEDDING_DIM * 4  # 4 bytes per float32

# SQLite busy-wait timeout in milliseconds.
_BUSY_TIMEOUT_MS = 5000


def _sha256_key(text: str) -> str:
    """Compute the SHA-256 hex digest of a UTF-8-encoded text string.

    Args:
        text: Chunk text to hash.

    Returns:
        64-character lowercase hex string.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Persistent, content-addressed embedding cache backed by SQLite.

    Args:
        cache_path: Path to the SQLite database file.  The parent
            directory is created if it does not exist.
        embedding_dim: Expected vector dimension.  Used to validate
            deserialized vectors and detect corrupt entries.

    Example:
        >>> cache = EmbeddingCache("outputs/rag/cache/embeddings.db")
        >>> key = cache.text_to_key("Hello world")
        >>> cache.get(key)  # None on miss
        >>> cache.put(key, [0.1, 0.2, ...])
        >>> cache.get(key)  # returns the vector
    """

    _TABLE_DDL = """
        CREATE TABLE IF NOT EXISTS embeddings (
            cache_key  TEXT PRIMARY KEY,
            vector_bytes BLOB NOT NULL,
            created_at TEXT NOT NULL
        );
    """

    def __init__(
        self,
        cache_path: str | Path = "outputs/rag/cache/embeddings.db",
        embedding_dim: int = _EMBEDDING_DIM,
    ) -> None:
        self._path = Path(cache_path)
        self._embedding_dim = embedding_dim
        self._vector_bytes = embedding_dim * 4
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._open()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def _open(self) -> None:
        """Open (or create) the SQLite connection and ensure the schema."""
        try:
            self._conn = sqlite3.connect(
                str(self._path),
                timeout=_BUSY_TIMEOUT_MS / 1000,
                check_same_thread=False,
            )
            self._conn.execute(self._TABLE_DDL)
            self._conn.commit()
        except sqlite3.Error as exc:
            log.error("EmbeddingCache: failed to open %s: %s", self._path, exc)
            self._conn = None

    def close(self) -> None:
        """Flush and close the SQLite connection."""
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
            self._conn = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def text_to_key(text: str) -> str:
        """Return the cache key (SHA-256 hex digest) for a text string.

        Args:
            text: The chunk text to hash.

        Returns:
            64-character hex digest string.
        """
        return _sha256_key(text)

    def get(self, cache_key: str) -> Optional[list[float]]:
        """Retrieve a cached vector, or None on a cache miss.

        Args:
            cache_key: SHA-256 hex digest returned by text_to_key().

        Returns:
            List of floats (length == embedding_dim) if found, else None.
            Returns None (and deletes the row) if the stored bytes are
            corrupt (wrong size).
        """
        if self._conn is None:
            return None
        try:
            row = self._conn.execute(
                "SELECT vector_bytes FROM embeddings WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        except sqlite3.Error as exc:
            log.warning("EmbeddingCache.get: sqlite error %s", exc)
            return None

        if row is None:
            return None

        raw: bytes = row[0]
        if len(raw) != self._vector_bytes:
            log.warning(
                "EmbeddingCache: corrupt entry (key=%s, bytes=%d, expected=%d) — deleting",
                cache_key,
                len(raw),
                self._vector_bytes,
            )
            self._delete(cache_key)
            return None

        n = self._embedding_dim
        return list(struct.unpack(f"{n}f", raw))

    def put(self, cache_key: str, vector: list[float]) -> None:
        """Store a vector in the cache.

        If a row with the same key already exists it is replaced.

        Args:
            cache_key: SHA-256 hex digest (from text_to_key).
            vector: Float list of length embedding_dim.

        Raises:
            ValueError: If len(vector) != embedding_dim.
        """
        if len(vector) != self._embedding_dim:
            raise ValueError(
                f"EmbeddingCache.put: vector has {len(vector)} floats, "
                f"expected {self._embedding_dim}."
            )
        if self._conn is None:
            return
        from datetime import datetime, timezone

        created_at = datetime.now(timezone.utc).isoformat()
        raw = struct.pack(f"{self._embedding_dim}f", *vector)
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO embeddings (cache_key, vector_bytes, created_at)"
                " VALUES (?, ?, ?)",
                (cache_key, raw, created_at),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            log.warning("EmbeddingCache.put: sqlite error %s", exc)

    def invalidate(self, cache_key: str) -> bool:
        """Remove a specific entry from the cache.

        Args:
            cache_key: SHA-256 hex digest to remove.

        Returns:
            True if a row was deleted, False if the key was not found.
        """
        return self._delete(cache_key)

    def clear(self) -> int:
        """Remove all entries from the cache.

        Returns:
            Number of rows deleted.
        """
        if self._conn is None:
            return 0
        try:
            cursor = self._conn.execute("DELETE FROM embeddings")
            self._conn.commit()
            return cursor.rowcount
        except sqlite3.Error as exc:
            log.warning("EmbeddingCache.clear: sqlite error %s", exc)
            return 0

    def __len__(self) -> int:
        """Return the number of cached embeddings."""
        if self._conn is None:
            return 0
        try:
            row = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
            return int(row[0]) if row else 0
        except sqlite3.Error:
            return 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _delete(self, cache_key: str) -> bool:
        """Delete one row by key.  Returns True if a row was deleted."""
        if self._conn is None:
            return False
        try:
            cursor = self._conn.execute(
                "DELETE FROM embeddings WHERE cache_key = ?", (cache_key,)
            )
            self._conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as exc:
            log.warning("EmbeddingCache._delete: sqlite error %s", exc)
            return False

    def __repr__(self) -> str:
        return (
            f"EmbeddingCache(path='{self._path}', "
            f"entries={len(self)}, "
            f"dim={self._embedding_dim})"
        )
