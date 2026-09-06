"""FAISSPersistence — save and load a FAISSManager state (Phase 4.3).

Persists three files under outputs/rag/index/:

    index.faiss        Binary FAISS index (via faiss.write_index).
    metadata.pkl       Python pickle of the metadata maps and Chunk objects.
    index_stats.json   Human-readable JSON summary of the index.

index_stats.json contains:
    total_chunks        Number of vectors in the FAISS index.
    total_documents     Number of unique document_ids.
    embedding_dimension Embedding vector dimension.
    index_type          'flat', 'ivf', or 'hnsw'.
    model_name          Embedding model identifier.

The pickle file stores:
    faiss_id_to_chunk_id  dict[int, str]
    chunk_id_to_chunk     dict[str, Chunk]
    stored_vectors        dict[str, np.ndarray]  (for rebuild support)
    dim                   int
    index_type            str
    model_name            str

Safe round-trip guarantee:
    A FAISSManager saved with persist() can be loaded with load() into a
    fresh FAISSManager.  After loading, search() and delete_document() work
    correctly because all metadata maps are restored.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

_DEFAULT_INDEX_DIR = Path("outputs/rag/index")


class FAISSPersistence:
    """Handles saving and loading FAISSManager state.

    Args:
        index_dir: Directory where all three persistence files are written.
            Default: outputs/rag/index.

    Example:
        >>> persistence = FAISSPersistence()
        >>> persistence.save(manager)
        >>> manager2 = FAISSManager(dim=1024)
        >>> persistence.load(manager2)
    """

    def __init__(self, index_dir: str | Path = _DEFAULT_INDEX_DIR) -> None:
        self._index_dir = Path(index_dir)

    @property
    def index_path(self) -> Path:
        """Path to the FAISS binary index file."""
        return self._index_dir / "index.faiss"

    @property
    def metadata_path(self) -> Path:
        """Path to the pickle metadata file."""
        return self._index_dir / "metadata.pkl"

    @property
    def stats_path(self) -> Path:
        """Path to the JSON stats file."""
        return self._index_dir / "index_stats.json"

    def save(self, manager: object) -> None:
        """Persist the FAISSManager state to disk.

        Args:
            manager: A FAISSManager instance to persist.

        Raises:
            RuntimeError: If faiss is not installed.
            OSError: If the directory cannot be created or files written.
        """
        from adaptive_framework.rag.vector_store.faiss_manager import FAISSManager

        assert isinstance(manager, FAISSManager)

        try:
            import faiss  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("faiss-cpu is not installed.") from exc

        self._index_dir.mkdir(parents=True, exist_ok=True)

        # 1. Write FAISS binary index.
        faiss.write_index(manager._index, str(self.index_path))
        log.info("FAISSPersistence: wrote FAISS index to '%s'.", self.index_path)

        # 2. Write metadata pickle.
        meta: dict[str, Any] = {
            "faiss_id_to_chunk_id": manager._faiss_id_to_chunk_id,
            "chunk_id_to_chunk": manager._chunk_id_to_chunk,
            "stored_vectors": manager._stored_vectors,
            "dim": manager._dim,
            "index_type": manager._index_type,
            "model_name": manager._model_name,
        }
        with open(self.metadata_path, "wb") as fh:
            pickle.dump(meta, fh, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("FAISSPersistence: wrote metadata to '%s'.", self.metadata_path)

        # 3. Write human-readable stats JSON.
        stats = manager.get_index_stats()
        with open(self.stats_path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2)
        log.info("FAISSPersistence: wrote stats to '%s'.", self.stats_path)

    def load(self, manager: object) -> None:
        """Restore FAISSManager state from disk into an existing manager instance.

        Args:
            manager: An empty FAISSManager instance whose internal state
                will be replaced with the persisted state.

        Raises:
            FileNotFoundError: If the index or metadata files do not exist.
            RuntimeError: If faiss is not installed.
            ValueError: If the loaded metadata has an incompatible dimension.
        """
        from adaptive_framework.rag.vector_store.faiss_manager import FAISSManager

        assert isinstance(manager, FAISSManager)

        try:
            import faiss  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("faiss-cpu is not installed.") from exc

        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"FAISS metadata not found: {self.metadata_path}")

        # 1. Load FAISS binary index.
        manager._index = faiss.read_index(str(self.index_path))
        log.info(
            "FAISSPersistence: loaded FAISS index from '%s' (%d vectors).",
            self.index_path,
            manager._index.ntotal,
        )

        # 2. Load metadata pickle.
        with open(self.metadata_path, "rb") as fh:
            meta: dict[str, Any] = pickle.load(fh)

        loaded_dim: int = meta.get("dim", manager._dim)
        if loaded_dim != manager._dim:
            raise ValueError(
                f"FAISSPersistence.load: dimension mismatch — "
                f"manager dim={manager._dim}, persisted dim={loaded_dim}."
            )

        manager._faiss_id_to_chunk_id = meta["faiss_id_to_chunk_id"]
        manager._chunk_id_to_chunk = meta["chunk_id_to_chunk"]
        manager._stored_vectors = meta.get("stored_vectors", {})
        manager._index_type = meta.get("index_type", manager._index_type)
        manager._model_name = meta.get("model_name", manager._model_name)

        log.info(
            "FAISSPersistence: loaded metadata — %d chunks, %d documents.",
            len(manager._chunk_id_to_chunk),
            len({c.document_id for c in manager._chunk_id_to_chunk.values()}),
        )

    def exists(self) -> bool:
        """Return True if a persisted index exists on disk."""
        return self.index_path.exists() and self.metadata_path.exists()

    def read_stats(self) -> dict[str, Any]:
        """Read the index_stats.json file without loading the full index.

        Returns:
            Dictionary from index_stats.json, or empty dict if not found.
        """
        if not self.stats_path.exists():
            return {}
        with open(self.stats_path, encoding="utf-8") as fh:
            return json.load(fh)  # type: ignore[no-any-return]
