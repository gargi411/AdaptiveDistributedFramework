"""tests/unit/test_document_registry.py — Unit tests for DocumentRegistry.

Tests all public API methods of DocumentRegistry and DocumentStatus.
No network access, no disk I/O (uses in-memory state only).
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

import sys
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from adaptive_framework.dataset_builder.document_registry import (
    DocumentRegistry,
    DocumentStatus,
    RegistrySummary,
)
from adaptive_framework.models.document import PDFMetadata


# =============================================================
# Helpers
# =============================================================


def _make_meta(
    pages: int = 10,
    source_type: str | None = "digital",
    size_mb: float = 1.0,
    path: str = "/fake/paper.pdf",
) -> PDFMetadata:
    """Create a minimal PDFMetadata for testing."""
    return PDFMetadata(
        pages=pages,
        estimated_size_mb=size_mb,
        file_path=path,
        source_type=source_type,
        language="en",
    )


def _make_registry(count: int = 3) -> tuple[DocumentRegistry, list[str]]:
    """Create a DocumentRegistry with ``count`` registered documents."""
    registry = DocumentRegistry()
    ids: list[str] = []
    for i in range(count):
        meta = _make_meta(pages=10 + i, path=f"/fake/paper_{i}.pdf")
        doc_id = registry.register(meta)
        ids.append(doc_id)
    return registry, ids


# =============================================================
# Tests — Registration
# =============================================================


class TestDocumentRegistryRegistration:
    """Tests for register() and register_batch()."""

    def test_register_returns_document_id(self) -> None:
        """register() must return a non-empty string ID."""
        registry = DocumentRegistry()
        meta = _make_meta()
        doc_id = registry.register(meta)
        assert isinstance(doc_id, str)
        assert len(doc_id) > 0

    def test_register_uses_metadata_document_id(self) -> None:
        """The returned ID must match PDFMetadata.document_id."""
        registry = DocumentRegistry()
        meta = _make_meta()
        doc_id = registry.register(meta)
        assert doc_id == meta.document_id

    def test_register_twice_same_doc_is_noop(self) -> None:
        """Re-registering the same metadata must not duplicate the entry."""
        registry = DocumentRegistry()
        meta = _make_meta()
        id1 = registry.register(meta)
        id2 = registry.register(meta)
        assert id1 == id2
        assert len(registry) == 1

    def test_register_batch_returns_all_ids(self) -> None:
        """register_batch() must return one ID per input metadata record."""
        registry = DocumentRegistry()
        metas = [_make_meta(path=f"/fake/{i}.pdf") for i in range(5)]
        ids = registry.register_batch(metas)
        assert len(ids) == 5
        assert all(isinstance(i, str) for i in ids)

    def test_register_batch_unique_ids(self) -> None:
        """All IDs returned by register_batch() must be unique."""
        registry = DocumentRegistry()
        metas = [_make_meta(path=f"/fake/{i}.pdf") for i in range(10)]
        ids = registry.register_batch(metas)
        assert len(set(ids)) == len(ids), "Duplicate document IDs detected."

    def test_len_reflects_registered_count(self) -> None:
        """len(registry) must equal the number of unique registered documents."""
        registry, _ = _make_registry(7)
        assert len(registry) == 7

    def test_contains_registered_id(self) -> None:
        """``in`` operator must return True for registered IDs."""
        registry, ids = _make_registry(3)
        for doc_id in ids:
            assert doc_id in registry

    def test_not_contains_unknown_id(self) -> None:
        """``in`` operator must return False for unknown IDs."""
        registry, _ = _make_registry(3)
        assert "nonexistent-id-xyz" not in registry


# =============================================================
# Tests — Status management
# =============================================================


class TestDocumentRegistryStatus:
    """Tests for get_status(), set_status(), and status transitions."""

    def test_initial_status_is_pending(self) -> None:
        """Newly registered documents must have PENDING status."""
        registry, ids = _make_registry(3)
        for doc_id in ids:
            assert registry.get_status(doc_id) == DocumentStatus.PENDING

    def test_set_status_changes_status(self) -> None:
        """set_status() must persist the new status."""
        registry, ids = _make_registry(1)
        registry.set_status(ids[0], DocumentStatus.IN_PROGRESS)
        assert registry.get_status(ids[0]) == DocumentStatus.IN_PROGRESS

    def test_set_status_all_transitions(self) -> None:
        """All valid status transitions must be accepted."""
        registry, ids = _make_registry(1)
        doc_id = ids[0]
        for status in DocumentStatus:
            registry.set_status(doc_id, status)
            assert registry.get_status(doc_id) == status

    def test_set_status_unknown_id_raises(self) -> None:
        """set_status() must raise KeyError for an unregistered ID."""
        registry = DocumentRegistry()
        with pytest.raises(KeyError):
            registry.set_status("not-registered", DocumentStatus.COMPLETED)

    def test_get_status_unknown_id_returns_pending(self) -> None:
        """get_status() for unknown ID returns PENDING (safe default)."""
        registry = DocumentRegistry()
        status = registry.get_status("nonexistent")
        assert status == DocumentStatus.PENDING


# =============================================================
# Tests — Reads
# =============================================================


class TestDocumentRegistryReads:
    """Tests for get_metadata(), get_path(), all_metadata(), etc."""

    def test_get_metadata_returns_correct_record(self) -> None:
        """get_metadata() must return the original PDFMetadata for a known ID."""
        registry = DocumentRegistry()
        meta = _make_meta(pages=42, path="/data/special.pdf")
        doc_id = registry.register(meta)
        retrieved = registry.get_metadata(doc_id)
        assert retrieved is not None
        assert retrieved.pages == 42
        assert retrieved.file_path == "/data/special.pdf"

    def test_get_metadata_unknown_id_returns_none(self) -> None:
        """get_metadata() must return None for unknown IDs."""
        registry = DocumentRegistry()
        assert registry.get_metadata("not-here") is None

    def test_get_path_returns_path_object(self) -> None:
        """get_path() must return a Path object matching the metadata file_path."""
        registry = DocumentRegistry()
        meta = _make_meta(path="/data/doc.pdf")
        doc_id = registry.register(meta)
        path = registry.get_path(doc_id)
        assert isinstance(path, Path)
        assert path == Path("/data/doc.pdf")

    def test_get_path_unknown_id_returns_none(self) -> None:
        """get_path() must return None for unknown IDs."""
        registry = DocumentRegistry()
        assert registry.get_path("ghost") is None

    def test_all_metadata_count(self) -> None:
        """all_metadata() must return one record per registered document."""
        registry, _ = _make_registry(5)
        all_meta = registry.all_metadata()
        assert len(all_meta) == 5
        assert all(isinstance(m, PDFMetadata) for m in all_meta)

    def test_all_ids_count(self) -> None:
        """all_ids() must return the correct number of IDs."""
        registry, ids = _make_registry(4)
        assert set(registry.all_ids()) == set(ids)

    def test_pending_returns_pending_only(self) -> None:
        """pending() must return only PENDING documents."""
        registry, ids = _make_registry(4)
        registry.set_status(ids[0], DocumentStatus.COMPLETED)
        registry.set_status(ids[1], DocumentStatus.FAILED)
        pending = registry.pending()
        assert len(pending) == 2

    def test_completed_returns_completed_ids(self) -> None:
        """completed() must return only IDs with COMPLETED status."""
        registry, ids = _make_registry(3)
        registry.set_status(ids[0], DocumentStatus.COMPLETED)
        registry.set_status(ids[2], DocumentStatus.COMPLETED)
        completed = registry.completed()
        assert set(completed) == {ids[0], ids[2]}

    def test_failed_returns_failed_ids(self) -> None:
        """failed() must return only IDs with FAILED status."""
        registry, ids = _make_registry(3)
        registry.set_status(ids[1], DocumentStatus.FAILED)
        failed = registry.failed()
        assert failed == [ids[1]]


# =============================================================
# Tests — UnifiedDocument slot
# =============================================================


class TestDocumentRegistryUnifiedDocument:
    """Tests for set_unified_document() and get_unified_document()."""

    def test_get_unified_document_initially_none(self) -> None:
        """Before processing, get_unified_document() must return None."""
        registry, ids = _make_registry(1)
        assert registry.get_unified_document(ids[0]) is None

    def test_set_unified_document_stores_object(self) -> None:
        """set_unified_document() must persist and return the stored object."""
        registry, ids = _make_registry(1)
        fake_doc = {"type": "UnifiedDocument", "pages": []}
        registry.set_unified_document(ids[0], fake_doc)
        retrieved = registry.get_unified_document(ids[0])
        assert retrieved is fake_doc

    def test_set_unified_document_transitions_to_completed(self) -> None:
        """set_unified_document() must automatically set status to COMPLETED."""
        registry, ids = _make_registry(1)
        registry.set_unified_document(ids[0], object())
        assert registry.get_status(ids[0]) == DocumentStatus.COMPLETED

    def test_set_unified_document_unknown_id_raises(self) -> None:
        """set_unified_document() for unknown ID must raise KeyError."""
        registry = DocumentRegistry()
        with pytest.raises(KeyError):
            registry.set_unified_document("ghost", object())

    def test_get_unified_document_unknown_id_returns_none(self) -> None:
        """get_unified_document() for unknown ID must return None."""
        registry = DocumentRegistry()
        assert registry.get_unified_document("nope") is None


# =============================================================
# Tests — Summary
# =============================================================


class TestDocumentRegistrySummary:
    """Tests for the summary() method and RegistrySummary dataclass."""

    def test_empty_registry_summary(self) -> None:
        """Empty registry must produce a zero-filled summary."""
        registry = DocumentRegistry()
        s = registry.summary()
        assert isinstance(s, RegistrySummary)
        assert s.total_pdfs == 0
        assert s.total_pages == 0
        assert s.pending == 0

    def test_summary_total_pdfs(self) -> None:
        """summary().total_pdfs must equal number of registered documents."""
        registry, _ = _make_registry(6)
        assert registry.summary().total_pdfs == 6

    def test_summary_total_pages(self) -> None:
        """summary().total_pages must equal sum of all page counts."""
        registry = DocumentRegistry()
        metas = [_make_meta(pages=p, path=f"/f{i}.pdf") for i, p in enumerate([5, 10, 15])]
        registry.register_batch(metas)
        assert registry.summary().total_pages == 30

    def test_summary_digital_scanned_unknown(self) -> None:
        """Source type counts must reflect the registered metadata."""
        registry = DocumentRegistry()
        registry.register(_make_meta(source_type="digital", path="/d.pdf"))
        registry.register(_make_meta(source_type="scanned", path="/s.pdf"))
        registry.register(_make_meta(source_type=None, path="/u.pdf"))
        s = registry.summary()
        assert s.digital_pdfs == 1
        assert s.scanned_pdfs == 1
        assert s.unknown_pdfs == 1

    def test_summary_status_counts(self) -> None:
        """Status counts in summary must reflect actual document statuses."""
        registry, ids = _make_registry(5)
        registry.set_status(ids[0], DocumentStatus.COMPLETED)
        registry.set_status(ids[1], DocumentStatus.FAILED)
        registry.set_status(ids[2], DocumentStatus.IN_PROGRESS)
        s = registry.summary()
        assert s.completed == 1
        assert s.failed == 1
        assert s.in_progress == 1
        assert s.pending == 2

    def test_summary_avg_pages(self) -> None:
        """summary().avg_pages must be the mean page count."""
        registry = DocumentRegistry()
        metas = [_make_meta(pages=p, path=f"/f{i}.pdf") for i, p in enumerate([10, 20])]
        registry.register_batch(metas)
        s = registry.summary()
        assert s.avg_pages == pytest.approx(15.0, abs=0.01)

    def test_summary_metadata_cached_flag(self) -> None:
        """metadata_cached must reflect the value passed at construction."""
        r_true = DocumentRegistry(metadata_cached=True)
        r_false = DocumentRegistry(metadata_cached=False)
        assert r_true.summary().metadata_cached is True
        assert r_false.summary().metadata_cached is False

    def test_summary_to_dict_keys(self) -> None:
        """to_dict() must contain all expected keys."""
        registry, _ = _make_registry(2)
        d = registry.summary().to_dict()
        expected_keys = {
            "total_pdfs", "total_pages", "digital_pdfs", "scanned_pdfs",
            "unknown_pdfs", "pending", "queued", "in_progress", "completed",
            "failed", "avg_pages", "avg_size_mb", "metadata_cached",
        }
        assert expected_keys.issubset(set(d.keys()))


# =============================================================
# Tests — Thread safety
# =============================================================


class TestDocumentRegistryThreadSafety:
    """Tests for thread-safe concurrent access."""

    def test_concurrent_registration(self) -> None:
        """Concurrent register() calls from multiple threads must be safe."""
        registry = DocumentRegistry()
        errors: list[Exception] = []

        def _register_batch(start: int, count: int) -> None:
            try:
                for i in range(start, start + count):
                    meta = _make_meta(path=f"/thread/doc_{i}.pdf")
                    registry.register(meta)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=_register_batch, args=(i * 20, 20))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        assert len(registry) == 100

    def test_concurrent_status_updates(self) -> None:
        """Concurrent set_status() calls must not corrupt internal state."""
        registry, ids = _make_registry(50)
        errors: list[Exception] = []

        def _update_statuses(doc_ids: list[str]) -> None:
            try:
                for doc_id in doc_ids:
                    registry.set_status(doc_id, DocumentStatus.IN_PROGRESS)
            except Exception as exc:
                errors.append(exc)

        half = len(ids) // 2
        t1 = threading.Thread(target=_update_statuses, args=(ids[:half],))
        t2 = threading.Thread(target=_update_statuses, args=(ids[half:],))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Thread errors: {errors}"
        for doc_id in ids:
            assert registry.get_status(doc_id) == DocumentStatus.IN_PROGRESS
