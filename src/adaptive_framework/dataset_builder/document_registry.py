"""Document Registry — Phase 3.5.

The ``DocumentRegistry`` is the single source of truth for every document in
the Adaptive Distributed Framework pipeline. It maps:

    document_id (UUID4 str)
        → file_path (Path)
        → PDFMetadata
        → DocumentStatus (PENDING | QUEUED | IN_PROGRESS | COMPLETED | FAILED)
        → UnifiedDocument | None   ← populated by coordinator after Phase 3

This registry is the bridge between Phase 3 (Document Processing) and Phase 4
(Knowledge Layer / RAG). Phase 4 can look up any document by ID:

    registry.get_unified_document(doc_id)

instead of scanning folders or rebuilding metadata objects from scratch.

Design:
    - Thread-safe: all public mutations are protected by ``threading.RLock``.
    - ID stability: the document_id from ``PDFMetadata`` is preserved and
      remains the primary key for the lifetime of the run.
    - ``UnifiedDocument`` slot is typed as ``object | None`` so this module
      has zero import dependency on the document_processing package.

Usage::

    from adaptive_framework.dataset_builder.document_registry import (
        DocumentRegistry,
        DocumentStatus,
    )

    registry = DocumentRegistry()
    ids = registry.register_batch(metadata_list)
    registry.set_status(ids[0], DocumentStatus.IN_PROGRESS)
    summary = registry.summary()
    print(summary.total_pdfs, summary.pending)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from adaptive_framework.models.document import PDFMetadata

logger = logging.getLogger("adaptive_framework.dataset_builder.registry")


# =============================================================
# DocumentStatus
# =============================================================


class DocumentStatus(str, Enum):
    """Lifecycle status of a document in the registry.

    Values:
        PENDING:     Registered but not yet submitted to the scheduler.
        QUEUED:      In the scheduler's priority queue.
        IN_PROGRESS: Currently being processed by a worker.
        COMPLETED:   Processing finished successfully; UnifiedDocument available.
        FAILED:      Processing failed; retry or escalation required.
    """

    PENDING = "pending"
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================
# RegistrySummary
# =============================================================


@dataclass
class RegistrySummary:
    """Aggregate statistics for the current state of the document registry.

    Produced by ``DocumentRegistry.summary()`` and consumed by the dashboard
    Dataset Health panel and the ``start_dev_cluster.py`` state loop.

    Attributes:
        total_pdfs: Total registered documents.
        total_pages: Grand total of pages across all documents.
        digital_pdfs: Count of documents detected as native digital PDFs.
        scanned_pdfs: Count of documents detected as scanned image PDFs.
        unknown_pdfs: Count of documents with unknown/undetected source type.
        pending: Documents in PENDING status.
        queued: Documents in QUEUED status.
        in_progress: Documents in IN_PROGRESS status.
        completed: Documents in COMPLETED status.
        failed: Documents in FAILED status.
        avg_pages: Mean page count across all documents.
        avg_size_mb: Mean file size in MB across all documents.
        metadata_cached: Whether a metadata cache file exists on disk.
    """

    total_pdfs: int = 0
    total_pages: int = 0
    digital_pdfs: int = 0
    scanned_pdfs: int = 0
    unknown_pdfs: int = 0
    pending: int = 0
    queued: int = 0
    in_progress: int = 0
    completed: int = 0
    failed: int = 0
    avg_pages: float = 0.0
    avg_size_mb: float = 0.0
    metadata_cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary for JSON state updates.

        Returns:
            Dictionary representation of the summary.
        """
        return {
            "total_pdfs": self.total_pdfs,
            "total_pages": self.total_pages,
            "digital_pdfs": self.digital_pdfs,
            "scanned_pdfs": self.scanned_pdfs,
            "unknown_pdfs": self.unknown_pdfs,
            "pending": self.pending,
            "queued": self.queued,
            "in_progress": self.in_progress,
            "completed": self.completed,
            "failed": self.failed,
            "avg_pages": round(self.avg_pages, 2),
            "avg_size_mb": round(self.avg_size_mb, 4),
            "metadata_cached": self.metadata_cached,
        }


# =============================================================
# _RegistryEntry (internal)
# =============================================================


@dataclass
class _RegistryEntry:
    """Internal record held per document.

    Attributes:
        metadata: Full ``PDFMetadata`` for the document.
        status: Current lifecycle status.
        unified_document: Completed ``UnifiedDocument`` from Phase 3, or None.
    """

    metadata: PDFMetadata
    status: DocumentStatus = DocumentStatus.PENDING
    unified_document: object | None = None


# =============================================================
# DocumentRegistry
# =============================================================


class DocumentRegistry:
    """Thread-safe registry mapping document IDs to full pipeline state.

    The registry is the single source of truth for a processing run. Every
    document passes through it from registration (PENDING) to completion
    (COMPLETED or FAILED).

    Phase 4 entry point:
        ``registry.get_unified_document(doc_id)``

    Attributes:
        _entries: Mapping from document_id to ``_RegistryEntry``.
        _lock: Re-entrant lock for thread safety.
        _metadata_cached: Whether the metadata cache was loaded from disk.

    Example:
        >>> registry = DocumentRegistry()
        >>> ids = registry.register_batch(metadata_list)
        >>> registry.set_status(ids[0], DocumentStatus.COMPLETED)
        >>> print(registry.summary().completed)
        1
    """

    def __init__(self, metadata_cached: bool = False) -> None:
        """Initialise an empty DocumentRegistry.

        Args:
            metadata_cached: Set to True if metadata was loaded from a cache
                file (reported in ``RegistrySummary.metadata_cached``).
        """
        self._entries: dict[str, _RegistryEntry] = {}
        self._lock = threading.RLock()
        self._metadata_cached = metadata_cached

    # ------------------------------------------------------------------ #
    # Registration                                                         #
    # ------------------------------------------------------------------ #

    def register(self, metadata: PDFMetadata) -> str:
        """Register a single document and return its document_id.

        The document_id from ``PDFMetadata.document_id`` is used as the
        primary key. If a document with the same ID is already registered,
        this call is a no-op and the existing ID is returned.

        Args:
            metadata: ``PDFMetadata`` for the document to register.

        Returns:
            The document's ``document_id`` string.
        """
        doc_id = metadata.document_id
        with self._lock:
            if doc_id not in self._entries:
                self._entries[doc_id] = _RegistryEntry(metadata=metadata)
                logger.debug(
                    "Registered document '%s' (pages=%d, path='%s').",
                    doc_id,
                    metadata.pages,
                    Path(metadata.file_path).name,
                )
        return doc_id

    def register_batch(self, metadata_list: list[PDFMetadata]) -> list[str]:
        """Register multiple documents and return their IDs.

        Args:
            metadata_list: List of ``PDFMetadata`` records to register.

        Returns:
            List of document_id strings in the same order as ``metadata_list``.
        """
        ids: list[str] = []
        for meta in metadata_list:
            ids.append(self.register(meta))
        logger.info("DocumentRegistry: %d document(s) registered.", len(ids))
        return ids

    # ------------------------------------------------------------------ #
    # Reads                                                                #
    # ------------------------------------------------------------------ #

    def get_metadata(self, doc_id: str) -> PDFMetadata | None:
        """Retrieve the ``PDFMetadata`` for a document.

        Args:
            doc_id: Document identifier.

        Returns:
            ``PDFMetadata`` if found, or None.
        """
        with self._lock:
            entry = self._entries.get(doc_id)
            return entry.metadata if entry else None

    def get_path(self, doc_id: str) -> Path | None:
        """Retrieve the filesystem path for a document.

        Args:
            doc_id: Document identifier.

        Returns:
            ``Path`` to the source PDF, or None if not found.
        """
        meta = self.get_metadata(doc_id)
        return Path(meta.file_path) if meta else None

    def get_status(self, doc_id: str) -> DocumentStatus:
        """Retrieve the current lifecycle status of a document.

        Args:
            doc_id: Document identifier.

        Returns:
            Current ``DocumentStatus``. Returns ``PENDING`` for unknown IDs
            (safe default for callers that don't check existence first).
        """
        with self._lock:
            entry = self._entries.get(doc_id)
            return entry.status if entry else DocumentStatus.PENDING

    def get_unified_document(self, doc_id: str) -> object | None:
        """Retrieve the completed ``UnifiedDocument`` for a document.

        This is the **Phase 4 entry point** for the Knowledge Layer. Phase 4
        components retrieve completed documents from this registry rather than
        re-scanning folders or rebuilding metadata objects.

        Args:
            doc_id: Document identifier.

        Returns:
            The ``UnifiedDocument`` produced by Phase 3, or None if processing
            has not yet completed for this document.
        """
        with self._lock:
            entry = self._entries.get(doc_id)
            return entry.unified_document if entry else None

    def all_metadata(self) -> list[PDFMetadata]:
        """Return ``PDFMetadata`` for all registered documents.

        Returns:
            List of ``PDFMetadata`` records, ordered by document_id.
        """
        with self._lock:
            return [e.metadata for e in self._entries.values()]

    def all_ids(self) -> list[str]:
        """Return all registered document IDs.

        Returns:
            List of document_id strings.
        """
        with self._lock:
            return list(self._entries.keys())

    def pending(self) -> list[PDFMetadata]:
        """Return metadata for all PENDING documents.

        Returns:
            List of ``PDFMetadata`` records with status == PENDING.
        """
        with self._lock:
            return [
                e.metadata
                for e in self._entries.values()
                if e.status == DocumentStatus.PENDING
            ]

    def completed(self) -> list[str]:
        """Return IDs of all COMPLETED documents.

        Returns:
            List of document_id strings with status == COMPLETED.
        """
        with self._lock:
            return [
                doc_id
                for doc_id, e in self._entries.items()
                if e.status == DocumentStatus.COMPLETED
            ]

    def failed(self) -> list[str]:
        """Return IDs of all FAILED documents.

        Returns:
            List of document_id strings with status == FAILED.
        """
        with self._lock:
            return [
                doc_id
                for doc_id, e in self._entries.items()
                if e.status == DocumentStatus.FAILED
            ]

    def __len__(self) -> int:
        """Return the total number of registered documents.

        Returns:
            Count of registered documents.
        """
        with self._lock:
            return len(self._entries)

    def __contains__(self, doc_id: str) -> bool:
        """Check whether a document ID is registered.

        Args:
            doc_id: Document identifier to check.

        Returns:
            True if the document is registered.
        """
        with self._lock:
            return doc_id in self._entries

    # ------------------------------------------------------------------ #
    # Mutations                                                            #
    # ------------------------------------------------------------------ #

    def set_status(self, doc_id: str, status: DocumentStatus) -> None:
        """Update the lifecycle status of a registered document.

        Args:
            doc_id: Document identifier.
            status: New status to apply.

        Raises:
            KeyError: If the document_id is not registered.
        """
        with self._lock:
            if doc_id not in self._entries:
                raise KeyError(
                    f"DocumentRegistry: unknown document_id '{doc_id}'. "
                    "Call register() before set_status()."
                )
            old_status = self._entries[doc_id].status
            self._entries[doc_id].status = status
            logger.debug(
                "Document '%s': %s → %s", doc_id, old_status.value, status.value
            )

    def set_unified_document(self, doc_id: str, unified_document: object) -> None:
        """Store the completed ``UnifiedDocument`` for a document.

        Automatically transitions the document status to ``COMPLETED``.

        Args:
            doc_id: Document identifier.
            unified_document: The ``UnifiedDocument`` produced by Phase 3.

        Raises:
            KeyError: If the document_id is not registered.
        """
        with self._lock:
            if doc_id not in self._entries:
                raise KeyError(
                    f"DocumentRegistry: unknown document_id '{doc_id}'."
                )
            self._entries[doc_id].unified_document = unified_document
            self._entries[doc_id].status = DocumentStatus.COMPLETED
            logger.debug(
                "Document '%s' UnifiedDocument stored; status → COMPLETED.", doc_id
            )

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #

    def summary(self) -> RegistrySummary:
        """Compute aggregate statistics for the current registry state.

        Returns:
            ``RegistrySummary`` populated from the current entries.
        """
        with self._lock:
            entries = list(self._entries.values())

        if not entries:
            return RegistrySummary(metadata_cached=self._metadata_cached)

        total_pages = sum(e.metadata.pages for e in entries)
        total_size = sum(e.metadata.estimated_size_mb for e in entries)
        n = len(entries)

        status_counts: dict[DocumentStatus, int] = {s: 0 for s in DocumentStatus}
        for e in entries:
            status_counts[e.status] += 1

        digital = sum(1 for e in entries if e.metadata.is_digital())
        scanned = sum(1 for e in entries if e.metadata.is_scanned())
        unknown = n - digital - scanned

        return RegistrySummary(
            total_pdfs=n,
            total_pages=total_pages,
            digital_pdfs=digital,
            scanned_pdfs=scanned,
            unknown_pdfs=unknown,
            pending=status_counts[DocumentStatus.PENDING],
            queued=status_counts[DocumentStatus.QUEUED],
            in_progress=status_counts[DocumentStatus.IN_PROGRESS],
            completed=status_counts[DocumentStatus.COMPLETED],
            failed=status_counts[DocumentStatus.FAILED],
            avg_pages=round(total_pages / n, 2) if n else 0.0,
            avg_size_mb=round(total_size / n, 4) if n else 0.0,
            metadata_cached=self._metadata_cached,
        )
