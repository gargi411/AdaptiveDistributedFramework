"""Immutable UnifiedDocument model — the coordinator-side merge output.

The coordinator collects immutable Page objects from all workers and merges
them into a single UnifiedDocument. After creation, the document is frozen.

Architecture rule:
    Workers → Page objects
    Coordinator ONLY → UnifiedDocument
    Workers NEVER build UnifiedDocument.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from adaptive_framework.models.page import (
    FigureData,
    Page,
    PageStatistics,
    TableData,
)


@dataclass(frozen=True)
class DocumentStatistics:
    """Aggregate statistics for a complete UnifiedDocument.

    Attributes:
        total_pages: Total number of pages in the document.
        processed_pages: Pages successfully processed.
        failed_pages: Pages that failed processing.
        digital_pages: Pages processed by direct text extraction.
        scanned_pages: Pages processed by OCR.
        mixed_pages: Pages processed by the mixed strategy.
        total_chars: Total character count across all pages.
        total_words: Total word count across all pages.
        total_tables: Total number of extracted tables.
        total_figures: Total number of extracted figures.
        avg_ocr_confidence: Average OCR confidence (1.0 = all digital).
        total_processing_time_seconds: Sum of all per-page processing times.
        coordinator_merge_time_seconds: Time spent merging pages.
        pages_per_second: Throughput metric (pages / total_wall_time).
    """

    total_pages: int = 0
    processed_pages: int = 0
    failed_pages: int = 0
    digital_pages: int = 0
    scanned_pages: int = 0
    mixed_pages: int = 0
    total_chars: int = 0
    total_words: int = 0
    total_tables: int = 0
    total_figures: int = 0
    avg_ocr_confidence: float = 1.0
    total_processing_time_seconds: float = 0.0
    coordinator_merge_time_seconds: float = 0.0
    pages_per_second: float = 0.0

    @property
    def success_rate(self) -> float:
        """Fraction of pages successfully processed [0.0, 1.0]."""
        if self.total_pages == 0:
            return 0.0
        return self.processed_pages / self.total_pages

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dictionary."""
        return {
            "total_pages": self.total_pages,
            "processed_pages": self.processed_pages,
            "failed_pages": self.failed_pages,
            "digital_pages": self.digital_pages,
            "scanned_pages": self.scanned_pages,
            "mixed_pages": self.mixed_pages,
            "total_chars": self.total_chars,
            "total_words": self.total_words,
            "total_tables": self.total_tables,
            "total_figures": self.total_figures,
            "avg_ocr_confidence": self.avg_ocr_confidence,
            "success_rate": self.success_rate,
            "total_processing_time_seconds": self.total_processing_time_seconds,
            "coordinator_merge_time_seconds": self.coordinator_merge_time_seconds,
            "pages_per_second": self.pages_per_second,
        }


@dataclass(frozen=True)
class DocumentLayout:
    """High-level structure of the entire document.

    Attributes:
        title: Detected document title. None if not found.
        authors: Detected author names.
        abstract: Detected abstract text. None if not found.
        sections: Ordered list of section headings.
        references: Detected bibliography / reference list entries.
        language: Dominant language (ISO 639-1). None if not detected.
        has_toc: True if a table of contents was detected.
    """

    title: str | None = None
    authors: tuple[str, ...] = field(default_factory=tuple)
    abstract: str | None = None
    sections: tuple[str, ...] = field(default_factory=tuple)
    references: tuple[str, ...] = field(default_factory=tuple)
    language: str | None = None
    has_toc: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dictionary."""
        return {
            "title": self.title,
            "authors": list(self.authors),
            "abstract": self.abstract,
            "sections": list(self.sections),
            "references": list(self.references),
            "language": self.language,
            "has_toc": self.has_toc,
        }


@dataclass(frozen=True)
class UnifiedDocument:
    """Immutable aggregated output of the full distributed processing pipeline.

    Built exclusively by UnifiedDocumentBuilder on the coordinator side.
    Workers never create UnifiedDocument — they only produce Page objects.

    After the coordinator calls UnifiedDocumentBuilder.build(), this object
    is returned as a frozen dataclass and never mutated.

    Attributes:
        document_id: Original document identifier (from PDFMetadata).
        file_path: Absolute path to the source PDF.
        pages: Ordered tuple of immutable Page objects (page_number ascending).
        full_text: Complete extracted text, all pages concatenated.
        tables: Tuple of all TableData across all pages.
        figures: Tuple of all FigureData across all pages.
        layout: High-level document structure (title, authors, sections).
        statistics: Aggregate processing statistics.
        processing_logs: Ordered tuple of log messages from all stages.
        created_at: ISO 8601 UTC timestamp of UnifiedDocument creation.
        run_id: Identifier for the pipeline run that produced this document.

    Example:
        >>> doc = UnifiedDocument(
        ...     document_id="doc-001",
        ...     file_path="/data/paper.pdf",
        ...     pages=(),
        ...     full_text="",
        ...     tables=(),
        ...     figures=(),
        ...     layout=DocumentLayout(),
        ...     statistics=DocumentStatistics(),
        ...     processing_logs=(),
        ... )
        >>> doc.statistics.total_pages
        0
    """

    document_id: str
    file_path: str
    pages: tuple[Page, ...]
    full_text: str
    tables: tuple[TableData, ...]
    figures: tuple[FigureData, ...]
    layout: DocumentLayout
    statistics: DocumentStatistics
    processing_logs: tuple[str, ...]
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def get_page(self, page_number: int) -> Page | None:
        """Retrieve a page by its 1-indexed page number.

        Args:
            page_number: 1-indexed page number to retrieve.

        Returns:
            Page object if found, None otherwise.
        """
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None

    def get_tables_for_page(self, page_number: int) -> tuple[TableData, ...]:
        """Return all tables extracted from a specific page.

        Args:
            page_number: 1-indexed page number.

        Returns:
            Tuple of TableData objects for the requested page.
        """
        return tuple(t for t in self.tables if t.page_number == page_number)

    def get_figures_for_page(self, page_number: int) -> tuple[FigureData, ...]:
        """Return all figures extracted from a specific page.

        Args:
            page_number: 1-indexed page number.

        Returns:
            Tuple of FigureData objects for the requested page.
        """
        return tuple(f for f in self.figures if f.page_number == page_number)

    @property
    def success_rate(self) -> float:
        """Fraction of pages successfully processed [0.0, 1.0]."""
        return self.statistics.success_rate

    @property
    def is_fully_processed(self) -> bool:
        """Return True if all pages were successfully processed."""
        return self.statistics.failed_pages == 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain JSON-compatible dictionary.

        Returns:
            Complete dictionary representation of the UnifiedDocument.
        """
        return {
            "document_id": self.document_id,
            "file_path": self.file_path,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "full_text": self.full_text,
            "tables": [t.to_dict() for t in self.tables],
            "figures": [f.to_dict() for f in self.figures],
            "layout": self.layout.to_dict(),
            "statistics": self.statistics.to_dict(),
            "processing_logs": list(self.processing_logs),
            "pages": [p.to_dict() for p in self.pages],
        }

    def __repr__(self) -> str:
        return (
            f"UnifiedDocument(document_id='{self.document_id}', "
            f"pages={len(self.pages)}, "
            f"tables={len(self.tables)}, "
            f"figures={len(self.figures)}, "
            f"success_rate={self.success_rate:.1%})"
        )
