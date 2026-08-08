"""Unified Document Builder — Module 10: Coordinator-side merge.

Architecture rule:
    ONLY the coordinator calls this.
    Workers NEVER call UnifiedDocumentBuilder.
    Workers only produce immutable Page objects.

The coordinator collects all Page objects from distributed workers,
merges them here, and returns one frozen UnifiedDocument.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from adaptive_framework.document_processing.event_bus import EventBus
from adaptive_framework.models.events import coordinator_merge_event
from adaptive_framework.models.page import FigureData, Page, PageType, TableData
from adaptive_framework.models.unified_document import (
    DocumentLayout,
    DocumentStatistics,
    UnifiedDocument,
)

logger = logging.getLogger(__name__)


class UnifiedDocumentBuilder:
    """Merges distributed Page objects into a single immutable UnifiedDocument.

    Called by the coordinator after all workers have returned their Pages.
    Workers never call this class.

    Usage:
        >>> builder = UnifiedDocumentBuilder()
        >>> doc = builder.build(
        ...     document_id="doc-001",
        ...     file_path="/data/paper.pdf",
        ...     pages=collected_pages,
        ...     run_id="run-abc",
        ... )
        >>> doc.statistics.total_pages
        42

    Args:
        event_bus: EventBus to publish COORDINATOR_MERGE and MERGE_COMPLETED events.
    """

    def __init__(self, event_bus: EventBus | None = None) -> None:
        self._bus = event_bus

    def build(
        self,
        document_id: str,
        file_path: str,
        pages: list[Page],
        run_id: str = "adf_run",
    ) -> UnifiedDocument:
        """Build a frozen UnifiedDocument from a list of Page objects.

        Pages are sorted by page_number ascending.
        Statistics are computed from all pages.
        Full text is assembled in page order.
        Tables and figures are collected from all pages.
        Layout (title, sections) is inferred from layout elements.

        Args:
            document_id: Source document identifier.
            file_path: Absolute path to the source PDF.
            pages: List of Page objects from distributed workers.
            run_id: Pipeline run identifier.

        Returns:
            Frozen, immutable UnifiedDocument.
        """
        merge_start = time.perf_counter()

        # Sort pages by page_number
        sorted_pages = sorted(pages, key=lambda p: p.page_number)

        # Aggregate statistics
        stats = self._compute_statistics(sorted_pages, merge_start)

        # Build full text
        full_text = self._build_full_text(sorted_pages)

        # Collect all tables and figures
        all_tables = self._collect_tables(sorted_pages)
        all_figures = self._collect_figures(sorted_pages)

        # Infer document layout
        layout = self._infer_layout(sorted_pages)

        # Collect processing logs
        logs = self._collect_logs(sorted_pages)

        # Finalise statistics with merge time
        merge_time = time.perf_counter() - merge_start
        stats = DocumentStatistics(
            total_pages=stats.total_pages,
            processed_pages=stats.processed_pages,
            failed_pages=stats.failed_pages,
            digital_pages=stats.digital_pages,
            scanned_pages=stats.scanned_pages,
            mixed_pages=stats.mixed_pages,
            total_chars=stats.total_chars,
            total_words=stats.total_words,
            total_tables=stats.total_tables,
            total_figures=stats.total_figures,
            avg_ocr_confidence=stats.avg_ocr_confidence,
            total_processing_time_seconds=stats.total_processing_time_seconds,
            coordinator_merge_time_seconds=merge_time,
            pages_per_second=(
                len(sorted_pages) / stats.total_processing_time_seconds
                if stats.total_processing_time_seconds > 0 else 0.0
            ),
        )

        doc = UnifiedDocument(
            document_id=document_id,
            file_path=file_path,
            pages=tuple(sorted_pages),
            full_text=full_text,
            tables=tuple(all_tables),
            figures=tuple(all_figures),
            layout=layout,
            statistics=stats,
            processing_logs=tuple(logs),
            run_id=run_id,
        )

        # Publish merge completed event
        if self._bus:
            try:
                self._bus.publish(
                    coordinator_merge_event(
                        document_id=document_id,
                        page_count=len(sorted_pages),
                        merge_time_seconds=merge_time,
                    )
                )
            except Exception:
                pass

        logger.info(
            "UnifiedDocument built: doc=%s, pages=%d, tables=%d, "
            "figures=%d, merge_time=%.3fs",
            document_id, len(sorted_pages),
            len(all_tables), len(all_figures), merge_time,
        )

        return doc

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _compute_statistics(
        pages: list[Page],
        merge_start: float,
    ) -> DocumentStatistics:
        """Compute aggregate statistics across all pages.

        Args:
            pages: All processed pages.
            merge_start: perf_counter value at merge start.

        Returns:
            Partial DocumentStatistics (merge time added later).
        """
        total_pages = len(pages)
        processed = sum(1 for p in pages if p.success)
        failed = total_pages - processed

        digital = sum(1 for p in pages if p.page_type == PageType.DIGITAL)
        scanned = sum(1 for p in pages if p.page_type == PageType.SCANNED)
        mixed = sum(1 for p in pages if p.page_type == PageType.MIXED)

        total_chars = sum(p.statistics.char_count for p in pages)
        total_words = sum(p.statistics.word_count for p in pages)
        total_tables = sum(p.statistics.table_count for p in pages)
        total_figures = sum(p.statistics.figure_count for p in pages)
        total_proc_time = sum(p.processing_time_seconds for p in pages)

        confidences = [p.ocr_confidence for p in pages if p.success]
        avg_conf = sum(confidences) / len(confidences) if confidences else 1.0

        return DocumentStatistics(
            total_pages=total_pages,
            processed_pages=processed,
            failed_pages=failed,
            digital_pages=digital,
            scanned_pages=scanned,
            mixed_pages=mixed,
            total_chars=total_chars,
            total_words=total_words,
            total_tables=total_tables,
            total_figures=total_figures,
            avg_ocr_confidence=round(avg_conf, 4),
            total_processing_time_seconds=round(total_proc_time, 4),
        )

    @staticmethod
    def _build_full_text(pages: list[Page]) -> str:
        """Concatenate all page texts in order.

        Args:
            pages: Pages sorted by page_number.

        Returns:
            Full document text with page separators.
        """
        parts: list[str] = []
        for page in pages:
            if page.text:
                parts.append(page.text)
        return "\n\n".join(parts)

    @staticmethod
    def _collect_tables(pages: list[Page]) -> list[TableData]:
        """Collect all tables from all pages.

        Args:
            pages: All processed pages.

        Returns:
            Flat list of TableData sorted by page_number.
        """
        tables: list[TableData] = []
        for page in pages:
            tables.extend(page.tables)
        return tables

    @staticmethod
    def _collect_figures(pages: list[Page]) -> list[FigureData]:
        """Collect all figures from all pages.

        Args:
            pages: All processed pages.

        Returns:
            Flat list of FigureData sorted by page_number.
        """
        figures: list[FigureData] = []
        for page in pages:
            figures.extend(page.figures)
        return figures

    @staticmethod
    def _infer_layout(pages: list[Page]) -> DocumentLayout:
        """Infer document-level layout from page layout elements.

        Finds title (first H1 heading), sections (all headings),
        and language (from classification).

        Args:
            pages: All processed pages.

        Returns:
            DocumentLayout with inferred structure.
        """
        title: str | None = None
        sections: list[str] = []
        language: str | None = None

        for page in pages:
            for elem in page.layout_elements:
                if elem.element_type == "heading":
                    if title is None and elem.level == 1:
                        title = elem.text
                    sections.append(elem.text)

        return DocumentLayout(
            title=title,
            sections=tuple(sections[:50]),  # cap at 50 section headings
            language=language,
        )

    @staticmethod
    def _collect_logs(pages: list[Page]) -> list[str]:
        """Collect all warnings and error messages from all pages.

        Args:
            pages: All processed pages.

        Returns:
            Ordered list of log messages.
        """
        logs: list[str] = []
        for page in pages:
            for warning in page.warnings:
                logs.append(f"[page {page.page_number}] WARNING: {warning}")
            if page.error_message:
                logs.append(
                    f"[page {page.page_number}] ERROR: {page.error_message}"
                )
        return logs
