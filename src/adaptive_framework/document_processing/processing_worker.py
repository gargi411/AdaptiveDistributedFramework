"""Document Processing Worker — Module 13: Integration with distributed framework.

DocumentProcessingWorker is the bridge between Phase 2B (DistributedCoordinator)
and Phase 3 (Document Processing Engine).

Flow:
    DistributedCoordinator
        └── dispatch PageWorkUnit → DocumentProcessingWorker.process_work_unit()
                └── open PDF with PyMuPDF (once per worker per document)
                └── for each page in (start_page, end_page):
                        PageObjectBuilder.build(fitz_page, ...)
                        → immutable Page
                └── emit WORKER_COMPLETED event
                └── return list[Page]

    Coordinator receives list[Page] → UnifiedDocumentBuilder.build()

Architecture rules:
    - Workers receive PageWorkUnit (never complete PDFs)
    - Workers return list[Page] (never UnifiedDocument)
    - UnifiedDocumentBuilder runs only on the coordinator
"""

from __future__ import annotations

import logging
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adaptive_framework.document_processing.event_bus import EventBus
from adaptive_framework.document_processing.page_builder import PageObjectBuilder
from adaptive_framework.models.events import worker_completed_event
from adaptive_framework.models.page import Page
from adaptive_framework.models.scheduling import PageWorkUnit

logger = logging.getLogger(__name__)

try:
    import fitz  # type: ignore[import]
    _FITZ_AVAILABLE = True
except ImportError:
    _FITZ_AVAILABLE = False


@dataclass
class WorkerProcessingResult:
    """Result returned by DocumentProcessingWorker after processing a PageWorkUnit.

    Attributes:
        worker_id: Worker that produced this result.
        document_id: Document that was processed.
        pages: Immutable Page objects produced (one per page processed).
        total_wall_time_seconds: End-to-end wall time for the entire work unit.
        pages_succeeded: Number of pages successfully processed.
        pages_failed: Number of pages that failed.
        error: Top-level error if the work unit itself failed (not page errors).
    """

    worker_id: str
    document_id: str
    pages: list[Page] = field(default_factory=list)
    total_wall_time_seconds: float = 0.0
    pages_succeeded: int = 0
    pages_failed: int = 0
    error: str | None = None

    @property
    def success(self) -> bool:
        """True if work unit completed without a top-level error."""
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dictionary."""
        return {
            "worker_id": self.worker_id,
            "document_id": self.document_id,
            "page_count": len(self.pages),
            "pages_succeeded": self.pages_succeeded,
            "pages_failed": self.pages_failed,
            "total_wall_time_seconds": self.total_wall_time_seconds,
            "success": self.success,
            "error": self.error,
        }


class DocumentProcessingWorker:
    """Processes a PageWorkUnit and returns immutable Page objects.

    Instantiated once per distributed worker process. Holds shared state
    (PageObjectBuilder, open fitz.Document per file) across multiple calls.

    This is the integration point between Phase 2B and Phase 3:
        Phase 2B TaskDispatcher dispatches PageWorkUnit
        → DocumentProcessingWorker.process_work_unit()
        → returns list[Page]
        → Coordinator collects all Pages
        → UnifiedDocumentBuilder builds UnifiedDocument

    Args:
        worker_id: Unique worker identifier.
        node_id: Node hostname. Auto-detected if None.
        event_bus: EventBus for publishing events. None = no events.
        ocr_dpi: DPI for rasterising scanned pages.
        ocr_lang: Language code for OCR.
    """

    def __init__(
        self,
        worker_id: str = "worker-0",
        node_id: str | None = None,
        event_bus: EventBus | None = None,
        ocr_dpi: int = 150,
        ocr_lang: str = "en",
    ) -> None:
        self._worker_id = worker_id
        self._node_id = node_id or socket.gethostname()
        self._bus = event_bus
        self._builder = PageObjectBuilder(
            worker_id=worker_id,
            node_id=self._node_id,
            event_bus=event_bus,
            ocr_dpi=ocr_dpi,
            ocr_lang=ocr_lang,
        )

    def process_work_unit(
        self,
        work_unit: PageWorkUnit,
        retry_count: int = 0,
    ) -> WorkerProcessingResult:
        """Process a PageWorkUnit and return a list of immutable Page objects.

        Opens the PDF once, processes all assigned pages, closes the PDF.
        Each page is processed independently by PageObjectBuilder.

        Args:
            work_unit: The PageWorkUnit assigned by the TaskDispatcher.
            retry_count: Number of retries for this work unit.

        Returns:
            WorkerProcessingResult with list of Page objects.
        """
        wall_start = time.perf_counter()
        doc_id = work_unit.document_id
        file_path = work_unit.file_path
        start_page = work_unit.start_page
        end_page = work_unit.end_page

        logger.info(
            "Worker %s processing pages %d–%d of '%s'",
            self._worker_id, start_page, end_page,
            Path(file_path).name,
        )

        if not _FITZ_AVAILABLE:
            return WorkerProcessingResult(
                worker_id=self._worker_id,
                document_id=doc_id,
                error="PyMuPDF not installed. Cannot process PDF.",
            )

        pages: list[Page] = []
        succeeded = 0
        failed = 0

        try:
            doc = fitz.open(file_path)
        except Exception as exc:
            logger.error(
                "Worker %s: cannot open '%s': %s",
                self._worker_id, file_path, exc,
            )
            return WorkerProcessingResult(
                worker_id=self._worker_id,
                document_id=doc_id,
                error=f"Cannot open PDF: {exc}",
            )

        try:
            total_pages = len(doc)
            for page_number in range(start_page, end_page + 1):
                idx = page_number - 1
                if idx < 0 or idx >= total_pages:
                    logger.warning(
                        "Worker %s: page %d out of range (doc has %d pages).",
                        self._worker_id, page_number, total_pages,
                    )
                    failed += 1
                    continue

                fitz_page = doc[idx]
                rect = fitz_page.rect

                try:
                    page = self._builder.build(
                        fitz_page=fitz_page,
                        page_number=page_number,
                        document_id=doc_id,
                        file_path=file_path,
                        width_pts=rect.width,
                        height_pts=rect.height,
                        retry_count=retry_count,
                    )
                    pages.append(page)
                    if page.success:
                        succeeded += 1
                    else:
                        failed += 1

                except Exception as exc:
                    logger.warning(
                        "Worker %s: page %d raised exception: %s",
                        self._worker_id, page_number, exc,
                    )
                    failed += 1

        finally:
            doc.close()

        total_wall = time.perf_counter() - wall_start

        # Emit WORKER_COMPLETED event
        if self._bus:
            try:
                self._bus.publish(
                    worker_completed_event(
                        worker_id=self._worker_id,
                        node_id=self._node_id,
                        document_id=doc_id,
                        pages_processed=len(pages),
                        total_wall_time_seconds=total_wall,
                    )
                )
            except Exception:
                pass

        logger.info(
            "Worker %s done: %d pages, %d succeeded, %d failed, %.3fs",
            self._worker_id, len(pages), succeeded, failed, total_wall,
        )

        return WorkerProcessingResult(
            worker_id=self._worker_id,
            document_id=doc_id,
            pages=pages,
            total_wall_time_seconds=total_wall,
            pages_succeeded=succeeded,
            pages_failed=failed,
        )

    def process_single_page(
        self,
        file_path: str,
        page_number: int,
        document_id: str,
    ) -> Page:
        """Convenience method: process a single page outside a work unit.

        Useful for testing and interactive use. Opens and closes the PDF
        for a single page — not efficient for batch processing.

        Args:
            file_path: Absolute path to the PDF.
            page_number: 1-indexed page number.
            document_id: Document identifier.

        Returns:
            Immutable Page object.
        """
        if not _FITZ_AVAILABLE:
            from adaptive_framework.models.page import ProcessingMethod
            return Page(
                document_id=document_id,
                page_number=page_number,
                page_type=__import__(
                    "adaptive_framework.models.page", fromlist=["PageType"]
                ).PageType.UNKNOWN,
                processing_method=ProcessingMethod.FAILED,
                text="",
                text_blocks=(),
                tables=(),
                figures=(),
                layout_elements=(),
                worker_id=self._worker_id,
                node_id=self._node_id,
                processing_time_seconds=0.0,
                ocr_confidence=0.0,
                success=False,
                error_message="PyMuPDF not installed.",
            )

        doc = fitz.open(file_path)
        try:
            idx = page_number - 1
            fitz_page = doc[idx]
            rect = fitz_page.rect
            return self._builder.build(
                fitz_page=fitz_page,
                page_number=page_number,
                document_id=document_id,
                file_path=file_path,
                width_pts=rect.width,
                height_pts=rect.height,
            )
        finally:
            doc.close()
