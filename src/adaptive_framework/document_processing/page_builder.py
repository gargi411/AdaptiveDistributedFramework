"""Page Object Builder — Module 9: Orchestrates the full per-page pipeline.

Each distributed worker calls PageObjectBuilder for every page in its
assigned PageWorkUnit. The builder:
    1. Loads the page via ZeroCopyPageLoader
    2. Classifies the page type (digital/scanned/mixed)
    3. Selects the correct ProcessingStrategy
    4. Runs layout analysis (Docling/heuristic)
    5. Extracts tables and figures
    6. Validates the result (PageValidator)
    7. Emits instrumentation metrics
    8. Publishes events to the EventBus
    9. Returns one immutable Page object

Workers never call UnifiedDocumentBuilder.
Workers always return a single Page per page processed.
"""

from __future__ import annotations

import logging
import socket
import time
from typing import Any

from adaptive_framework.document_processing.document_type_detector import (
    DocumentTypeDetector,
)
from adaptive_framework.document_processing.event_bus import EventBus
from adaptive_framework.document_processing.figure_detection import FigureExtractor
from adaptive_framework.document_processing.instrumentation import ProcessingInstrument
from adaptive_framework.document_processing.layout_analysis import (
    DoclingLayoutAnalyser,
)
from adaptive_framework.document_processing.page_validator import PageValidator
from adaptive_framework.document_processing.processing_strategy import (
    ProcessingStrategyFactory,
)
from adaptive_framework.document_processing.table_extraction import TableExtractor
from adaptive_framework.models.events import (
    EventType,
    ProcessingEvent,
    page_finished_event,
    page_started_event,
)
from adaptive_framework.models.page import (
    FigureData,
    LayoutElement,
    Page,
    PageStatistics,
    PageType,
    ProcessingMethod,
    TableData,
    TextBlock,
)

logger = logging.getLogger(__name__)


def _processing_method_from_str(method_str: str) -> ProcessingMethod:
    """Convert strategy name string to ProcessingMethod enum.

    Args:
        method_str: Strategy name ('direct_text', 'ocr', 'mixed', etc.).

    Returns:
        Corresponding ProcessingMethod enum value.
    """
    mapping = {
        "direct_text": ProcessingMethod.DIRECT_TEXT,
        "ocr": ProcessingMethod.OCR,
        "mixed": ProcessingMethod.MIXED,
        "skipped": ProcessingMethod.SKIPPED,
        "failed": ProcessingMethod.FAILED,
    }
    return mapping.get(method_str, ProcessingMethod.FAILED)


class PageObjectBuilder:
    """Orchestrates the complete per-page processing pipeline.

    Instantiated once per worker. Processes each page in the assigned
    PageWorkUnit through all extraction stages and returns an immutable Page.

    Args:
        worker_id: Unique identifier of the owning worker.
        node_id: Hostname of the worker's node. Auto-detected if None.
        strategy_factory: Factory for selecting processing strategies.
        layout_analyser: Docling or heuristic layout analyser.
        table_extractor: Table extraction component.
        figure_extractor: Figure/image extraction component.
        page_validator: Output validation component.
        event_bus: EventBus for publishing processing events. None = no events.
        ocr_dpi: DPI for rasterising scanned pages.
        ocr_lang: Language code for OCR.
    """

    def __init__(
        self,
        worker_id: str = "worker-0",
        node_id: str | None = None,
        strategy_factory: ProcessingStrategyFactory | None = None,
        layout_analyser: DoclingLayoutAnalyser | None = None,
        table_extractor: TableExtractor | None = None,
        figure_extractor: FigureExtractor | None = None,
        page_validator: PageValidator | None = None,
        event_bus: EventBus | None = None,
        ocr_dpi: int = 150,
        ocr_lang: str = "en",
    ) -> None:
        self._worker_id = worker_id
        self._node_id = node_id or socket.gethostname()
        self._factory = strategy_factory or ProcessingStrategyFactory(
            ocr_dpi=ocr_dpi, ocr_lang=ocr_lang
        )
        self._layout = layout_analyser or DoclingLayoutAnalyser()
        self._tables = table_extractor or TableExtractor()
        self._figures = figure_extractor or FigureExtractor()
        self._validator = page_validator or PageValidator()
        self._detector = DocumentTypeDetector()
        self._bus = event_bus

    def build(
        self,
        fitz_page: Any,
        page_number: int,
        document_id: str,
        file_path: str,
        width_pts: float = 0.0,
        height_pts: float = 0.0,
        retry_count: int = 0,
    ) -> Page:
        """Build one immutable Page from a PyMuPDF page object.

        Args:
            fitz_page: Open fitz.Page object (zero-copy, no disk writes).
            page_number: 1-indexed page number.
            document_id: Parent document identifier.
            file_path: Absolute path to the source PDF.
            width_pts: Page width in PDF points (from PDF metadata).
            height_pts: Page height in PDF points.
            retry_count: Number of times this page was retried.

        Returns:
            Immutable Page object.
        """
        instrument = ProcessingInstrument(
            document_id=document_id,
            page_number=page_number,
            worker_id=self._worker_id,
            node_id=self._node_id,
        )

        # ── Classify page type ────────────────────────────────────────────
        classification = self._detector.classify_page_from_fitz(
            fitz_page, page_number
        )
        page_type = classification.page_type

        # ── Emit PAGE_STARTED event ───────────────────────────────────────
        self._publish(
            page_started_event(
                document_id=document_id,
                page_number=page_number,
                worker_id=self._worker_id,
                node_id=self._node_id,
                processing_method=page_type.value,
            )
        )

        # ── Select strategy and extract ───────────────────────────────────
        strategy = self._factory.get_strategy(page_type)
        try:
            with instrument.stage("text_extraction" if page_type == PageType.DIGITAL else "ocr"):
                extraction = strategy.process(
                    fitz_page, page_number, document_id, file_path
                )
        except Exception as exc:
            logger.error(
                "Strategy.process raised for page %d of '%s': %s",
                page_number, file_path, exc,
            )
            return self._failure_page(
                document_id, page_number, page_type, str(exc),
                width_pts, height_pts,
            )

        # ── Layout analysis ───────────────────────────────────────────────
        with instrument.stage("layout"):
            layout_result = self._layout.analyse_page(
                fitz_page, page_number, extraction.text_blocks
            )

        self._publish(ProcessingEvent(
            event_type=EventType.LAYOUT_COMPLETED,
            document_id=document_id,
            page_number=page_number,
            worker_id=self._worker_id,
            node_id=self._node_id,
            wall_time_seconds=layout_result.analysis_time_seconds,
        ))

        # ── Table extraction ──────────────────────────────────────────────
        with instrument.stage("table_extraction"):
            table_result = self._tables.extract_page(fitz_page, page_number)

        if table_result.count > 0:
            self._publish(ProcessingEvent(
                event_type=EventType.TABLE_EXTRACTED,
                document_id=document_id,
                page_number=page_number,
                worker_id=self._worker_id,
                payload={"table_count": table_result.count},
            ))

        # ── Figure extraction ─────────────────────────────────────────────
        with instrument.stage("image_extraction"):
            figure_result = self._figures.extract_page(
                fitz_page, page_number, extraction.text_blocks
            )

        if figure_result.count > 0:
            self._publish(ProcessingEvent(
                event_type=EventType.FIGURE_EXTRACTED,
                document_id=document_id,
                page_number=page_number,
                worker_id=self._worker_id,
                payload={"figure_count": figure_result.count},
            ))

        # ── Validation ────────────────────────────────────────────────────
        with instrument.stage("validation"):
            validation = self._validator.validate(
                page_number=page_number,
                text=extraction.text,
                text_blocks=extraction.text_blocks,
                ocr_confidence=extraction.ocr_confidence,
                processing_method=extraction.processing_method,
                image_density=classification.image_density,
            )

        if not validation.passed:
            self._publish(ProcessingEvent(
                event_type=EventType.VALIDATION_FAILED,
                document_id=document_id,
                page_number=page_number,
                worker_id=self._worker_id,
                message=f"Validation failed: {[e.code for e in validation.errors]}",
            ))

        # ── Build instrumentation metrics ─────────────────────────────────
        instrument.set("ocr_confidence", extraction.ocr_confidence)
        instrument.set("char_count", len(extraction.text))
        instrument.set("table_count", table_result.count)
        instrument.set("figure_count", figure_result.count)

        # ── Assemble Page object ──────────────────────────────────────────
        with instrument.stage("page_build"):
            all_warnings = list(extraction.warnings) + validation.warning_messages

            statistics = PageStatistics(
                text_block_count=len(extraction.text_blocks),
                table_count=table_result.count,
                figure_count=figure_result.count,
                char_count=len(extraction.text),
                word_count=len(extraction.text.split()) if extraction.text else 0,
                image_density=classification.image_density,
                has_text_layer=classification.has_text_layer,
                ocr_confidence_avg=extraction.ocr_confidence,
            )

            page = Page(
                document_id=document_id,
                page_number=page_number,
                page_type=page_type,
                processing_method=_processing_method_from_str(
                    extraction.processing_method
                ),
                text=extraction.text,
                text_blocks=tuple(extraction.text_blocks),
                tables=tuple(table_result.tables),
                figures=tuple(figure_result.figures),
                layout_elements=tuple(layout_result.elements),
                worker_id=self._worker_id,
                node_id=self._node_id,
                processing_time_seconds=0.0,  # updated below
                ocr_confidence=extraction.ocr_confidence,
                success=extraction.error is None,
                error_message=extraction.error,
                warnings=tuple(all_warnings),
                statistics=statistics,
                width_pts=width_pts,
                height_pts=height_pts,
            )

        # Build metrics (total wall time captured here)
        metrics = instrument.build_metrics(
            processing_method=extraction.processing_method,
            retry_count=retry_count,
        )

        # Re-create Page with correct total processing time
        # (frozen dataclass — must use object.__setattr__ trick or reconstruct)
        page = Page(
            document_id=page.document_id,
            page_number=page.page_number,
            page_type=page.page_type,
            processing_method=page.processing_method,
            text=page.text,
            text_blocks=page.text_blocks,
            tables=page.tables,
            figures=page.figures,
            layout_elements=page.layout_elements,
            worker_id=page.worker_id,
            node_id=page.node_id,
            processing_time_seconds=metrics.total_wall_time_seconds,
            ocr_confidence=page.ocr_confidence,
            success=page.success,
            error_message=page.error_message,
            warnings=page.warnings,
            statistics=page.statistics,
            width_pts=page.width_pts,
            height_pts=page.height_pts,
        )

        # ── Emit PAGE_FINISHED event ──────────────────────────────────────
        self._publish(
            page_finished_event(
                document_id=document_id,
                page_number=page_number,
                worker_id=self._worker_id,
                node_id=self._node_id,
                wall_time_seconds=metrics.total_wall_time_seconds,
                success=page.success,
                char_count=statistics.char_count,
                table_count=table_result.count,
                figure_count=figure_result.count,
            )
        )

        logger.debug(
            "Built Page(doc=%s, page=%d, method=%s, time=%.3fs)",
            document_id, page_number,
            extraction.processing_method,
            metrics.total_wall_time_seconds,
        )

        return page

    def _failure_page(
        self,
        document_id: str,
        page_number: int,
        page_type: PageType,
        error: str,
        width_pts: float,
        height_pts: float,
    ) -> Page:
        """Return a failed Page when the strategy raises an unhandled exception.

        Args:
            document_id: Document identifier.
            page_number: Page number.
            page_type: Detected page type.
            error: Error message.
            width_pts: Page width.
            height_pts: Page height.

        Returns:
            Page with success=False and error_message set.
        """
        return Page(
            document_id=document_id,
            page_number=page_number,
            page_type=page_type,
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
            error_message=error,
        )

    def _publish(self, event: ProcessingEvent) -> None:
        """Safely publish an event to the bus (no-op if bus is None).

        Args:
            event: ProcessingEvent to publish.
        """
        if self._bus is not None:
            try:
                self._bus.publish(event)
            except Exception as exc:
                logger.debug("EventBus publish failed: %s", exc)
